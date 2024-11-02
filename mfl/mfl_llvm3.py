"""
LLVM IR Generator for curried functions using llvmlite.

This module generates LLVM IR code from AST nodes, with special handling for
curried functions using closure state to hold captured variables.
"""

from llvmlite import ir
import llvmlite.binding as llvm
from typing import Dict, Optional, Tuple, Any

from mfl_ast import (
    ASTNode, Var, Function, Apply, Let, Int, Bool, BinOp, UnaryOp
)
from mfl_type_checker import (
    TyVar, TyCon
)

import builtins
import inspect

# Override the print function to include line number
def print(*args, **kwargs):
    frame = inspect.currentframe().f_back
    builtins.print(f"<{frame.f_lineno}>: ", end="")
    builtins.print(*args, **kwargs)

# Initialize LLVM
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

class LLVMGenerator:
    """
    Generates LLVM IR code from AST nodes.
    Handles curried functions by using closure state to hold captured variables.
    """
    def __init__(self, verbose=False, generate_comments=True):
        # Create module to hold IR code
        self.module = ir.Module(name="curried_functions")
        self.module.triple = llvm.get_default_triple()

        # Declare the printf function
        self.printf_ty = ir.FunctionType(ir.IntType(32), [ir.PointerType(ir.IntType(8))], var_arg=True)
        self.printf = ir.Function(self.module, self.printf_ty, name="printf")
       
        # Define some useful string constants
        self.str_int = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 3), name=".str.int")
        self.str_int.global_constant = True
        self.str_int.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 3), bytearray(b"%d\00"))

        self.str_bool = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 3), name=".str.bool")
        self.str_bool.global_constant = True
        self.str_bool.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 3), bytearray(b"%s\00"))

        self.str_true = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 5), name=".str.true")
        self.str_true.global_constant = True
        self.str_true.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 5), bytearray(b"true\00"))

        self.str_false = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 6), name=".str.false")
        self.str_false.global_constant = True
        self.str_false.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 6), bytearray(b"false\00"))

        
        # Current IR builder
        self.builder: Optional[ir.IRBuilder] = None
        
        # Track current function being generated
        self.current_function: Optional[ir.Function] = None
        
        # Variable scope management
        self.variables: Dict[str, Tuple[ir.Value, ir.Type]] = {}
        
        # Counter for generating unique names
        self.fresh_counter = 0
        
        # Debug output flag
        self.verbose = verbose
        self.generate_comments = generate_comments
        
        # Basic types we'll use
        self.int_type = ir.IntType(32)
        self.bool_type = ir.IntType(1)
        self.void_type = ir.VoidType()
        
        # Type for closure state (holds captured variables)
        self.state_type = ir.LiteralStructType([self.int_type] * 3)  # Can hold 3 ints
        self.state_ptr_type = ir.PointerType(self.state_type)

    def debug(self, msg: str) -> None:
        """Print debug message if verbose mode is enabled"""
        if self.verbose:
            print(f"LLVM: {msg}")

    def comment(self, msg: str) -> None:
        """Print debug message if verbose mode is enabled"""
        if self.generate_comments:
            self.builder.comment(msg)

    def fresh_name(self, prefix: str = "") -> str:
        """Generate a unique name"""
        name = f"{prefix}_{self.fresh_counter}"
        self.fresh_counter += 1
        return name

    def generate(self, node: ASTNode) -> Tuple[ir.Value, ir.Type]:
        """
        Generate LLVM IR for an AST node.
        Returns (value, type) tuple.
        """
        self.debug(f"Generating code for {type(node).__name__}")
        
        # Initialize builder if not already done
        if self.builder is None:
            # Create main function
            main_type = ir.FunctionType(self.int_type, [])
            main_func = ir.Function(self.module, main_type, name="main")
            block = main_func.append_basic_block(name="entry")
            self.builder = ir.IRBuilder(block)
            self.current_function = main_func

        if isinstance(node, Int):
            return ir.Constant(self.int_type, node.value), self.int_type

        elif isinstance(node, Bool):
            return ir.Constant(self.bool_type, 1 if node.value else 0), self.bool_type

        elif isinstance(node, Var):
            return self.generate_var(node)

        elif isinstance(node, Function):
            return self.generate_function(node)

        elif isinstance(node, Apply):
            return self.generate_apply(node)

        elif isinstance(node, Let):
            return self.generate_let(node)

        elif isinstance(node, BinOp):
            return self.generate_binop(node)

        elif isinstance(node, UnaryOp):
            return self.generate_unaryop(node)

        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def generate_function(self, node: Function) -> Tuple[ir.Value, ir.Type]:
        """
        Generate a curried function.
        Returns the function and its type.
        """

        # Save current state
        old_builder = self.builder
        old_vars = self.variables.copy()
        old_func = self.current_function

        # Initialize builder if not already done
        if self.builder is None:
            block = func.append_basic_block(name="entry")
            self.builder = ir.IRBuilder(block)

        if isinstance(node.body, Function):
            # Create function type that always returns a function pointer
            inner_func_type = ir.FunctionType(self.int_type, 
                                        [self.state_ptr_type, self.int_type])
            return_type = ir.PointerType(inner_func_type)
            func_type = ir.FunctionType(return_type, 
                                  [self.state_ptr_type, self.int_type])

            # Create function
            func_name = self.fresh_name("func")
            func = ir.Function(self.module, func_type, name=func_name)

            self.debug(f"Generating function {func.name} of type: {node.type.args} with arg: {node.arg.name}")

            # Create entry block
            block = func.append_basic_block(name="entry")
            self.builder = ir.IRBuilder(block)
            self.current_function = func

            # Add parameters to scope
            state_ptr, arg = func.args

            # Store argument in closure state
            idx = len(self.variables)
            arg_ptr = self.builder.gep(state_ptr, 
                                 [ir.Constant(self.int_type, 0),
                                  ir.Constant(self.int_type, idx)],
                                 name=f"arg_ptr_{node.arg.name}")
            self.builder.store(arg, arg_ptr)

            # Add stored argument to variables
            self.variables[node.arg.name] = (arg_ptr, self.int_type)

            # Generate next function in curry chain
            next_func, _ = self.generate(node.body)
            self.builder.ret(next_func)
            self.comment(f"Generated curried function({func_name}): {node}")
        else:
            # Create computation function
            print(f"Function {node} with arg: {node.arg}")

            # Figure out the correct return type
            if isinstance(node.type, TyCon):
                type_name = node.type.name
                type_args = node.type.args
                ret_type = type_args[0].find()
                arg_type = type_args[1].find()
            else:
                raise TypeError(f"Wrong function type: {node.type}")
            
            # Set up function types
            func_ret_type = self.int_type if ret_type.name == "int" else self.bool_type
            func_arg_type = self.int_type if arg_type.name == "int" else self.bool_type
            func_type = ir.FunctionType(func_ret_type, 
                                        [self.state_ptr_type, func_arg_type])
        
            # Create function
            func_name = self.fresh_name("comp")
            func = ir.Function(self.module, func_type, name=func_name)
        
            # Set up entry block
            block = func.append_basic_block(name="entry")
            self.builder = ir.IRBuilder(block)
            self.current_function = func

            # Get function parameters
            state_ptr, arg = func.args

            # Store argument with proper type
            idx = len(self.variables)
            arg_ptr = self.builder.gep(state_ptr,
                                    [ir.Constant(self.int_type, 0),
                                    ir.Constant(self.int_type, idx)])
        
            # Store with type conversion if needed
            if arg_type.name != ret_type.name:
                if arg_type.name == "bool" and ret_type.name == "int":
                    arg = self.builder.zext(arg, self.int_type)
                elif arg_type.name == "int" and ret_type.name == "bool":
                    arg = self.builder.icmp_unsigned('!=', arg, 
                                                ir.Constant(self.int_type, 0))
        
            self.builder.store(arg, arg_ptr)
            self.variables[node.arg.name] = (arg_ptr, func_arg_type)

            # Generate body computation
            result, _ = self.generate(node.body)
        
            # Return result
            self.builder.ret(result)
            self.comment(f"Generated final function({func_name}): {node}")

        print(f"Generated function: {node.raw_structure()}\n{func}")
        # Restore state
        self.builder = old_builder
        self.variables = old_vars
        self.current_function = old_func
        return func, func_type

    def generate_apply(self, node: Apply) -> Tuple[ir.Value, ir.Type]:
        """
        Generate function application.
        Returns the result value and type.
        Handles nested applications by processing innermost first.
        """
        self.debug(f"Generating function application: {node.func} {node.arg} : {node.raw_structure()}")

        # Process function value, handling nested applications
        if isinstance(node.func, Apply):
            # Recursively process inner application first
            func_val, func_type = self.generate_apply(node.func)
        elif isinstance(node.func, Var):
            # Look up function from variables
            if node.func.name not in self.variables:
                raise NameError(f"Undefined function: {node.func.name}")
            func_val, func_type = self.variables[node.func.name]
        else:
            raise TypeError(f"No function to be applied {node.raw_structure()}")

        print(f"func_val: {func_val}, func_type: {func_type}")

        # Generate argument
        arg_val, arg_type = self.generate(node.arg)
        self.comment(f"Generated Apply argument: {node.arg}")

        # Always allocate new closure state
        state_ptr = self.builder.alloca(self.state_type)

        # Load function value if needed
        if isinstance(func_val, (ir.GEPInstr, ir.AllocaInstr)):
            func_val = self.builder.load(func_val)

        # Get function type
        if isinstance(func_val.type, ir.PointerType):
            func_ptr_type = func_val.type.pointee
            if isinstance(func_ptr_type, ir.FunctionType):
                # Call the function to get next function pointer
                next_func = self.builder.call(func_val, [state_ptr, arg_val])
                return next_func, next_func.type

        raise TypeError(f"Expected function pointer, got {func_val.type}")

    def generate_let(self, node: Let) -> Tuple[ir.Value, ir.Type]:
        """
        Generate let binding.
        Returns the body value and type.
        """
        self.debug(f"Generating Let binding for {node.name.name}")

        # Generate value
        val, val_type = self.generate(node.value)

        self.comment(f"Generated Let value: '{node.value}' ,return type: {val_type}")

        # For function values, store directly in variables
        if isinstance(val, ir.Function) or (isinstance(val_type, ir.PointerType) and 
                                          isinstance(val_type.pointee, ir.FunctionType)):
            self.variables[node.name.name] = (val, val_type)
        else:
            # For non-function values, store in an alloca
            alloca = self.builder.alloca(val_type)
            self.builder.store(val, alloca)
            self.variables[node.name.name] = (alloca, val_type)

        # Generate body
        result, result_type = self.generate(node.body)

        self.comment(f"Generated Let body: '{node.body}' , return type: {result_type}")

        # Add return if we're in the main function
        if self.current_function.name == "main":
            # Load the value from the result pointer
            #loaded_result = self.builder.load(result, name="final_result")
    
            # Get a pointer to the format string
            str_ptr = self.builder.gep(self.str_int, [ir.Constant(ir.IntType(64), 0), ir.Constant(ir.IntType(64), 0)], inbounds=True, name="str_ptr")
    
            # Call the printf function with the format string and the loaded result
            self.builder.call(self.printf, [str_ptr, result])

            self.builder.ret(result)

        return result, result_type

    def generate_var(self, node: Var) -> Tuple[ir.Value, ir.Type]:
        """
        Generate variable reference.
        Returns the variable value and type.
        """
        self.debug(f"Generating variable reference: {node.name}")

        if node.name not in self.variables:
            raise NameError(f"Undefined variable: {node.name} in {self.variables}")

        val, ty = self.variables[node.name]

        # For function values, return directly
        if isinstance(val, ir.Function):
            self.comment(f"Generate function value, Var: {node}")
            return val, ty
        elif isinstance(ty, ir.PointerType) and isinstance(ty.pointee, ir.FunctionType):
            if isinstance(val, (ir.GEPInstr, ir.AllocaInstr)):
                self.comment(f"Generate func ptr stored in memory, Var: {node}")
                return self.builder.load(val), ty
            self.comment(f"Generate func ptr, Var: {node}")
            return val, ty

        # For values stored in memory
        if isinstance(val, (ir.GEPInstr, ir.AllocaInstr)):
            self.comment(f"Generated value stored in memory, Var: {node}")
            return self.builder.load(val), ty

        # For immediate values
        self.comment(f"Generate immediate value, Var: {node}")
        return val, ty

    def generate_binop(self, node: BinOp) -> Tuple[ir.Value, ir.Type]:
        """
        Generate binary operation.
        Returns the result value and type.
        """
        self.debug(f"Generating binary operation: {node.op}")

        # Generate operands
        left_val, left_type = self.generate(node.left)
        right_val, right_type = self.generate(node.right)

        # Generate operation
        if node.op == '+':
            return self.builder.add(left_val, right_val), self.int_type
        elif node.op == '-':
            return self.builder.sub(left_val, right_val), self.int_type
        elif node.op == '*':
            return self.builder.mul(left_val, right_val), self.int_type
        elif node.op == '/':
            return self.builder.sdiv(left_val, right_val), self.int_type
        elif node.op == '==':
            return self.builder.icmp_signed('==', left_val, right_val), self.bool_type
        elif node.op == '<':
            return self.builder.icmp_signed('<', left_val, right_val), self.bool_type
        elif node.op == '>':
            return self.builder.icmp_signed('>', left_val, right_val), self.bool_type
        else:
            raise ValueError(f"Unknown operator: {node.op}")

    def generate_unaryop(self, node: UnaryOp) -> Tuple[ir.Value, ir.Type]:
        """
        Generate unary operation.
        Returns the result value and type.
        """
        self.debug(f"Generating unary operation: {node.op}")
        
        # Generate operand
        val, ty = self.generate(node.operand)
        
        # Generate operation
        if node.op == '-':
            return self.builder.neg(val), self.int_type
        elif node.op == '!':
            return self.builder.not_(val), self.bool_type
        else:
            raise ValueError(f"Unknown operator: {node.op}")

def main():
    """Test the LLVM generator with a simple curried function"""
    # Create AST for: let add = λx.λy.λz.(x + y + z) in ((add 1) 2) 3
    #expr_str = "let add = λx.λy.λz.(x + y + z) in ((add 1) 2) 3"
    #expr_str = "let add = 3 + 4 in add"
    expr_str = "let id = λx.(x) in (id 8)"

    from mfl_ply_parser import parser as ply_parser
    ast = ply_parser.parse(expr_str)

    from mfl_type_checker import infer_j
    type_ctx = {}  # Empty typing context

    # This will annotate the AST with types!
    infer_j(ast, type_ctx)
    print(f"AST(typed): {ast.typed_structure()}")

    # Generate code
    generator = LLVMGenerator(verbose=True, generate_comments=True)
    result, rt = generator.generate(ast)

    print(f"Generated result: {result} ,rt: {rt}")

    # Print generated IR
    print(f"\nGenerated LLVM IR for: '{ast}'")
    print(f"AST(typed): '{ast.typed_structure()}'\n")
    llvm_ir = str(generator.module)
    
    # Verify module
    print(llvm_ir)
    llvm.parse_assembly(llvm_ir)
    print("\nModule verification successful!")

    print(llvm_ir)

    # Write the generated code to file
    ll_file = "mfl.ll"
    with open(ll_file, "w") as f:
        f.write(llvm_ir)
    print(f"\nGenerated LLVM IR code written to: {ll_file}")

    

if __name__ == "__main__":
    main()
