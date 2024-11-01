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

# Initialize LLVM
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

class LLVMGenerator:
    """
    Generates LLVM IR code from AST nodes.
    Handles curried functions by using closure state to hold captured variables.
    """
    def __init__(self, verbose=False):
        # Create module to hold IR code
        self.module = ir.Module(name="curried_functions")
        self.module.triple = llvm.get_default_triple()
        
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

    def generate_function(self, node: Function) -> Tuple[ir.Function, ir.Type]:
        """
        Generate a curried function.
        Returns the function and its type.
        """
        self.debug(f"Generating function with arg: {node.arg.name}")
        
        # Save current state
        old_builder = self.builder
        old_vars = self.variables.copy()
        
        # Check if this is a curried function (has nested Function in body)
        is_curried = isinstance(node.body, Function)
        
        if is_curried:
            # This function returns a pointer to the next function
            inner_func_type = ir.FunctionType(self.int_type, 
                                            [self.state_ptr_type, self.int_type])
            return_type = ir.PointerType(inner_func_type)
        else:
            # This is the computation function that returns an int
            return_type = self.int_type
            
        # Create function type
        func_type = ir.FunctionType(return_type, 
                                  [self.state_ptr_type, self.int_type])
        
        # Create function
        func = ir.Function(self.module, func_type, 
                         name=self.fresh_name("func"))
        
        # Create entry block
        block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        self.current_function = func
        
        # Add parameters to scope
        state_ptr, arg = func.args
        self.variables[node.arg.name] = (arg, self.int_type)
        
        if is_curried:
            # Store argument in closure state
            idx = len(self.variables) - 1
            arg_ptr = self.builder.gep(state_ptr, 
                                     [ir.Constant(self.int_type, 0),
                                      ir.Constant(self.int_type, idx)],
                                     name=f"arg_ptr_{node.arg.name}")
            self.builder.store(arg, arg_ptr)
            
            # Generate next function
            next_func, _ = self.generate(node.body)
            self.builder.ret(next_func)
        else:
            # Generate computation
            result, _ = self.generate(node.body)
            self.builder.ret(result)
        
        # Restore state
        self.builder = old_builder
        self.variables = old_vars
        self.current_function = None
        
        return func, func_type

    def generate_apply(self, node: Apply) -> Tuple[ir.Value, ir.Type]:
        """
        Generate function application.
        Returns the result value and type.
        """
        self.debug("Generating function application")
        
        # Generate function and argument
        func_val, func_type = self.generate(node.func)
        arg_val, arg_type = self.generate(node.arg)
        
        # Always allocate new closure state
        state_ptr = self.builder.alloca(self.state_type)
        
        # Handle different function types
        if isinstance(func_val, ir.Function):
            # Direct function reference
            result = self.builder.call(func_val, [state_ptr, arg_val])
            return result, result.type
            
        elif isinstance(func_type, ir.PointerType):
            # Load function pointer if needed
            if isinstance(func_type.pointee, ir.FunctionType):
                func_ptr = func_val
            else:
                # Keep loading until we get a function pointer
                func_ptr = func_val
                while isinstance(func_ptr.type, ir.PointerType) and \
                      not isinstance(func_ptr.type.pointee, ir.FunctionType):
                    func_ptr = self.builder.load(func_ptr)
                
            if not isinstance(func_ptr.type, ir.PointerType) or \
               not isinstance(func_ptr.type.pointee, ir.FunctionType):
                raise TypeError(f"Expected function pointer, got: {func_ptr.type}")
                
            # Call the function
            result = self.builder.call(func_ptr, [state_ptr, arg_val])
            return result, result.type
            
        else:
            raise TypeError(f"Expected function or function pointer, got: {func_type}")

    def generate_let(self, node: Let) -> Tuple[ir.Value, ir.Type]:
        """
        Generate let binding.
        Returns the body value and type.
        """
        self.debug(f"Generating let binding for {node.name.name}")
        
        # Generate value
        val, val_type = self.generate(node.value)
        
        # Add to scope
        self.variables[node.name.name] = (val, val_type)
        
        # Generate body
        result, result_type = self.generate(node.body)
        
        # Add return if we're in the main function
        if self.current_function.name == "main":
            self.builder.ret(result)
            
        return result, result_type

    def generate_var(self, node: Var) -> Tuple[ir.Value, ir.Type]:
        """
        Generate variable reference.
        Returns the variable value and type.
        """
        self.debug(f"Generating variable reference: {node.name}")
        
        if node.name not in self.variables:
            raise NameError(f"Undefined variable: {node.name}")
            
        val, ty = self.variables[node.name]
        
        # For function values
        if isinstance(val, ir.Function):
            # Return function pointer type for curried functions
            if len(val.function_type.args) == 2:  # state_ptr and one arg
                return val, val.type
                
        # For values stored in closure state
        if isinstance(val, ir.GEPInstr):
            loaded_val = self.builder.load(val)
            return loaded_val, ty
            
        # For local variables that need loading
        if isinstance(val, ir.AllocaInstr):
            loaded_val = self.builder.load(val)
            return loaded_val, ty
            
        # For function pointers, return as is
        if isinstance(ty, ir.PointerType) and isinstance(ty.pointee, ir.FunctionType):
            return val, ty
            
        # For immediate values
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
    ast = Let(
        Var("add"),
        Function(
            Var("x"),
            Function(
                Var("y"), 
                Function(
                    Var("z"),
                    BinOp(op = "+",
                        left = BinOp(op = "+", left = Var("x"), right = Var("y")),
                        right = Var("z")
                    )
                )
            )
        ),
        Apply(
            Apply(
                Apply(Var("add"), Int(1)),
                Int(2)
            ),
            Int(3)
        )
    )
    
    # Generate code
    generator = LLVMGenerator(verbose=True)
    result, _ = generator.generate(ast)
    
    # Print generated IR
    print("\nGenerated LLVM IR:")
    print(str(generator.module))
    
    # Verify module
    llvm.parse_assembly(str(generator.module))
    print("\nModule verification successful!")

if __name__ == "__main__":
    main()
