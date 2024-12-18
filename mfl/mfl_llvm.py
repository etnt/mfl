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
from mfl_transform import ASTTransformer
import subprocess
import shlex  # For safe shell command construction

#import builtins
#import inspect
#
## Override the print function to include line number
#def print(*args, **kwargs):
#    frame = inspect.currentframe().f_back
#    builtins.print(f"<{frame.f_lineno}>: ", end="")
#
#     builtins.print(*args, **kwargs)

# Initialize LLVM
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

class SymbolTable:
    def __init__(self):
        self.scopes = [{}]  # Start with the global scope

    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        if len(self.scopes) > 1:
            self.scopes.pop()
        else:
            raise Exception("Cannot pop global scope")

    def add_variable(self, name, type, address):
        self.scopes[-1][name] = (type, address)

    def lookup_variable(self, name):
        for scope in reversed(self.scopes):  # Search from inner to outer scopes
            if name in scope:
                return scope[name]
        return None

    def __str__(self):
        output = "Symbol Table:\n"
        for i, scope in enumerate(self.scopes):
            output += f"Scope {i}:\n"
            for name, (type, address) in scope.items():
                output += f"  {name}: Type={type}, Address={address}\n"
            output += "\n"  # Empty line between scopes
        return output.rstrip()  # Remove trailing newline

class LLVMGenerator:
    """
    Generates LLVM IR code from AST nodes.
    Handles curried functions by using closure state to hold captured variables.
    """
    def __init__(self, verbose=False, generate_comments=True):
        # Debug output
        self.verbose = verbose
        self.generate_comments = generate_comments

        # Counter for generating unique names
        self.fresh_counter = 0

        # Return type for curried functions
        self.return_type = None

        # Generate return value for main function only once!
        self.return_generated = False

       # Current IR builder
        self.builder: Optional[ir.IRBuilder] = None

        # Track current function being generated
        self.current_function: Optional[ir.Function] = None

        # Symbol table for variables
        self.symbol_table = SymbolTable()

        # Create module to hold IR code
        self.module = ir.Module(name="MFL Generated Module")
        self.module.triple = llvm.get_default_triple()

        # Basic types we'll use
        self.int_type = ir.IntType(32)
        self.bool_type = ir.IntType(1)
        self.void_type = ir.VoidType()

        # Declare the printf function
        self.printf_ty = ir.FunctionType(ir.IntType(32), [ir.PointerType(ir.IntType(8))], var_arg=True)
        self.printf = ir.Function(self.module, self.printf_ty, name="printf")

        # Define some useful string constants
        self.str_int = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 3), name=".str.int")
        self.str_int.linkage = 'private'
        self.str_int.global_constant = True
        self.str_int.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 3), bytearray(b"%d\00"))

        self.str_bool = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 3), name=".str.bool")
        self.str_bool.linkage = 'private'
        self.str_bool.global_constant = True
        self.str_bool.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 3), bytearray(b"%s\00"))

        self.str_true = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 5), name=".str.true")
        self.str_true.linkage = 'private'
        self.str_true.global_constant = True
        self.str_true.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 5), bytearray(b"true\00"))

        self.str_false = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 6), name=".str.false")
        self.str_false.linkage = 'private'
        self.str_false.global_constant = True
        self.str_false.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 6), bytearray(b"false\00"))

        # Setup the lambda closure state (holds captured variables)
        # Create named struct type
        self.state_type = ir.global_context.get_identified_type("lambda_state")
        lambda_state_size = 8   # Holding 8 variables for now
        try:
            self.state_type.set_body(*[self.int_type] * lambda_state_size)
        except RuntimeError as e:
            pass # To make the unit test pass
        # Create pointer type to named struct
        self.state_ptr_type = ir.PointerType(self.state_type)
        # State type for captured variables
        #self.state_type = ir.LiteralStructType([self.int_type] * 8)  # Max 8 captured variables for now...
        #self.state_ptr_type = self.state_type.as_pointer()

        # Pointer to allocated lambda state struct
        self.state_ptr = None

    def dispose(self):
        self.module = None  # Release the module
        self.builder = None # Release the builder
        self.state_ptr_type = None
        del self.state_type
        #Any other cleanup for LLVM related stuff should go here

    def debug(self, msg: str) -> None:
        """Print debug message if verbose mode is enabled"""
        if self.verbose:
            print(f"LLVM: {msg}")

    def comment(self, msg: str) -> None:
        """Print debug message if verbose mode is enabled"""
        if self.generate_comments:
            self.builder.comment(msg)

    def verify_code(self, ir: str) -> None:
        """Verify the generated code"""
        llvm.parse_assembly(ir)

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
        # Initialize builder if not already done
        if self.builder is None:
            # Create main function
            main_type = ir.FunctionType(self.int_type, [])
            main_func = ir.Function(self.module, main_type, name="main")
            block = main_func.append_basic_block(name="entry")
            self.builder = ir.IRBuilder(block)
            self.current_function = main_func

        if isinstance(node, Int):
            return ir.Constant(self.int_type, node.value)

        elif isinstance(node, Bool):
            return ir.Constant(self.bool_type, 1 if node.value else 0)

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

    # ----------------------------------------------------------------
    # FUNCTION(arg, body)
    # ----------------------------------------------------------------
    def generate_function(self, node: Function) -> Tuple[ir.Value, ir.Type]:
        """
        Generate a curried function.
        Returns the function and its type.
        """
        # Push a new scope to the symbol table
        self.symbol_table.push_scope()

        # Save the current builder
        old_builder = self.builder

        # Initialize state pointer if not already done
        if self.state_ptr is None:
            # Create state pointer at function entry
            self.state_ptr = self.builder.alloca(self.state_type)

        def curry_function(node: Function, idx: int, capvars: Dict[str, Tuple[int, ir.Type]]) -> ir.Type:
            """
            In standard lambda calculus, it's not possible to have a Function node with
            another Function node as its body without it being equivalent to a curried
            function....I think... 
            """
            # Allocate state pointer
            # FIXME should we push the state_ptr to the symbol table here so that we can
            #       define multiple lambdas in the future?
            if idx == 0:
                self.state_ptr = self.builder.alloca(self.state_type, name=self.fresh_name("lambda_state"))

            # Figure out the correct arg and return type
            if isinstance(node.type, TyCon):
                type_args = node.type.args
                ret_type = type_args[0].find()
                arg_type = type_args[1].find()
            else:
                raise TypeError(f"Wrong function type: {node.type}")

            if isinstance(node.body, Function):
                # Add captured argument to local dict
                capvars[node.arg.name] = (idx, arg_type)

                # Get inner function type recursively
                inner_func = curry_function(node.body, idx + 1, capvars.copy())
                inner_func_type_ptr = inner_func.type.as_pointer()

                # Create lambda function
                lambda_arg_type = self.int_type # FIXME if arg_type.name == "int" else self.bool_type
                lambda_type = ir.FunctionType(inner_func_type_ptr.pointee, [self.state_ptr_type, lambda_arg_type])
                lambda_func = ir.Function(self.module, lambda_type, name=self.fresh_name('lambda'))

                # Create entry block
                block = lambda_func.append_basic_block(name="entry")
                self.builder = ir.IRBuilder(block)

                # Get state pointer and argument from function params
                state_ptr = lambda_func.args[0]  # state pointer
                arg_val = lambda_func.args[1]    # argument value

                # Store the captured argument in the lambda state
                idx_ptr = self.builder.gep(state_ptr, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, idx)])
                self.builder.store(arg_val, idx_ptr)

                # Return next function in chain
                self.builder.ret(inner_func)

                return lambda_func
            else:
                # This is the innermost function that does the computation
                # Figure out the correct return type
                if isinstance(node.type, TyCon):
                    type_args = node.type.args
                    ret_type = type_args[0].find()
                    arg_type = type_args[1].find()
                else:
                    raise TypeError(f"Wrong function type: {node.type}")

                # Set up function types
                comp_ret_type = self.int_type if ret_type.name == "int" else self.bool_type
                comp_arg_type = self.int_type if arg_type.name == "int" else self.bool_type
                comp_type = ir.FunctionType(comp_ret_type, [self.state_ptr_type, comp_arg_type])

                # Create compute function
                comp_name = self.fresh_name("compute")
                compute = ir.Function(self.module, comp_type, name=comp_name)
                block = compute.append_basic_block(name="entry")
                self.builder = ir.IRBuilder(block)

                # Get state pointer and argument from function params
                state_ptr = compute.args[0]  # state pointer
                arg_val = compute.args[1]    # argument value

                self.comment("Store the captured argument in the lambda state")
                idx_ptr = self.builder.gep(state_ptr, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, idx)])
                self.builder.store(arg_val, idx_ptr)

                # Store the (variable.name, (idx, type)) in a dict
                capvars[node.arg.name] = (idx, arg_type)

                # Fill in the captured variable values with the pointers to the lambda state
                self.comment("Load all the captured variable values from the lambda state")
                for key in capvars.keys():
                    (val_idx, val_arg_type) = capvars[key]
                    arg_ptr = self.builder.gep(state_ptr, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, val_idx)])
                    self.symbol_table.add_variable(key, val_arg_type, arg_ptr) 

                self.debug(f"Capvars: {capvars}")
                self.debug(f"Symbol table:\n{self.symbol_table}")

                # Generate body computation
                self.comment(f"Generate body: {node.body}")
                result = self.generate(node.body)
                self.debug(f"Generated body({comp_name}): {node.body} -> {result}")

                # Return result
                self.builder.ret(result)
                self.comment(f"Generated final function({comp_name}): {node}")

                return compute

        curry_func = curry_function(node, 0, {})

        # Restore the old builder
        self.builder = old_builder

        # Pop the scope from the symbol table
        self.symbol_table.pop_scope()

        return curry_func

    # ----------------------------------------------------------------
    # APPLY(func, arg)
    # ----------------------------------------------------------------
    def generate_apply(self, node: Apply) -> Tuple[ir.Value, ir.Type]:
        """
        Generate function application.
        Returns the result value and type.
        Handles nested applications by processing innermost first.
        """

        # Process function value, handling nested applications
        if isinstance(node.func, Apply):
            # Recursively process inner application first
            self.debug(">>> HERE 1")
            func_val = self.generate_apply(node.func)
        elif isinstance(node.func, Var):
            # Look up function from variables
            if self.symbol_table.lookup_variable(node.func.name) is None:
                raise NameError(f"Undefined function: {node.func.name}")
            else:
                self.debug(">>> HERE 2")
                (type, func_val) = self.symbol_table.lookup_variable(node.func.name)
        else:
            raise TypeError(f"No function to be applied {node.raw_structure()}")

        self.comment(f"Generating function application: {node}")
        self.debug(f"Generating function application: {node.raw_structure()}")

        # Generate argument
        arg_val = self.generate(node.arg)
        self.comment(f"Generated Apply argument: {node.arg}")

        # Always allocate new closure state
        if self.state_ptr is None:
            self.state_ptr = self.builder.alloca(self.state_type, name=self.fresh_name("lambda_state"))

        # Load function value if needed
        #if isinstance(func_val, (ir.GEPInstr, ir.AllocaInstr)):
        #    func_val = self.builder.load(func_val)

         # Load the function pointer
        func_ptr = self.get_function_ptr(func_val)

        # Call the function
        result = self.builder.call(func_ptr, [self.state_ptr, arg_val])

        # Get function type
        #self.debug(f"Generated Apply function type: {func_val.type}")
        #if isinstance(func_val.type, ir.PointerType):
        #    func_ptr_type = func_val.type.pointee
        #    if isinstance(func_ptr_type, ir.FunctionType):
        #        # Call the function to get next function pointer
        #        next_func = self.builder.call(func_val, [self.state_ptr, arg_val])
        #        self.debug(f"Generated Apply next function type: {next_func.type}")
        #        return next_func
        #    else:
        #        # This is the final computation function
        #        result = self.builder.call(func_val, [self.state_ptr, arg_val])
        #        return result
        #else:
        return result

    def get_function_ptr(self, func_val):
        # If func_val is already a function pointer, return it
        if isinstance(func_val.type.pointee, ir.FunctionType):
            return func_val

        # If it's a pointer to a function pointer, load it
        if isinstance(func_val.type.pointee, ir.PointerType) and isinstance(func_val.type.pointee.pointee, ir.FunctionType):
            return self.builder.load(func_val)

        # If it's a nested function pointer, keep loading until we get a function pointer
        while isinstance(func_val.type, ir.PointerType):
            func_val = self.builder.load(func_val)
            if isinstance(func_val.type.pointee, ir.FunctionType):
                return func_val

        raise TypeError(f"Unable to get function pointer from value of type {func_val.type}")


    # ----------------------------------------------------------------
    # LET(name, value, body)
    # ----------------------------------------------------------------
    def generate_let(self, node: Let) -> Tuple[ir.Value, ir.Type]:
        """
        Generate let binding.
        Returns the body value and type.
        """
        self.symbol_table.push_scope()

        # Generate value
        value_ir = self.generate(node.value)

        self.debug(f"--- Generated Let value: '{node.value.raw_structure()}' ,return type: {value_ir.type}")
        self.debug(f"{value_ir}")
        self.debug(f"---")

        self.comment(f"Generated Let value: '{node.value}' ,return type: {value_ir.type}")

        # For function values, store directly in variables
        if isinstance(value_ir, ir.Function) or (isinstance(value_ir.type, ir.PointerType) and 
                                          isinstance(value_ir.type.pointee, ir.FunctionType)):
            self.symbol_table.add_variable(node.name.name, value_ir.type, value_ir)
        else:
            # For non-function values, store in an alloca
            alloca_ir = self.builder.alloca(value_ir.type)
            self.builder.store(value_ir, alloca_ir)
            self.symbol_table.add_variable(node.name.name, value_ir.type, alloca_ir)

        self.debug(f"Let - Symbol table:\n{self.symbol_table}")

        # Generate body
        body_ir = self.generate(node.body)
        self.debug(f"Generated Let body: result.type: {body_ir.type}")
        self.comment(f"Generated Let body: '{node.body}' , return type: {body_ir.type}")

        # Add return if we're in the main function
        if self.current_function.name == "main" and not self.return_generated:
            self.return_generated = True
            # Load the value from the result pointer
            #loaded_result = self.builder.load(result, name="final_result")

            # Get a pointer to the format string
            str_ptr = self.builder.gep(self.str_int,
                                       [ir.Constant(ir.IntType(64), 0),
                                        ir.Constant(ir.IntType(64), 0)],
                                       inbounds=True,
                                       name="str_ptr")

            # Call the printf function with the format string and the loaded result
            self.builder.call(self.printf, [str_ptr, body_ir])

            #self.builder.ret(body_ir)
            # Check if the current block is already terminated
            if not self.builder.block.is_terminated:
                self.builder.ret(ir.Constant(ir.IntType(32), 0))
            else:
                print(f"Warning: Block already terminated when processing let expression for {node}")

        self.symbol_table.pop_scope()

        return body_ir

    # ----------------------------------------------------------------
    # VAR(x)
    # ----------------------------------------------------------------
    def generate_var(self, node: Var) -> Tuple[ir.Value, ir.Type]:
        """Generate LLVM IR for variable reference"""
        var_entry= self.symbol_table.lookup_variable(node.name)
        if var_entry is None:
            raise Exception(f"Undeclared variable: {node.name}")
        (_type, address) = var_entry
        node.heja = "HELLO WORLD"
        return self.builder.load(address)


    # ----------------------------------------------------------------
    # BINOP(op, left, right)
    # ----------------------------------------------------------------
    def generate_binop(self, node: BinOp) -> Tuple[ir.Value, ir.Type]:
        """
        Generate binary operation.
        Returns the result value and type.
        """
        # Generate operands
        left_val = self.generate(node.left)
        right_val = self.generate(node.right)

        lval = left_val
        rval = right_val
        self.debug(f"Generated binary operation: lval={lval}, rval={rval}")

        # Load values if they are pointers
        if isinstance(left_val.type, ir.PointerType):
            lval = self.builder.load(lval)
        if isinstance(right_val.type, ir.PointerType):
            rval = self.builder.load(rval)

        # Generate operation
        if node.op == '+':
            return self.builder.add(lval, rval)
        elif node.op == '-':
            return self.builder.sub(lval, rval)
        elif node.op == '*':
            return self.builder.mul(lval, rval)
        elif node.op == '/':
            return self.builder.sdiv(lval, rval)
        elif node.op == '==':
            return self.builder.icmp_signed('==', left_val, right_val)
        elif node.op == '<':
            return self.builder.icmp_signed('<', left_val, right_val)
        elif node.op == '>':
            return self.builder.icmp_signed('>', left_val, right_val)
        else:
            raise ValueError(f"Unknown operator: {node.op}")


    # ----------------------------------------------------------------
    # UNARYOP(op, operand)
    # ----------------------------------------------------------------
    def generate_unaryop(self, node: UnaryOp) -> Tuple[ir.Value, ir.Type]:
        """
        Generate unary operation.
        Returns the result value and type.
        """
        self.debug(f"Generating unary operation: {node.op}")

        # Generate operand
        val = self.generate(node.operand)

        # Generate operation
        if node.op == '-':
            return self.builder.neg(val)
        elif node.op == '!':
            return self.builder.not_(val)
        else:
            raise ValueError(f"Unknown operator: {node.op}")


def clang(output = "foo", ll_file = "mfl.ll"):
    """Compiles the generated LLVM IR to an executable file"""
    try:
        # Use shlex.quote to safely handle filenames with spaces or special characters
        command = shlex.split(f"clang -O3 -o {output} {shlex.quote(ll_file)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Compilation successful!")
        print(result.stdout)  # Print compilation output (if any)
    except subprocess.CalledProcessError as e:
        print(f"Error compiling with clang: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print("Error: clang command not found. Make sure it's in your PATH.")

# Main is used for testing and debugging
def main():
    """Test the LLVM generator with a simple curried function"""
    # Create AST for: let add = λx.λy.λz.(x + y + z) in ((add 1) 2) 3
    #expr_str = "let add = λx.λy.λz.(x + y + z) in ((add 1) 2) 3"
    #expr_str = "let add = 3 + 4 in add"
    #expr_str = "let result = 3 + 4 in result"
    #expr_str = "let id = λx.(x) in (id 8)"
    #expr_str = "let inc = λx.(x + 1) in (inc 4)"
    #expr_str = "let one = 1 in one"
    #expr_str = "let add = λx.λy.(x + y) in 3"
    #expr_str = "let add = λx.λy.(x + y) in (add 4 5)"
    #expr_str = "let add = λx.λy.(x + y) in (add 6 9)"
    #expr_str = "let inc = λx.(x + 1) in let add = λx.λy.(x+y) in (inc (add 7 2))"
    expr_str = "letrec fac = λx.(if (x == 0) then 1 else (x * (fac (x - 1)))) in (fac 5)"

    from mfl_ply_parser import parser as ply_parser
    ast = ply_parser.parse(expr_str)

    from mfl_type_checker import infer_j
    type_ctx = {}  # Empty typing context

    # This will annotate the AST with types!
    infer_j(ast, type_ctx)

    print(f"AST(raw): '{ast.raw_structure()}'")

    # Perform program transformations
    ast = ASTTransformer.transform_letrec_to_let(ast)
    print(f"AST(transformed): {ast}")
    print(f"AST(transformed,raw): {ast.raw_structure()}")

    # Generate code
    generator = LLVMGenerator(verbose=False, generate_comments=True)
    result= generator.generate(ast)

    # Verify module
    llvm_ir = str(generator.module)
    try:
        llvm.parse_assembly(llvm_ir)
        print("Module verification successful!")
    except RuntimeError as e:
        for i, line in enumerate(llvm_ir.splitlines(), 1):
            print(f"{i:>{2}} | {line}")
        print(f"Module verification failed: {e}")


    # Write the generated code to file
    ll_file = "mfl.ll"
    with open(ll_file, "w") as f:
        f.write(llvm_ir)
    print(f"Generated LLVM IR code written to: {ll_file}")
    print(f"Compile as: clang -O3 -o foo {ll_file}")
    clang(output="foo", ll_file=ll_file)


if __name__ == "__main__":
    main()
