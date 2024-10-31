import os
import sys
from llvmlite import ir

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add parent directory to Python path
sys.path.insert(0, parent_dir)

from mfl_ast import (
    Var, Int, Function, BinOp, 
)
from mfl_type_checker import TyCon

# Create the LLVM module and int type
module = ir.Module(name="curried_functions")
int_type = ir.IntType(32)

def create_curried_function(func_node, depth=0, lambda_state=None):
    """
    Recursively creates LLVM functions from nested Function nodes in the AST.
    - func_node: The AST node representing a function
    - depth: Current nesting depth for unique naming
    - lambda_state: The LLVM IR struct pointer to hold captured arguments
    """
    if isinstance(func_node.body, Function):        
        # This is a curried function - returns pointer to next function
        return_type = ir.FunctionType(int_type, [lambda_state_type, int_type]).as_pointer()
        func_type = ir.FunctionType(return_type, [int_type, lambda_state_type])
        func = ir.Function(module, func_type, name=f"curried_func_{depth}")

        # Create entry block
        entry_block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(entry_block)

        # Allocate state if this is the outermost function
        if lambda_state is None:
            lambda_state = builder.alloca(lambda_state_type, name="lambda_state")

        # Store current argument in state
        arg = func.args[0]
        arg_ptr = builder.gep(lambda_state, [int_type(0), int_type(depth)], 
                            name=f"arg_ptr_{func_node.arg}")
        builder.store(arg, arg_ptr)

        # Create next function
        next_func = create_curried_function(func_node.body, depth + 1, lambda_state)
        builder.ret(next_func)

        return func

    else:
        # This is the innermost function that computes the final result
        func_type = ir.FunctionType(int_type, [lambda_state_type, int_type])
        func = ir.Function(module, func_type, name=f"compute_{depth}")

        entry_block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(entry_block)

        # Generate code for the body expression
        if isinstance(func_node.body, BinOp):
            # Load captured arguments from state
            args = []
            for i in range(depth):
                arg_ptr = builder.gep(lambda_state, [int_type(0), int_type(i)])
                arg = builder.load(arg_ptr, name=f"arg_{i}")
                args.append(arg)

            # Add final argument
            args.append(func.args[1])

            # Generate binary operation
            if func_node.body.op == '+':
                result = builder.add(args[0], args[1], name="add")
                for arg in args[2:]:
                    result = builder.add(result, arg, name="add")
                builder.ret(result)
            else:
                # Default case for unsupported operations
                builder.ret(int_type(0))
        elif isinstance(func_node.body, Var):
            # Handle variable references
            var_name = func_node.body.name
            # Find the variable's position in the captured arguments
            for i in range(depth):
                if func_node.body.name == f"arg_{i}":
                    arg_ptr = builder.gep(lambda_state, [int_type(0), int_type(i)])
                    result = builder.load(arg_ptr, name=f"load_{var_name}")
                    builder.ret(result)
                    break
            else:
                # If variable not found, return the final argument
                builder.ret(func.args[1])
        else:
            # Default case for other types
            builder.ret(int_type(0))

        return func

# Assuming lambda_state_type is a structure type holding all captured variables
#lambda_state_type = ir.LiteralStructType([int_type, int_type, int_type])  # Modify based on depth

def get_lambda_state_types(ast):
    """
    Traverses the AST and determines the types of captured variables in lambda functions.

    Args:
        ast: The root of the AST (a Function node representing the outermost lambda).

    Returns:
        A list of types representing the captured variables.  Returns an empty list if no 
        captured variables are found.  Currently only supports 'int' type.
    """
    captured_var_types = []
    # Use a recursive helper function to traverse the AST
    def traverse(node):
        if isinstance(node, Function) and isinstance(node.arg, Var):
            captured_var_types.append(ir.IntType(32)) # FIXME - currently only supports 'int' type
            print(f"Captured variable: {node.arg.name}, Type: {node.arg.type}")
            traverse(node.body)
        else: 
            return

    traverse(ast)
    return captured_var_types

def main():
    # Example AST for λx.λy.λz.(x + y + z)
    x = Var("x")
    x.type = TyCon("int",[])
    y = Var("y")
    y.type = TyCon("int",[]),
    z = Var("z")
    z.type = TyCon("int",[]),
    ast = Function(x, Function(y, Function(z, BinOp( BinOp(x, "+", y), "+", z))))

    captured_var_types = get_lambda_state_types(ast)
    print(f"Captured variable types: {captured_var_types}")

    global lambda_state_type
    lambda_state_type = ir.LiteralStructType(captured_var_types)

    # Create the outermost function and recursively generate IR
    curried_function_ir = create_curried_function(ast)

    # Print the generated LLVM IR
    print(module)

if __name__ == "__main__":
    main()


