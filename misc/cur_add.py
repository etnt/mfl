# %%
import os
import sys
from llvmlite import ir

# Get the directory of the current file
current_dir = os.getcwd()

# add ../mfl to the Python path
sys.path.insert(0, os.path.join(current_dir, '../mfl'))

from mfl_ast import (
    Var, Int, Function, BinOp, 
)

# %%
# Create the LLVM module and int type
module = ir.Module(name="curried_functions")
int_type = ir.IntType(32)

# %%
def create_curried_function(func_node, lambda_state=None):
    """
    Recursively creates LLVM functions from nested Function nodes in the AST.
    - func_node: The AST node representing a function.
    - lambda_state: The LLVM IR struct or pointer to hold captured arguments.
    """

    # Check if this function node has a body that's also a function (curried)
    if isinstance(func_node.body, Function):
        # This is a curried function, so we create a function that returns
        # a pointer to another function.
        
        # Define the function type to return a pointer to the next function
        func_type = ir.FunctionType(ir.IntType(32).as_pointer(), [ir.IntType(32), lambda_state_type])
        func = ir.Function(module, func_type, name="curried_func_level")

        # Entry block for current function
        entry_block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(entry_block)

        # Load the state, capture the current argument
        x = func.args[0]  # Argument for this level
        lambda_state_ptr = func.args[1]
        
        # Store 'x' into %lambda_state
        x_ptr = builder.gep(lambda_state_ptr, [int_type(0), int_type(func_node.arg_index)], name=f"x_ptr_{func_node.arg_name}")
        builder.store(x, x_ptr)
        
        # Recursively create the next curried function
        next_func = create_curried_function(func_node.body, lambda_state_ptr)

        # Return pointer to the next function
        builder.ret(next_func)

        return func

    else:
        # This is the innermost function, so we perform the actual computation
        # using the captured arguments in lambda_state.
        
        # Define the function type to return an int
        func_type = ir.FunctionType(ir.IntType(32), [ir.IntType(32), lambda_state_type])
        func = ir.Function(module, func_type, name="innermost_func")

        # Entry block
        entry_block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(entry_block)

        # Load 'x', 'y', 'z' from %lambda_state to compute ((x + y) + z)
        x = builder.load(builder.gep(lambda_state_ptr, [int_type(0), int_type(0)]), name="x")
        y = builder.load(builder.gep(lambda_state_ptr, [int_type(0), int_type(1)]), name="y")
        z = builder.load(builder.gep(lambda_state_ptr, [int_type(0), int_type(2)]), name="z")

        # Perform the computation ((x + y) + z)
        sum_xy = builder.add(x, y, name="sum_xy")
        sum_xyz = builder.add(sum_xy, z, name="sum_xyz")

        # Return the result
        builder.ret(sum_xyz)

        return func

# %%
# Assuming lambda_state_type is a structure type holding all captured variables
lambda_state_type = ir.LiteralStructType([int_type, int_type, int_type])  # Modify based on depth


# %%
# Assume AST for curried function (simplified)
class Function:
    def __init__(self, arg_name, body, arg_index):
        self.arg_name = arg_name
        self.body = body
        self.arg_index = arg_index

# %%
# Simulated AST for a curried function λx.λy.λz.((x + y) + z)
curried_ast = Function("x", Function("y", Function("z", "((x + y) + z)", 2), 1), 0)

# Create the outermost function and recursively generate IR
curried_function_ir = create_curried_function(curried_ast)

# Print the generated LLVM IR
print(module)


