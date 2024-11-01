"""
LLVM IR Code Generator

This module generates LLVM IR code from the AST produced by the functional parser.
It uses the llvmlite library to create LLVM IR instructions and functions.

Key features implemented:
- Integer literals
- Boolean literals
- Variables
- Lambda functions (as LLVM functions with currying support)
- Function applications
- Let bindings
- Basic arithmetic and comparison operations
"""

import os
import sys
from llvmlite import ir

from typing import Any, Dict, List, Optional, Tuple
from mfl_ast import (
    Var, Int, Bool, Function, Apply, Let, BinOp
)
from mfl_ast import (
    Var, Int, Function, BinOp, 
)
from mfl_type_checker import (MonoType, TyCon, TyVar)

# Create the LLVM module and int type
module = ir.Module(name="curried_functions")
int_type = ir.IntType(32)

class LLVMGenerator:
    """
    Generates LLVM IR code from AST nodes.
    Implements the visitor pattern to traverse the AST.
    """

    def __init__(self, verbose=False):
        self.fresh_counter = 0
        self.functions: Dict[str, str] = {}  # Maps function names to their types
        self.variables: Dict[str, str] = {}  # Maps variable names to their LLVM registers
        self.current_function: Optional[str] = None
        self.declarations = []
        self.definitions = []
        self.verbose = verbose
        self.lambda_depth = 0  # Track nested lambda depth

    def debug(self, msg: str):
        """Print debug message if verbose mode is enabled"""
        if self.verbose:
            print(f"LLVM: {msg}")

    def fresh_var(self, prefix: str = "") -> str:
        """Generate a fresh LLVM register name"""
        self.fresh_counter += 1
        name = f"%{prefix}_{self.fresh_counter}"
        self.debug(f"Generated fresh variable: {name}")
        return name

    def fresh_label(self, prefix: str = "label") -> str:
        """Generate a fresh label name"""
        self.fresh_counter += 1
        name = f"{prefix}_{self.fresh_counter}"
        self.debug(f"Generated fresh label: {name}")
        return name

    def generate(self, node: Any, type_info: MonoType = None) -> Tuple[str, str]:
        """
        Generate LLVM IR code for an AST node.
        Returns (register_name, type) tuple.
        """
        self.debug(f"Generating code for node type: {type(node).__name__}")
        if isinstance(node, Var):
            return self.generate_var(node)
        elif isinstance(node, Function):
            return self.generate_function(node)
        elif isinstance(node, BinOp):
            return self.generate_binop(node)
        else:
            raise ValueError(f"Unknown AST node type: {type(node)}")

    def generate_var(self, node: Var) -> Tuple[str, str]:
        """Generate LLVM IR for variable reference"""
        self.debug(f"Generating variable reference: {node.name}")
        if node.name in self.variables:
            load_reg = self.fresh_var("load")
            var_info = self.variables[node.name]
            if var_info[1] == "function":
                self.debug(f"Loading function pointer: {node.name}")
                self.definitions.append(f"    {load_reg} = load i32 (%lambda_args*)*, i32 (%lambda_args*)** {var_info[0]}")
                return load_reg, "function"
            else:
                self.debug(f"Loading variable value: {node.name}")
                self.definitions.append(f"    {load_reg} = load i32, i32* {var_info[0]}")
                return load_reg, "i32"
        raise ValueError(f"Undefined variable: {node.name}")

    def generate_binop(self, node: BinOp) -> Tuple[str, str]:
        """Generate LLVM IR for binary operations"""
        self.debug(f"Generating binary operation: {node.op}")
        left_reg, left_type = self.generate(node.left)
        right_reg, right_type = self.generate(node.right)

        result_reg = self.fresh_var("binop")

        # Map Python operators to LLVM instructions
        op_map = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "sdiv",
            "&": "and",
            "|": "or",
            "==": "icmp eq",
            "<": "icmp slt",
            ">": "icmp sgt",
            "<=": "icmp sle",
            ">=": "icmp sge"
        }

        if node.op in op_map:
            llvm_op = op_map[node.op]
            if llvm_op.startswith("icmp"):
                self.debug(f"Generating comparison: {llvm_op}")
                self.definitions.append(f"    {result_reg} = {llvm_op} i32 {left_reg}, {right_reg}")
                return result_reg, "i1"
            else:
                self.debug(f"Generating arithmetic: {llvm_op}")
                self.definitions.append(f"    {result_reg} = {llvm_op} i32 {left_reg}, {right_reg}")
                return result_reg, "i32"
        else:
            raise ValueError(f"Unsupported operator: {node.op}")

    def generate_function(self, func_node, depth=0, lambda_state=None):
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
            next_func = self.generate_function(func_node.body, depth + 1, lambda_state)
            builder.ret(next_func)

            return self.generate(func_node.body)

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

            return self.generate(func_node.body)

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

    code_gen = LLVMGenerator()
    code_gen.generate(ast)
    # Create the outermost function and recursively generate IR
    #curried_function_ir = create_curried_function(ast)

    # Print the generated LLVM IR
    print(module)

if __name__ == "__main__":
    main()


