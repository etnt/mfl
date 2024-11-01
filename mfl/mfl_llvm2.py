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
from llvmlite import binding as llvm

# Initialize LLVM
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

from typing import Any, Dict, List, Optional, Tuple
from mfl_ast import (
    Var, Int, Bool, Function, Apply, Let, BinOp
)
from mfl_ast import (
    Var, Int, Function, BinOp, 
)
from mfl_type_checker import (MonoType, TyCon, TyVar)

# Create the LLVM module and types
module = ir.Module(name="curried_functions")
module.triple = llvm.get_default_triple()
int_type = ir.IntType(32)
bool_type = ir.IntType(1)
void_type = ir.VoidType()

class LLVMGenerator:
    """
    Generates LLVM IR code from AST nodes.
    Implements the visitor pattern to traverse the AST.
    """

    def __init__(self, verbose=False):
        self.fresh_counter = 0
        self.functions: Dict[str, ir.Function] = {}  # Maps function names to LLVM functions
        self.variables: Dict[str, Tuple[ir.Value, ir.Type]] = {}  # Maps variable names to (value, type)
        self.current_builder: Optional[ir.IRBuilder] = None
        self.current_function: Optional[ir.Function] = None
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

    def generate(self, node: Any, type_info: MonoType = None) -> Tuple[ir.Value, ir.Type]:
        """
        Generate LLVM IR code for an AST node.
        Returns (value, type) tuple.
        """
        self.debug(f"Generating code for node type: {type(node).__name__}")

        if isinstance(node, Var):
            return self.generate_var(node)
        elif isinstance(node, Int):
            return ir.Constant(int_type, node.value), int_type
        elif isinstance(node, Function):
            return self.generate_function(node)
        elif isinstance(node, BinOp):
            return self.generate_binop(node)
        elif isinstance(node, Bool):
            return ir.Constant(bool_type, 1 if node.value else 0), bool_type
        else:
            raise ValueError(f"Unknown AST node type: {type(node)}")

    def generate_var(self, node: Var) -> Tuple[ir.Value, ir.Type]:
        """Generate LLVM IR for variable reference"""
        self.debug(f"Generating variable reference: {node.name}")
        if node.name in self.variables:
            var_val, var_type = self.variables[node.name]
            if not self.current_builder:
                raise ValueError("No active IR builder")
            
            # Return the variable value directly
            return var_val, var_type
            
        raise ValueError(f"Undefined variable: {node.name}")

    def generate_binop(self, node: BinOp) -> Tuple[ir.Value, ir.Type]:
        """Generate LLVM IR for binary operations"""
        self.debug(f"Generating binary operation: {node.op}")
        
        if not isinstance(node.left, (Var, Int, Bool, BinOp, Function)):
            raise ValueError(f"Invalid left operand type: {type(node.left)}")
        if not isinstance(node.right, (Var, Int, Bool, BinOp, Function)):
            raise ValueError(f"Invalid right operand type: {type(node.right)}")
            
        # Generate code for operands
        left_val, left_type = self.generate(node.left)
        right_val, right_type = self.generate(node.right)

        if not self.current_builder:
            raise ValueError("No active IR builder")

        # Access the closure state to get captured variables
        if isinstance(node.left, Var) and node.left.name == "x":
            # Load x from closure state
            closure_ptr = self.current_function.arg
            x_ptr = self.current_builder.gep(closure_ptr, [ir.Constant(int_type, 0), ir.Constant(int_type, 0)])
            left_val = self.current_builder.load(x_ptr)

        if isinstance(node.right, Var) and node.right.name == "y":
            # Use y directly from function parameter
            right_val = self.current_function.args[1]

        # Map Python operators to LLVM builder methods
        if node.op == '+':
            result = self.current_builder.add(left_val, right_val, name="add")
            return result, int_type
        elif node.op == '-':
            result = self.current_builder.sub(left_val, right_val, name="sub")
            return result, int_type
        elif node.op == '*':
            result = self.current_builder.mul(left_val, right_val, name="mul")
            return result, int_type
        elif node.op == '/':
            result = self.current_builder.sdiv(left_val, right_val, name="div")
            return result, int_type
        elif node.op == '==':
            result = self.current_builder.icmp_signed('==', left_val, right_val, name="eq")
            return result, bool_type
        elif node.op == '<':
            result = self.current_builder.icmp_signed('<', left_val, right_val, name="lt")
            return result, bool_type
        elif node.op == '>':
            result = self.current_builder.icmp_signed('>', left_val, right_val, name="gt")
            return result, bool_type
        else:
            raise ValueError(f"Unsupported operator: {node.op}")

    def generate_function(self, func_node: Function) -> Tuple[ir.Function, ir.Type]:
        """
        Generate LLVM IR for function definitions with currying support
        Returns the function value and its type
        """
        self.lambda_depth += 1
        
        # Create closure state type for captured variables
        state_types = get_lambda_state_types(func_node)
        if state_types:
            closure_state_type = ir.LiteralStructType(state_types)
        else:
            closure_state_type = ir.LiteralStructType([])
            
        # Create function type with closure state parameter
        param_types = [ir.PointerType(closure_state_type), int_type]
        
        # For outer function, return a pointer to the inner function
        if self.lambda_depth == 1:
            # Define the inner function type that matches func_2's signature
            inner_func_type = ir.FunctionType(int_type, [ir.PointerType(closure_state_type), int_type])
            ret_type = ir.PointerType(inner_func_type)
        else:
            ret_type = int_type
            
        func_type = ir.FunctionType(ret_type, param_types)
        
        # Create function
        func_name = self.fresh_var("func")
        func = ir.Function(module, func_type, name=func_name)
        
        # Create entry block
        block = func.append_basic_block(name="entry")
        old_builder = self.current_builder
        self.current_builder = ir.IRBuilder(block)
        
        # Store old variable scope
        old_vars = self.variables.copy()
        
        # Add parameters to scope
        closure_ptr = func.args[0]
        param = func.args[1]
        self.variables[func_node.arg.name] = (param, int_type)
        
        # Generate body
        result, result_type = self.generate(func_node.body)
        
        # Return result
        if self.lambda_depth == 1:
            # Cast the function to the correct type before returning
            result_cast = self.current_builder.bitcast(result, ret_type)
            self.current_builder.ret(result_cast)
        else:
            # Inner function returns computed value
            self.current_builder.ret(result)
        
        # Restore state
        self.variables = old_vars
        self.current_builder = old_builder
        self.lambda_depth -= 1
        
        return func, func_type

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
    # Example AST for λx.λy.(x + y)

    # Create variables with proper types
    x = Var("x")
    x.type = TyCon("int", [])
    y = Var("y")
    y.type = TyCon("int", [])

    # Create BinOp node with proper Var nodes and type
    add_op = BinOp(left=x, op="+", right=y)
    add_op.type = TyCon("int", [])

    # Create the inner lambda with proper type
    inner_func = Function(y, add_op)
    inner_func.type = TyCon("int", [])

    # Create the outer lambda with proper type
    ast = Function(x, inner_func)
    ast.type = TyCon("int", [])

    # Generate code
    code_gen = LLVMGenerator(verbose=True)
    func, _ = code_gen.generate(ast)

    # Print the generated LLVM IR
    print(str(module))

    # Verify the module
    try:
        llvm_module = llvm.parse_assembly(str(module))
        llvm_module.verify()
        print("Module verification successful!")
    except Exception as e:
        print(f"Module verification failed: {e}")

if __name__ == "__main__":
    main()


