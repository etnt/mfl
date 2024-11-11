"""
Direct LLVM IR Generator for curried functions.

This module generates LLVM IR code from AST nodes without using llvmlite,
with special handling for curried functions using closure state to hold
captured variables.
"""

import subprocess
import shlex
from typing import Dict, Optional, Any

from .mfl_ast import (
    ASTNode, Var, Function, Apply, Let, Int, Bool, BinOp, UnaryOp
)
from .mfl_type_checker import TyVar, TyCon
from .mfl_llvm_ir import LLVMIRGenerator

# Rest of the file remains unchanged
class DirectLLVMGenerator:
    """
    Generates LLVM IR code from AST nodes without using llvmlite.
    Handles curried functions by using closure state to hold captured variables.
    """
    def __init__(self, verbose=False, generate_comments=True):
        self.verbose = verbose
        self.generate_comments = generate_comments
        self.ir_gen = LLVMIRGenerator(verbose=verbose)

    def debug(self, msg: str) -> None:
        """Print debug message if verbose mode is enabled"""
        if self.verbose:
            print(f"LLVM: {msg}")

    def generate(self, node: ASTNode) -> str:
        """
        Generate LLVM IR for an AST node.
        Returns the generated IR code.
        """
        # Generate module header
        self.ir_gen.generate_module_header()
        
        # Generate main function
        self.ir_gen.generate_main_function_header()
        
        # Allocate initial state
        state_name = self.ir_gen.generate_state_alloc("state")
        
        # Generate code based on node type
        self._generate_expression(node, state_name)
        
        # Close main function
        self.ir_gen.generate_main_function_footer()
        
        # Get the complete IR code
        return self.ir_gen.get_ir()

    def _generate_expression(self, node: ASTNode, state_ptr: str) -> str:
        """Generate code for an expression"""
        if isinstance(node, Int):
            # Print integer value directly
            self.ir_gen.generate_print_int(str(node.value))
            return str(node.value)
            
        elif isinstance(node, Bool):
            # Print boolean using appropriate string constant
            if node.value:
                self.ir_gen.emit('call i32 (i8*, ...) @"printf"(i8* getelementptr ([3 x i8], [3 x i8]* @".str.bool", i64 0, i64 0), i8* getelementptr ([5 x i8], [5 x i8]* @".str.true", i64 0, i64 0))')
            else:
                self.ir_gen.emit('call i32 (i8*, ...) @"printf"(i8* getelementptr ([3 x i8], [3 x i8]* @".str.bool", i64 0, i64 0), i8* getelementptr ([6 x i8], [6 x i8]* @".str.false", i64 0, i64 0))')
            return "1" if node.value else "0"
            
        elif isinstance(node, Var):
            # Load variable from state
            index = self.ir_gen.symbol_table.lookup_variable(node.name)
            if index is None:
                raise NameError(f"Undefined variable: {node.name}")
            value = self.ir_gen.generate_state_load(state_ptr, index)
            self.ir_gen.generate_print_int(f"%\"{value}\"")
            return f"%\"{value}\""
            
        elif isinstance(node, Function):
            # Generate function definition
            func_name = self._generate_function(node)
            return func_name
            
        elif isinstance(node, Apply):
            # Generate function application
            return self._generate_apply(node, state_ptr)
            
        elif isinstance(node, Let):
            # Generate let binding
            return self._generate_let(node, state_ptr)
            
        elif isinstance(node, BinOp):
            # Generate binary operation
            left = self._generate_expression(node.left, state_ptr)
            right = self._generate_expression(node.right, state_ptr)
            return self.ir_gen.generate_binop(node.op, left, right)
            
        elif isinstance(node, UnaryOp):
            # Generate unary operation
            val = self._generate_expression(node.operand, state_ptr)
            result = self.ir_gen.fresh_name("unop")
            if node.op == "!":
                self.ir_gen.emit(f'%"{result}" = xor i1 {val}, true')
            elif node.op == "-":
                self.ir_gen.emit(f'%"{result}" = sub i32 0, {val}')
            return result
            
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def _generate_function(self, node: Function) -> str:
        """Generate code for lambda function"""
        # Push new scope for function parameters
        self.ir_gen.symbol_table.push_scope()
        
        # Generate unique function name
        func_name = self.ir_gen.fresh_name("compute")
        
        # Start function definition
        self.ir_gen.generate_function_header(func_name)
        
        # Add parameter to symbol table
        param_index = self.ir_gen.symbol_table.add_variable(node.arg.name)
        
        # Store parameter in state
        self.ir_gen.generate_state_store(".1", param_index, "%\".2\"")
        
        # Generate body code
        if isinstance(node.body, Function):
            # Handle curried function by generating inner function
            inner_func = self._generate_function(node.body)
            self.ir_gen.emit(f'ret i32* @"{inner_func}"')
        else:
            # Generate regular body code
            result = self._generate_expression(node.body, ".1")
            self.ir_gen.emit(f'ret i32 {result}')
        
        # End function definition
        self.ir_gen.generate_function_footer()
        
        # Pop scope
        self.ir_gen.symbol_table.pop_scope()
        
        return func_name

    def _generate_apply(self, node: Apply, state_ptr: str) -> str:
        """Generate code for function application"""
        # Generate function code
        func_name = self._generate_expression(node.func, state_ptr)
        
        # Generate argument
        if isinstance(node.arg, Int):
            arg_val = str(node.arg.value)
        else:
            arg_val = self._generate_expression(node.arg, state_ptr)
        
        # Generate function call
        result = self.ir_gen.fresh_name("result")
        self.ir_gen.emit(f'%"{result}" = call i32 @"{func_name}"(%"lambda_state"* %"{state_ptr}", i32 {arg_val})')
        
        # Print result
        self.ir_gen.generate_print_int(f"%\"{result}\"")
        
        return result

    def _generate_let(self, node: Let, state_ptr: str) -> str:
        """Generate code for let binding"""
        # Push new scope
        self.ir_gen.symbol_table.push_scope()
        
        # Generate value
        value = self._generate_expression(node.value, state_ptr)
        
        # Add variable to symbol table and store in state
        var_index = self.ir_gen.symbol_table.add_variable(node.name.name)
        self.ir_gen.generate_state_store(state_ptr, var_index, value)
        
        # Generate body
        result = self._generate_expression(node.body, state_ptr)
        
        # Pop scope
        self.ir_gen.symbol_table.pop_scope()
        
        return result

def clang(output = "foo", ll_file = "mfl.ll"):
    """Compiles the generated LLVM IR to an executable file"""
    try:
        command = shlex.split(f"clang -O3 -o {output} {shlex.quote(ll_file)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Compilation successful!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error compiling with clang: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print("Error: clang command not found. Make sure it's in your PATH.")

def main():
    """Test the direct LLVM generator with a simple expression"""
    expr_str = "let id = λx.(x) in (id 8)"
    
    from .mfl_ply_parser import parser as ply_parser
    ast = ply_parser.parse(expr_str)
    
    from .mfl_type_checker import infer_j
    type_ctx = {}
    infer_j(ast, type_ctx)
    
    print(f"AST(raw): '{ast.raw_structure()}'")
    
    # Generate code
    generator = DirectLLVMGenerator(verbose=False, generate_comments=True)
    llvm_ir = generator.generate(ast)
    
    # Write the generated code to file
    ll_file = "mfl.ll"
    with open(ll_file, "w") as f:
        f.write(llvm_ir)
    print(f"Generated LLVM IR code written to: {ll_file}")
    print(f"Compile as: clang -O3 -o foo {ll_file}")
    clang(output="foo", ll_file=ll_file)

if __name__ == "__main__":
    main()
