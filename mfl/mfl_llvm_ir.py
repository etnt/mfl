"""
LLVM IR Generator for MFL.

This module provides helper classes for generating LLVM IR code directly.
It follows the translation strategy outlined inTRANSLATION-STRATEGY.md.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from mfl_ast import ASTNode, Var, Int, Bool, Function, Apply, Let, BinOp, UnaryOp

class SymbolTable:
    """Tracks variable bindings and their locations in lambda state."""
    def __init__(self):
        self.scopes: List[Dict[str, int]] = [{}]  # Stack of scopes
        self.current_index: int = 0  # Next available index in lambda state

    def push_scope(self) -> None:
        """Create a new scope."""
        self.scopes.append({})

    def pop_scope(self) -> None:
        """Remove the current scope."""
        if len(self.scopes) > 1:
            self.scopes.pop()
        else:
            raise Exception("Cannot pop global scope")

    def add_variable(self, name: str) -> int:
        """Add a variable to current scope and return its state index."""
        index = self.current_index
        self.scopes[-1][name] = index
        self.current_index += 1
        return index

    def lookup_variable(self, name: str) -> Optional[int]:
        """Look up a variable's state index through all scopes."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def __str__(self) -> str:
        """Pretty print the symbol table."""
        result = []
        for i, scope in enumerate(self.scopes):
            result.append(f"Scope {i}:")
            for name, index in scope.items():
                result.append(f"  {name}: state[{index}]")
        return "\n".join(result)

class LLVMIRGenerator:
    """
    Generates LLVM IR code for MFL AST nodes.

    This class handles the direct generation of LLVM IR without using llvmlite.
    It maintains internal state for tracking variables, generating unique names,
    and managing the output IR code.
    """
    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose
        self.fresh_counter: int = 0
        self.output: List[str] = []
        self.indent_level: int = 0
        self.symbol_table = SymbolTable()
        self.current_function: Optional[str] = None

    def debug(self, msg: str) -> None:
        """Print debug message if verbose mode is enabled."""
        if self.verbose:
            print(f"LLVM IR: {msg}")

    def emit(self, line: str) -> None:
        """Add a line of IR code with proper indentation."""
        self.output.append("  " * self.indent_level + line)

    def emit_comment(self, comment: str) -> None:
        """Add a comment line with proper indentation."""
        self.emit(f"; {comment}")

    def fresh_name(self, prefix: str = "") -> str:
        """Generate a unique name with optional prefix."""
        name = f"{prefix}_{self.fresh_counter}"
        self.fresh_counter += 1
        return name

    def get_ir(self) -> str:
        """Return the complete generated IR code."""
        return "\n".join(self.output)

    def generate_module_header(self) -> None:
        """Generate the module header with required declarations."""
        self.emit('; ModuleID = "MFL Generated Module"')
        self.emit('target triple = "arm64-apple-darwin23.6.0"')
        self.emit('target datalayout = ""')
        self.emit("")

        # Lambda state type definition
        self.emit('%"lambda_state" = type {i32, i32, i32, i32, i32, i32, i32, i32}')
        self.emit("")

        # External declarations
        self.emit('declare i32 @"printf"(i8* %".1", ...)')
        self.emit("")

        # String constants
        self.emit('@".str.int" = private constant [3 x i8] c"%d\\00"')
        self.emit('@".str.bool" = private constant [3 x i8] c"%s\\00"')
        self.emit('@".str.true" = private constant [5 x i8] c"true\\00"')
        self.emit('@".str.false" = private constant [6 x i8] c"false\\00"')
        self.emit("")

    def generate_main_function_header(self) -> None:
        """Generate the main function header."""
        self.emit('define i32 @"main"() {')
        self.indent_level += 1
        self.emit("entry:")
        self.indent_level += 1

    def generate_main_function_footer(self) -> None:
        """Generate the main function footer."""
        self.emit("ret i32 0")
        self.indent_level -= 2
        self.emit("}")
        self.emit("")

    def generate_function_header(self, name: str) -> None:
        """Generate a function header."""
        self.emit(f'define i32 @"{name}"(%"lambda_state"* %".1", i32 %".2") {{')
        self.indent_level += 1
        self.emit("entry:")
        self.indent_level += 1
        self.current_function = name

    def generate_function_footer(self) -> None:
        """Generate a function footer."""
        self.indent_level -= 2
        self.emit("}")
        self.emit("")
        self.current_function = None

    def generate_state_alloc(self, name: str) -> str:
        """Generate lambda state allocation."""
        var_name = self.fresh_name(name)
        self.emit(f'%"{var_name}" = alloca %"lambda_state"')
        return var_name

    def generate_state_store(self, state_ptr: str, index: int, value: str) -> str:
        """Generate code to store a value in lambda state."""
        ptr_name = self.fresh_name("ptr")
        self.emit(f'%"{ptr_name}" = getelementptr %"lambda_state", %"lambda_state"* %"{state_ptr}", i32 0, i32 {index}')
        self.emit(f'store i32 {value}, i32* %"{ptr_name}"')
        return ptr_name

    def generate_state_load(self, state_ptr: str, index: int) -> str:
        """Generate code to load a value from lambda state."""
        ptr_name = self.fresh_name("ptr")
        val_name = self.fresh_name("val")
        self.emit(f'%"{ptr_name}" = getelementptr %"lambda_state", %"lambda_state"* %"{state_ptr}", i32 0, i32 {index}')
        self.emit(f'%"{val_name}" = load i32, i32* %"{ptr_name}"')
        return val_name

    def generate_print_int(self, value: str) -> None:
        """Generate code to print an integer value."""
        ptr_name = self.fresh_name("str_ptr")
        call_name = self.fresh_name("printf")
        self.emit(f'%"{ptr_name}" = getelementptr [3 x i8], [3 x i8]* @".str.int", i64 0, i64 0')
        self.emit(f'%"{call_name}" = call i32 (i8*, ...) @"printf"(i8* %"{ptr_name}", i32 {value})')

    def generate_binop(self, op: str, left: str, right: str) -> str:
        """Generate code for binary operations."""
        result_name = self.fresh_name("binop")
        if op == "+":
            self.emit(f'%"{result_name}" = add i32 {left}, {right}')
        elif op == "-":
            self.emit(f'%"{result_name}" = sub i32 {left}, {right}')
        elif op == "*":
            self.emit(f'%"{result_name}" = mul i32 {left}, {right}')
        elif op == "/":
            self.emit(f'%"{result_name}" = sdiv i32 {left}, {right}')
        elif op in ["==", "<", ">", "<=", ">="]:
            self.emit(f'%"{result_name}" = icmp {op} i32 {left}, {right}')
        else:
            raise ValueError(f"Unknown operator: {op}")
        return result_name

    def generate(self, ast: ASTNode) -> None:
        """Generate LLVM IR for the complete AST."""
        self.generate_module_header()
        self.generate_main_function_header()

        # Generate code for the AST
        self.emit_comment("Main expression evaluation")
        # TODO: Implement AST traversal and code generation

        self.generate_main_function_footer()
