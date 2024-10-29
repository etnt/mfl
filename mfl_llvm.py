"""
LLVM IR Code Generator

This module generates LLVM IR code from the AST produced by the functional parser.
It follows the LLVM Language Reference:
https://llvm.org/docs/LangRef.html

Key features implemented:
- Integer literals
- Boolean literals
- Variables
- Lambda functions (as LLVM functions)
- Function applications
- Let bindings
- Basic arithmetic and comparison operations
"""

from typing import Any, Dict, List, Optional, Tuple
from mfl_type_checker import (
    Var, Int, Bool, Function, Apply, Let, BinOp,
    MonoType, TyCon, TyVar
)

class LLVMGenerator:
    """
    Generates LLVM IR code from AST nodes.
    Implements the visitor pattern to traverse the AST.
    """
    
    def __init__(self):
        self.fresh_counter = 0
        self.functions: Dict[str, str] = {}  # Maps function names to their types
        self.variables: Dict[str, str] = {}  # Maps variable names to their LLVM registers
        self.current_function: Optional[str] = None
        self.declarations = []
        self.definitions = []
        self._init_runtime()

    def _init_runtime(self):
        """Initialize LLVM IR with necessary declarations"""
        self.declarations.extend([
            "; MFL to LLVM IR Compiler Output",
            "",
            'target triple = "arm64-apple-macosx14.0.0"',
            "",
            "declare i32 @printf(i8* nocapture readonly, ...)",
            "",
            "; String constants for printing",
            '@.str.int = private unnamed_addr constant [3 x i8] c"%d\00"',
            '@.str.bool = private unnamed_addr constant [3 x i8] c"%s\00"',
            '@.str.true = private unnamed_addr constant [5 x i8] c"true\00"',
            '@.str.false = private unnamed_addr constant [6 x i8] c"false\00"',
            ""
        ])

    def fresh_var(self, prefix: str = "") -> str:
        """Generate a fresh LLVM register name"""
        self.fresh_counter += 1
        return f"%{prefix}_{self.fresh_counter}"

    def fresh_label(self, prefix: str = "label") -> str:
        """Generate a fresh label name"""
        self.fresh_counter += 1
        return f"{prefix}_{self.fresh_counter}"

    def generate(self, node: Any, type_info: MonoType = None) -> Tuple[str, str]:
        """
        Generate LLVM IR code for an AST node.
        Returns (register_name, type) tuple.
        """
        if isinstance(node, Int):
            return self.generate_int(node)
        elif isinstance(node, Bool):
            return self.generate_bool(node)
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
        else:
            raise ValueError(f"Unknown AST node type: {type(node)}")

    def generate_int(self, node: Int) -> Tuple[str, str]:
        """Generate LLVM IR for integer literal"""
        reg = self.fresh_var("int")
        self.definitions.append(f"    {reg} = add i32 0, {node.value}")
        return reg, "i32"

    def generate_bool(self, node: Bool) -> Tuple[str, str]:
        """Generate LLVM IR for boolean literal"""
        reg = self.fresh_var("bool")
        value = 1 if node.value else 0
        self.definitions.append(f"    {reg} = add i1 0, {value}")
        return reg, "i1"

    def generate_var(self, node: Var) -> Tuple[str, str]:
        """Generate LLVM IR for variable reference"""
        if node.name in self.variables:
            load_reg = self.fresh_var("load")
            var_type = self.variables[node.name][1]
            if var_type == "function":
                self.definitions.append(f"    {load_reg} = load i32 (i32)*, i32 (i32)** {self.variables[node.name][0]}")
                return load_reg, "function"
            else:
                self.definitions.append(f"    {load_reg} = load i32, i32* {self.variables[node.name][0]}")
                return load_reg, "i32"
        raise ValueError(f"Undefined variable: {node.name}")

    def generate_function(self, node: Function) -> Tuple[str, str]:
        """Generate LLVM IR for function definition"""
        func_name = f"@func_{self.fresh_counter}"
        self.fresh_counter += 1
        
        # Save current function context
        old_function = self.current_function
        old_variables = self.variables.copy()
        old_definitions = self.definitions
        
        self.current_function = func_name
        self.variables = {}
        self.definitions = []
        
        # Function header
        self.definitions.extend([
            f"define i32 {func_name}(i32 %{node.arg.name}) {{",
            "entry:"
        ])
        
        # Store argument in alloca
        arg_ptr = self.fresh_var("arg_ptr")
        self.definitions.append(f"    {arg_ptr} = alloca i32")
        self.definitions.append(f"    store i32 %{node.arg.name}, i32* {arg_ptr}")
        self.variables[node.arg.name] = (arg_ptr, "i32")
        
        # Generate function body
        body_reg, body_type = self.generate(node.body)
        
        # Return the result
        self.definitions.extend([
            f"    ret i32 {body_reg}",
            "}"
        ])
        
        # Add function definition to declarations
        func_def = "\n".join(self.definitions)
        
        # Restore function context
        self.current_function = old_function
        self.variables = old_variables
        self.definitions = old_definitions
        
        # Add function definition to declarations
        self.declarations.append("")  # Add blank line before function
        self.declarations.extend(func_def.split("\n"))
        
        return func_name, "function"

    def generate_apply(self, node: Apply) -> Tuple[str, str]:
        """Generate LLVM IR for function application"""
        func_reg, func_type = self.generate(node.func)
        arg_reg, arg_type = self.generate(node.arg)
        
        result_reg = self.fresh_var("call")
        self.definitions.append(f"    {result_reg} = call i32 {func_reg}(i32 {arg_reg})")
        return result_reg, "i32"

    def generate_let(self, node: Let) -> Tuple[str, str]:
        """Generate LLVM IR for let binding"""
        value_reg, value_type = self.generate(node.value)
        
        # Allocate space for the variable
        if value_type == "function":
            ptr_reg = self.fresh_var(f"{node.name.name}_ptr")
            self.definitions.append(f"    {ptr_reg} = alloca i32 (i32)*")
            self.definitions.append(f"    store i32 (i32)* {value_reg}, i32 (i32)** {ptr_reg}")
            self.variables[node.name.name] = (ptr_reg, "function")
        else:
            ptr_reg = self.fresh_var(f"{node.name.name}_ptr")
            self.definitions.append(f"    {ptr_reg} = alloca i32")
            self.definitions.append(f"    store i32 {value_reg}, i32* {ptr_reg}")
            self.variables[node.name.name] = (ptr_reg, "i32")
        
        # Generate body with new variable in scope
        body_reg, body_type = self.generate(node.body)
        
        return body_reg, body_type

    def generate_binop(self, node: BinOp) -> Tuple[str, str]:
        """Generate LLVM IR for binary operations"""
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
                self.definitions.append(f"    {result_reg} = {llvm_op} i32 {left_reg}, {right_reg}")
                return result_reg, "i1"
            else:
                self.definitions.append(f"    {result_reg} = {llvm_op} i32 {left_reg}, {right_reg}")
                return result_reg, "i32"
        else:
            raise ValueError(f"Unsupported operator: {node.op}")

    def generate_main(self, ast: Any) -> str:
        """Generate the main function that wraps the expression"""
        self.definitions = []
        self.definitions.extend([
            "define i32 @main() {",
            "entry:"
        ])
        
        # Generate code for the expression
        result_reg, result_type = self.generate(ast)
        
        # Print the result
        if result_type == "i32":
            self.definitions.extend([
                f"    call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.int, i64 0, i64 0), i32 {result_reg})"
            ])
        elif result_type == "i1":
            # Convert boolean to string pointer
            true_label = self.fresh_label("true")
            false_label = self.fresh_label("false")
            end_label = self.fresh_label("end")
            str_reg = self.fresh_var("str")
            
            self.definitions.extend([
                f"    br i1 {result_reg}, label %{true_label}, label %{false_label}",
                f"{true_label}:",
                f"    {str_reg} = getelementptr inbounds [5 x i8], [5 x i8]* @.str.true, i64 0, i64 0",
                f"    br label %{end_label}",
                f"{false_label}:",
                f"    {str_reg} = getelementptr inbounds [6 x i8], [6 x i8]* @.str.false, i64 0, i64 0",
                f"    br label %{end_label}",
                f"{end_label}:",
                f"    call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.bool, i64 0, i64 0), i8* {str_reg})"
            ])
        
        self.definitions.extend([
            "    ret i32 0",
            "}"
        ])
        
        # Add main function to declarations
        self.declarations.append("")  # Add blank line before main
        self.declarations.extend(self.definitions)
        
        return "\n".join(self.declarations)

def generate_llvm(ast: Any, type_info: MonoType = None) -> str:
    """
    Generate LLVM IR code from an AST.
    Entry point for code generation.
    """
    generator = LLVMGenerator()
    return generator.generate_main(ast)
