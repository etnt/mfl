"""
Core Erlang Code Generator

This module generates Core Erlang code from the AST produced by the functional parser.
It follows the Core Erlang specification version 1.0.3:
https://www2.it.uu.se/research/group/hipe/cerl/doc/core_erlang-1.0.3.pdf

Core Erlang is a functional intermediate language used in the Erlang compiler pipeline.
It provides a simpler syntax than Erlang while maintaining its semantic properties.

Key features implemented:
- Integer literals
- Variables (converted to Core Erlang variables)
- Lambda abstractions (fun expressions)
- Function applications
- Let bindings
- Letrec bindings
- Basic arithmetic operations

Example Core Erlang output:
    let <X> = 42 in X         % Simple let binding
    fun(X) -> X end           % Identity function
    let <F> = fun(X) -> X end in apply F (42)  % Function application
"""

from typing import Any, Dict, List
from mfl_ast import (
    Var, Int, Bool, Function, Apply, Let, BinOp, If, LetRec
)

class CoreErlangGenerator:
    """
    Generates Core Erlang code from AST nodes.
    Implements the visitor pattern to traverse the AST.
    """

    def __init__(self):
        self.fresh_var_counter = 0
        self.indentation = 0

    def fresh_var(self, prefix="V") -> str:
        """Generate a fresh Core Erlang variable name"""
        self.fresh_var_counter += 1
        return f"{prefix}_{self.fresh_var_counter}"

    def indent(self) -> str:
        """Return current indentation as spaces"""
        return "  " * self.indentation

    def generate(self, node: Any) -> str:
        """
        Generate Core Erlang code for an AST node.
        Dispatches to appropriate handler method based on node type.
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
        elif isinstance(node, LetRec):
            return self.generate_letrec(node)
        elif isinstance(node, BinOp):
            return self.generate_binop(node)
        elif isinstance(node, If):
            return self.generate_if(node)
        else:
            raise ValueError(f"Unknown AST node type: {type(node)}")

    def generate_int(self, node: Int) -> str:
        """Generate Core Erlang code for integer literal"""
        return str(node.value)

    def generate_bool(self, node: Bool) -> str:
        """Generate Core Erlang code for boolean literal"""
        return "'true'" if node.value else "'false'"

    def generate_var(self, node: Var) -> str:
        """
        Generate Core Erlang code for variable reference.
        Core Erlang variables must start with uppercase.
        """
        # Ensure variable names start with uppercase
        name = node.name
        if name[0].islower():
            name = name[0].upper() + name[1:]
        return name

    def generate_function(self, node: Function) -> str:
        """
        Generate Core Erlang code for lambda abstraction.
        Format: fun(Var) -> Body end
        """
        param = self.generate_var(node.arg)
        self.indentation += 1
        body = self.generate(node.body)
        self.indentation -= 1
        return f"fun({param}) ->\n{self.indent()}{body}\n"

    def generate_apply(self, node: Apply) -> str:
        """
        Generate Core Erlang code for function application.
        Format: apply Func (Arg)
        """
        func = self.generate(node.func)
        arg = self.generate(node.arg)
        return f"apply {func} ({arg})"

    def generate_let(self, node: Let) -> str:
        """
        Generate Core Erlang code for let binding.
        Format: let <Var> = Value in Body
        """
        var = self.generate_var(node.name)
        value = self.generate(node.value)
        self.indentation += 1
        body = self.generate(node.body)
        self.indentation -= 1
        return f"let <{var}> =\n{self.indent()}{value}\nin\n{self.indent()}{body}"

    def generate_letrec(self, node: LetRec) -> str:
        """
        Generate Core Erlang code for recursive let binding.
        Format: letrec <Var> = Value in Body.
        """
        var = self.generate_var(node.name)
        value = self.generate(node.value)
        self.indentation += 1
        # The body here must have been transformed to be e.g:  Apply(Var("'Loop'/1") , ...)
        # And we are only interested in the function body to fit how Core Erlang wants it.
        body = self.generate(node.body.func)
        self.indentation -= 1
        return f"letrec {var} =\n{self.indent()}{value}\nin\n{self.indent()}{body}"

    def generate_binop(self, node: BinOp) -> str:
        """
        Generate Core Erlang code for binary operations.
        Format: call 'erlang' Op (Left, Right)
        """
        # Map Python operators to Erlang operator names
        op_map = {
            "+": "'+'",
            "-": "'-'", 
            "*": "'*'",
            "/": "'div'",  # Integer division in Core Erlang
            "&": "'and'",
            "|": "'or'",
            "!": "'not'",
            "==": "'=='",
            "!=": "'/='",
            "<": "'<'",
            "<=": "'=<'",
            ">": "'>'",
            ">=": "'>='"
        }

        left = self.generate(node.left)
        right = self.generate(node.right)
        op = op_map[node.op]

        # Operators such as + are not part of the Core Erlang language, so the compiler
        # has translated the use of + to a call to the BIF erlang:'+'/2.
        return f"call 'erlang':{op} ({left}, {right})"

    def generate_if(self, node: If) -> str:
        """
        Generate Core Erlang code for if-then-else expression.
        Format: if Cond then Expr1 else Expr2
        """
        cond = self.generate(node.cond)
        then_expr = self.generate(node.then_expr)
        else_expr = self.generate(node.else_expr)
        return f"case <> of <> when {cond} -> {then_expr} <> when 'true' -> {else_expr} end"

def generate_core_erlang(ast: Any, output="mfl") -> str:
    """
    Generate Core Erlang code from an AST.
    Entry point for code generation.
    Args:
        ast: The AST node to generate code from
        type_info: Optional type information from type inference

    Returns:
        A string containing the generated Core Erlang code
    """
    generator = CoreErlangGenerator()
    code = generator.generate(ast)

    # Wrap in module structure if this is a top-level expression
    return f"module '{output}' ['main'/0]\nattributes ['file' = [{{\"rune.mfl\",1}}]]\n\n'main'/0 =\n  fun() -> {code}\n\nend\n"
