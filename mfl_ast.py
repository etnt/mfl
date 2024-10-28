"""
This module provides an interpreter for an Abstract Syntax Tree (AST) representing a simple functional language.

The interpreter evaluates expressions in the AST, handling variables, functions, applications, and basic arithmetic operations.
It supports curried functions and let expressions.
"""
from typing import Union, Dict
from mfl_type_checker import ASTNode, Var, Function, Apply, Let, Int, Bool, BinOp, UnaryOp


class ASTInterpreter:
    """
    An interpreter for the AST.  It maintains an environment to store variable bindings.
    """
    def __init__(self):
        """Initializes the interpreter with an empty environment."""
        self.env: Dict[str, ASTNode] = {}

    def eval(self, node: ASTNode) -> ASTNode:
        """
        Evaluates the given AST node.

        Args:
            node: The AST node to evaluate.

        Returns:
            The result of the evaluation.
        """

        # Variable lookup
        if isinstance(node, Var):
            return self.env.get(node.name, node)

        # Function handling - functions are self-evaluating
        elif isinstance(node, Function):
            return node

        # Function application
        elif isinstance(node, Apply):
            func = self.eval(node.func)
            arg = self.eval(node.arg)

            if isinstance(func, Function):
                # Curried function handling
                if isinstance(func.body, Function):
                    result = self.substitute(func.body, func.arg, arg)
                    return result

                # Primitive operation handling (addition and multiplication)
                if isinstance(func.body, BinOp):
                    if isinstance(arg, Int) and isinstance(func.body.left, Int) and isinstance(func.body.right, Int): #Check if all are Ints
                        if func.body.op == "+":
                            return Int(func.body.left.value + func.body.right.value) #Fixed: Use func.body.left.value and func.body.right.value
                        elif func.body.op == "*":
                            return Int(func.body.left.value * func.body.right.value) #Fixed: Use func.body.left.value and func.body.right.value

                # Regular function application
                return self.eval(self.substitute(func.body, func.arg, arg))

            return Apply(func, arg)

        # Let expression handling
        elif isinstance(node, Let):
            value = self.eval(node.value)
            self.env[node.name.name] = value
            return self.eval(node.body)

        # Binary operation handling
        elif isinstance(node, BinOp):
            left = self.eval(node.left)
            right = self.eval(node.right)
            if isinstance(left, Int) and isinstance(right, Int):
                if node.op == "+":
                    return Int(left.value + right.value)
                elif node.op == "*":
                    return Int(left.value * right.value)
            return BinOp(node.op, left, right)

        return node


    def substitute(self, expr: ASTNode, var: Var, value: ASTNode) -> ASTNode:
        """
        Substitutes 'value' for 'var' in 'expr'.  This is a recursive function that traverses the AST.

        Args:
            expr: The expression to substitute in.
            var: The variable to replace.
            value: The value to substitute.

        Returns:
            The expression with the substitution applied.
        """
        if isinstance(expr, Var):
            return value if expr.name == var.name else expr
        elif isinstance(expr, Apply):
            return Apply(
                self.substitute(expr.func, var, value),
                self.substitute(expr.arg, var, value)
            )
        elif isinstance(expr, Function):
            if expr.arg.name == var.name:
                return expr
            if isinstance(expr.body, Function) and expr.body.arg.name == var.name:
                return expr
            return Function(
                expr.arg,
                self.substitute(expr.body, var, value)
            )
        elif isinstance(expr, BinOp):
            return BinOp(
                expr.op,
                self.substitute(expr.left, var, value),
                self.substitute(expr.right, var, value)
            )
        elif isinstance(expr, Let):
            if expr.name.name == var.name:
                return expr
            new_value = self.substitute(expr.value, var, value)
            new_body = self.substitute(expr.body, var, value)
            return Let(expr.name, new_value, new_body)
        return expr


def execute_ast(ast: ASTNode, verbose: bool = False) -> ASTNode:
    """
    Executes an AST using the AST interpreter.

    Args:
        ast: The AST to execute.
        verbose: Whether to print verbose output.

    Returns:
        The result of the execution.
    """
    machine = ASTInterpreter()
    if verbose:
        print(f"Executing AST: {ast}")
    result = machine.eval(ast)
    if verbose:
        print(f"Result: {result}")
    return result

if __name__ == "__main__":
    # Example usage (assuming AST tree is constructed accordingly):
    ast_interpreter = ASTInterpreter()

    # Example, identity function: λx.x applied to 42
    result = ast_interpreter.eval(Apply(Function(Var("x"), Var("x")), Int(42)))
    assert result == 42
    print(result)

    # Example, double function: λx.x*2 applied to 3
    result = ast_interpreter.eval(Let(Var("double"), Function(Var("x"), BinOp("*", Var("x"), Int(2))), Apply(Var("double"), Int(3))))
    assert result == 6
    print(result)

    # Example: let compose = λf.λg.λx.(f (g x)) in let add1 = λx.(x + 1) in let double = λx.(x + x) in (((compose double) add1) 4)
    compose = Let(Var("compose"), 
                 Function(Var("f"), Function(Var("g"), Function(Var("x"), 
                         Apply(Var("f"), Apply(Var("g"), Var("x")))))),
                 Let(Var("add1"), 
                     Function(Var("x"), BinOp("+", Var("x"), Int(1))),
                     Let(Var("double"), 
                         Function(Var("x"), BinOp("+", Var("x"), Var("x"))),
                         Apply(Apply(Apply(Var("compose"), Var("double")), Var("add1")), Int(4)))))
    result = ast_interpreter.eval(compose)
    assert result == 10
    print(result)
