"""
This module provides an interpreter for an Abstract Syntax Tree (AST) representing a simple functional language.

The interpreter evaluates expressions in the AST, handling variables, functions, applications, basic arithmetic operations,
and comparison operations.
"""
from typing import Union, Dict
from mfl_type_checker import ASTNode, Var, Function, Apply, Let, Int, Bool, BinOp, UnaryOp


class ASTInterpreter:
    """
    An interpreter for the AST.  It maintains an environment to store variable bindings.
    """
    def __init__(self, verbose: bool = False):
        """Initializes the interpreter with an empty environment."""
        self.verbose = verbose
        self.env: Dict[str, ASTNode] = {}

    def eval(self, node: ASTNode) -> ASTNode:
        """
        Evaluates the given AST node.

        Args:
            node: The AST node to evaluate.

        Returns:
            The result of the evaluation.
        """
        if self.verbose:
            print(f"Evaluating {type(node).__name__}: {node}")

        # Variable lookup
        if isinstance(node, Var):
            if self.verbose:
                print(f"Looking up variable {node.name}")
            result = self.env.get(node.name, node)
            if self.verbose:
                print(f"Variable lookup result: {result}")
            return result

        # Function handling - functions are self-evaluating
        elif isinstance(node, Function):
            if self.verbose:
                print(f"Function is self-evaluating: {node}")
            return node

        # Function application
        elif isinstance(node, Apply):
            if self.verbose:
                print(f"Evaluating function: {node.func}")
            func = self.eval(node.func)
            if self.verbose:
                print(f"Evaluating argument: {node.arg}")
            arg = self.eval(node.arg)

            if isinstance(func, Function):
                # Curried function handling
                if isinstance(func.body, Function):
                    result = self.substitute(func.body, func.arg, arg)
                    if self.verbose:
                        print(f"Curried function result: {result}")
                    return result

                # Primitive operation handling (addition and multiplication)
                if isinstance(func.body, BinOp):
                    if isinstance(arg, Int) and isinstance(func.body.left, Int) and isinstance(func.body.right, Int):
                        if func.body.op == "+":
                            result = Int(func.body.left.value + func.body.right.value)
                            if self.verbose:
                                print(f"Primitive addition result: {result}")
                            return result
                        elif func.body.op == "*":
                            result = Int(func.body.left.value * func.body.right.value)
                            if self.verbose:
                                print(f"Primitive multiplication result: {result}")
                            return result

                # Regular function application
                if self.verbose:
                    print(f"Substituting {arg} for {func.arg} in {func.body}")
                substituted = self.substitute(func.body, func.arg, arg)
                if self.verbose:
                    print(f"After substitution: {substituted}")
                return self.eval(substituted)

            return Apply(func, arg)

        # Let expression handling
        elif isinstance(node, Let):
            if self.verbose:
                print(f"Evaluating let binding for {node.name.name}")
            value = self.eval(node.value)
            self.env[node.name.name] = value
            if self.verbose:
                print(f"Evaluating let body: {node.body}")
            return self.eval(node.body)

        # Binary operation handling
        elif isinstance(node, BinOp):
            if self.verbose:
                print(f"Evaluating left operand: {node.left}")
            left = self.eval(node.left)
            if self.verbose:
                print(f"Evaluating right operand: {node.right}")
            right = self.eval(node.right)
            if isinstance(left, Int) and isinstance(right, Int):
                # Arithmetic operations
                if node.op == "+":
                    result = Int(left.value + right.value)
                elif node.op == "*":
                    result = Int(left.value * right.value)
                elif node.op == "-":
                    result = Int(left.value - right.value)
                elif node.op == "/":
                    result = Int(left.value // right.value)  # Using integer division
                # Comparison operations
                elif node.op == ">":
                    result = Bool(left.value > right.value)
                elif node.op == "<":
                    result = Bool(left.value < right.value)
                elif node.op == "==":
                    result = Bool(left.value == right.value)
                elif node.op == "<=":
                    result = Bool(left.value <= right.value)
                elif node.op == ">=":
                    result = Bool(left.value >= right.value)
                if self.verbose:
                    print(f"Binary operation result: {result}")
                return result
            # Boolean operations
            elif isinstance(left, Bool) and isinstance(right, Bool):
                if node.op == "&":
                    result = Bool(left.value and right.value)
                elif node.op == "|":
                    result = Bool(left.value or right.value)
                if self.verbose:
                    print(f"Boolean operation result: {result}")
                return result
            return BinOp(node.op, left, right)

        # Unary operation handling
        elif isinstance(node, UnaryOp):
            if self.verbose:
                print(f"Evaluating unary operand: {node.operand}")
            operand = self.eval(node.operand)
            if isinstance(operand, Bool) and node.op == "!":
                result = Bool(not operand.value)
                if self.verbose:
                    print(f"Unary operation result: {result}")
                return result
            return UnaryOp(node.op, operand)

        if self.verbose:
            print(f"Default return: {node}")
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
    machine = ASTInterpreter(verbose)
    if verbose:
        print(f"Executing AST: {ast}")
    result = machine.eval(ast)
    if verbose:
        print(f"Final result: {result}")
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
