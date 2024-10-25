# Import the ASTNode classes (assuming they are implemented based on the document)
from typing import Union, Dict
from mfl_type_checker import ASTNode, Var, Function, Apply, Let, Int, Bool, BinOp, UnaryOp


# SKI-specific combinators
class S(ASTNode):
    def __init__(self, f=None, g=None):
        self.f = f
        self.g = g

class K(ASTNode):
    def __init__(self, x=None):
        self.x = x

class I(ASTNode):
    def __init__(self):
        pass

# SKI Machine for evaluation
class SKIMachine:
    def __init__(self):
        self.env: Dict[str, ASTNode] = {}

    def eval(self, node: ASTNode) -> ASTNode:
        """Evaluates the AST node based on SKI rules."""
        if isinstance(node, (Int, Bool, S, K, I)):
            return node

        if isinstance(node, Var):
            return self.env.get(node.name, node)

        elif isinstance(node, Function):
            return node

        elif isinstance(node, Apply):
            func = self.eval(node.func)
            arg = self.eval(node.arg)

            if isinstance(func, Function):
                # Handle curried functions
                if isinstance(func.body, Function):
                    # First apply the outer function
                    result = self.substitute(func.body, func.arg, arg)
                    return result

                # Handle primitive operations
                if isinstance(func.body, BinOp):
                    if isinstance(arg, Int):
                        if func.body.op == "+":
                            if isinstance(func.body.left, Var) and func.body.left.name == func.arg.name:
                                return Int(arg.value + func.body.right.value)
                            elif isinstance(func.body.right, Var) and func.body.right.name == func.arg.name:
                                return Int(func.body.left.value + arg.value)
                        elif func.body.op == "*":
                            if isinstance(func.body.left, Var) and func.body.left.name == func.arg.name:
                                return Int(arg.value * func.body.right.value)
                            elif isinstance(func.body.right, Var) and func.body.right.name == func.arg.name:
                                return Int(func.body.left.value * arg.value)

                # Regular function application
                return self.eval(self.substitute(func.body, func.arg, arg))

            return Apply(func, arg)

        elif isinstance(node, Let):
            # Evaluate the value
            value = self.eval(node.value)
            # Store in environment
            self.env[node.name.name] = value
            # Evaluate the body
            return self.eval(node.body)

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
        """Substitutes value for var in expr."""
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
    """Execute an AST using the SKI combinator machine."""
    machine = SKIMachine()
    if verbose:
        print(f"Executing AST: {ast}")
    result = machine.eval(ast)
    if verbose:
        print(f"Result: {result}")
    return result

if __name__ == "__main__":
    # Example usage (assuming AST tree is constructed accordingly):
    ski_machine = SKIMachine()
    result = ski_machine.eval(Apply(Function(Var("x"), Var("x")), Int(42)))
    print(result.raw_structure())
    print("\n")

    result = ski_machine.eval(Let(Var("double"), Function(Var("x"), BinOp("*", Var("x"), Int(2))), Apply(Var("double"), Int(3))))
    print(result.raw_structure())
