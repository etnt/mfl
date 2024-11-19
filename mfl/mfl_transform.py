import dataclasses
from typing import Any
from mfl_ast import (
    ASTNode, Var, Function, Apply, Let,
    BinOp, UnaryOp, If, LetRec, Int, Bool
)

class ASTTransformer:
    def __init__(self) -> None:
        self.counter = 0

    @staticmethod
    def create_y_combinator() -> Function:
        """
        Original implementation:

          Y = λf.(λx.f (x x)) (λx.f (x x))

        Create a more controlled Y combinator as an AST Function node
        that introduces an extra argument to control recursion and
        prevent immediate infinite expansion.
        New implementation: 

          Y = λf.(λx.f (λy. x x y)) (λx.f (λy. x x y))

        """
        # Create variables
        f = Var('f')
        x = Var('x')
        y = Var('y')

        # Inner lambda: λy. x x y
        inner_application = Apply(
            Apply(x, x), 
            y
        )
        inner_lambda = Function(y, inner_application)

        # Outer lambda: λx.f (λy. x x y)
        outer_body = Apply(f, inner_lambda)
        outer_lambda = Function(x, outer_body)

        # Final Y combinator: λf.(λx.f (λy. x x y)) (λx.f (λy. x x y))
        y_combinator = Function(f, Apply(outer_lambda, outer_lambda))

        return y_combinator

    @classmethod
    def transform_letrec_to_let(cls, ast: ASTNode) -> ASTNode:
        """
        Transform letrec construct to let construct using Y combinator

        Transformation rule:
        letrec v = B in E ==
            let val Y = <y-combinator>
            in let val v = Y(\v.B) in E)
        """
        def transform_node(node: ASTNode) -> ASTNode:
            # Recursively transform child nodes first
            if isinstance(node, LetRec):
                # Create Y combinator
                y_var = Var('Y')
                y_combinator = cls.create_y_combinator()

                # Create lambda for the letrec value
                rec_lambda = Function(node.name, node.value)

                # Construct the transformation:
                # let Y = <y-combinator> in
                # let v = Y(λv.B) in E
                transformed = Let(
                    y_var,
                    y_combinator,
                    Let(
                        node.name,
                        Apply(y_var, rec_lambda),
                        transform_node(node.body)
                    )
                )
                return transformed

            # Recursively transform child nodes for other AST types
            if isinstance(node, Function):
                return Function(
                    node.arg,
                    transform_node(node.body)
                )
            elif isinstance(node, Apply):
                return Apply(
                    transform_node(node.func),
                    transform_node(node.arg)
                )
            elif isinstance(node, Let):
                return Let(
                    node.name,
                    transform_node(node.value),
                    transform_node(node.body)
                )
            elif isinstance(node, If):
                return If(
                    transform_node(node.cond),
                    transform_node(node.then_expr),
                    transform_node(node.else_expr)
                )
            elif isinstance(node, BinOp):
                return BinOp(
                    node.op,
                    transform_node(node.left),
                    transform_node(node.right)
                )
            elif isinstance(node, UnaryOp):
                return UnaryOp(
                    node.op,
                    transform_node(node.operand)
                )

            # For simple nodes like Var, Int, Bool, return as-is
            return node

        # Transform the entire AST
        return transform_node(ast)


    def generate_variable_name(self) -> str:
        """
        Generate a new variable name
        """
        self.counter += 1
        return f"V{self.counter}"


    def letrec_for_core_erlang(self, node: LetRec) -> ASTNode:
        """
        Generate Core Erlang code for the letrec binding.

        Example: letrec fac = λx.(if (x == 0) then 1 else (x * (fac (x - 1)))) in (fac 5)

        gives us the following AST:

        LetRec(Var("fac"),
               Function(Var("x"),
                        If(BinOp("==", Var("x"), Int(0)),
                           Int(1),
                           BinOp("*", Var("x"), Apply(Var("fac"),
                                                      BinOp("-", Var("x"), Int(1)))))),
               Apply(Var("fac"), Int    (5)))

        we want to transform it to:

        Let(Var("fac"),
            LetRec(Var("'v1'/1"),              # v1 and v2 are generated variable names
                   Function(Var("X"),
                            Let(Var("v2"),
                                Var("'v1'/1"),
                                If(BinOp("==", Var("x"), Int(0)),
                                   Int(1),
                                   BinOp("*", Var("x"), Apply(Var("v2"),
                                                              BinOp("-", Var("x"), Int(1))))))
                   Apply(Var("'v1'/1")))
            Apply(Var("fac"), Int(5)))
        """
        # Generate variable names for the transformation
        v1_name = f"'{self.generate_variable_name()}'/1"
        v2_name = self.generate_variable_name()

        # Create variables
        v1_var = Var(v1_name)
        v2_var = Var(v2_name)

        def replace_recursive_calls(expr: ASTNode, orig_name: str, new_name: str) -> ASTNode:
            """Replace recursive function calls with the new variable"""
            if isinstance(expr, Var) and expr.name == orig_name:
                return Var(new_name)
            elif isinstance(expr, Apply):
                return Apply(
                    replace_recursive_calls(expr.func, orig_name, new_name),
                    replace_recursive_calls(expr.arg, orig_name, new_name)
                )
            elif isinstance(expr, Function):
                return Function(
                    expr.arg,
                    replace_recursive_calls(expr.body, orig_name, new_name)
                )
            elif isinstance(expr, If):
                return If(
                    replace_recursive_calls(expr.cond, orig_name, new_name),
                    replace_recursive_calls(expr.then_expr, orig_name, new_name),
                    replace_recursive_calls(expr.else_expr, orig_name, new_name)
                )
            elif isinstance(expr, BinOp):
                return BinOp(
                    expr.op,
                    replace_recursive_calls(expr.left, orig_name, new_name),
                    replace_recursive_calls(expr.right, orig_name, new_name)
                )
            elif isinstance(expr, Let):
                return Let(
                    expr.name,
                    replace_recursive_calls(expr.value, orig_name, new_name),
                    replace_recursive_calls(expr.body, orig_name, new_name)
                )
            return expr

        # Transform the function body to use v2 for recursive calls
        transformed_body = replace_recursive_calls(node.value.body, node.name.name, v2_name)

        # Create the inner function with the transformed body
        inner_func = Function(node.value.arg, 
            Let(v2_var,
                v1_var,
                transformed_body))

        # Create the final transformed structure
        return Let(
            node.name,
            LetRec(
                v1_var,
                inner_func,
                Apply(v1_var, node.body)
            ),
            node.body
        )

if __name__ == "__main__":
    # Create the example AST from the docstring
    # letrec fac = λx.(if (x == 0) then 1 else (x * (fac (x - 1)))) in (fac 5)
    fac_ast = LetRec(
        Var("fac"),
        Function(
            Var("x"),
            If(
                BinOp("==", Var("x"), Int(0)),
                Int(1),
                BinOp(
                    "*",
                    Var("x"),
                    Apply(
                        Var("fac"),
                        BinOp("-", Var("x"), Int(1))
                    )
                )
            )
        ),
        Apply(Var("fac"), Int(5))  # Changed from Bool(True) to Int(5) to match the example
    )

    # Create transformer and transform the AST
    transformer = ASTTransformer()
    result = transformer.letrec_for_core_erlang(fac_ast)

    # Print the original and transformed ASTs
    print("Original AST:")
    print(fac_ast.raw_structure())
    print("\nTransformed AST:")
    print(result.raw_structure())
