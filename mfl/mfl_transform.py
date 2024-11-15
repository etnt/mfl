from typing import Any
from mfl_ast import (
    ASTNode, Var, Function, Apply, Let,
    BinOp, UnaryOp, If, LetRec
)

class ASTTransformer:
    @staticmethod
    def create_y_combinator() -> Function:
        """
        Create the Y combinator as an AST Function node
        Y = λf.(λx.f (x x)) (λx.f (x x))
        """
        # Create the inner lambda x.f (x x)
        inner_x = Var('x')
        inner_f = Var('f')
        inner_body = Apply(inner_f, Apply(inner_x, inner_x))
        inner_lambda = Function(inner_x, inner_body)

        # Create the outer lambda f.(λx.f (x x)) (λx.f (x x))
        f = Var('f')
        y_body = Apply(inner_lambda, inner_lambda)
        y_combinator = Function(f, y_body)

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
