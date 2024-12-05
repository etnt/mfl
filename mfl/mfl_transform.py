import dataclasses
from typing import Any
from mfl_ast import (
    ASTNode, Var, Function, Apply, Let, Lets, LetBinding,
    BinOp, UnaryOp, If, LetRec, Int, Bool
)

class ASTTransformer:
    def __init__(self) -> None:
        self.counter = 0

    @staticmethod
    def multiple_bindings_to_let(ast: ASTNode) -> ASTNode:
        """
        Transform multiple let bindings to a single let binding

        Example: let x = 3, y = 4 in x + y

        result in:

          Lets([LetBinding(Var("x"), Int(3)),
                LetBinding(Var("y"), Int(4))],
                BinOp("+", Var("x"), Var("y")))

        will be transformed to:

          Let(Var("x"),
              Int(3),
              Let(Var("y"),
                  Int(4),
                  BinOp("+", Var("x"), Var("y"))))

        """
        def transform_node(node: ASTNode) -> ASTNode:
            if isinstance(node, Lets):
                # Get the bindings and body
                bindings = node.bindings[0]  # Assuming bindings is a tuple/list with one element
                body = node.body

                # Start with the innermost expression (the body)
                result = body

                if type(bindings) is list:
                    # Work backwards through the bindings to create nested Let expressions
                    for binding in reversed(bindings):
                        result = Let(binding.name, binding.value, result)
                else:
                    # Create a single Let expression
                    result = Let(bindings.name, bindings.value, result)

                return result

            # For other node types, recursively transform their children
            elif isinstance(node, Function):
                return Function(node.arg, transform_node(node.body))
            elif isinstance(node, Apply):
                return Apply(transform_node(node.func), transform_node(node.arg))
            elif isinstance(node, Let):
                return Let(node.name, transform_node(node.value), transform_node(node.body))
            elif isinstance(node, If):
                return If(
                    transform_node(node.cond),
                    transform_node(node.then_expr),
                    transform_node(node.else_expr)
                )
            elif isinstance(node, BinOp):
                return BinOp(node.op, transform_node(node.left), transform_node(node.right))
            elif isinstance(node, UnaryOp):
                return UnaryOp(node.op, transform_node(node.operand))

            # For other nodes (Var, Int, Bool), return as-is
            return node

        # Transform the entire AST
        return transform_node(ast)

    @staticmethod
    def create_y_combinator(type_info=None) -> Function:
        """
        Original implementation:

          Y = λf.(λx.f (x x)) (λx.f (x x))

        Create a more controlled Y combinator as an AST Function node
        that introduces an extra argument to control recursion and
        prevent immediate infinite expansion.
        New implementation: 

          Y = λf.(λx.f (λy. x x y)) (λx.f (λy. x x y))

        """
        # Create variables with type info
        f = Var('f')
        f.type = type_info
        x = Var('x')
        x.type = type_info
        y = Var('y')
        y.type = type_info

        # Inner lambda: λy. x x y
        x_x = Apply(x, x)
        x_x.type = type_info
        inner_application = Apply(x_x, y)
        inner_application.type = type_info
        inner_lambda = Function(y, inner_application)
        inner_lambda.type = type_info

        # Outer lambda: λx.f (λy. x x y)
        outer_body = Apply(f, inner_lambda)
        outer_body.type = type_info
        outer_lambda = Function(x, outer_body)
        outer_lambda.type = type_info

        # Final Y combinator: λf.(λx.f (λy. x x y)) (λx.f (λy. x x y))
        final_apply = Apply(outer_lambda, outer_lambda)
        final_apply.type = type_info
        y_combinator = Function(f, final_apply)
        y_combinator.type = type_info

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
                # Create Y combinator with type info from original LetRec
                y_var = Var('Y')
                y_var.type = node.type
                y_combinator = cls.create_y_combinator(node.type)

                # Create lambda for the letrec value
                rec_lambda = Function(node.name, node.value)
                rec_lambda.type = node.value.type

                # Create the Y application
                y_apply = Apply(y_var, rec_lambda)
                y_apply.type = node.type

                # Construct the transformation:
                # let Y = <y-combinator> in
                # let v = Y(λv.B) in E
                transformed = Let(
                    y_var,
                    y_combinator,
                    Let(
                        node.name,
                        y_apply,
                        transform_node(node.body)
                    )
                )
                # Preserve type information in the Let nodes
                transformed.type = node.type
                transformed.body.type = node.type

                return transformed

            # Recursively transform child nodes for other AST types
            if isinstance(node, Function):
                node.body = transform_node(node.body)
                return node

            elif isinstance(node, Apply):
                node.func = transform_node(node.func)
                node.arg = transform_node(node.arg) 
                return node

            elif isinstance(node, Let):
                node.value = transform_node(node.value)
                node.body = transform_node(node.body)
                return node

            elif isinstance(node, If):
                node.cond = transform_node(node.cond)
                node.then_expr = transform_node(node.then_expr)
                node.else_expr = transform_node(node.else_expr)
                return node

            elif isinstance(node, BinOp):
                node.left = transform_node(node.left)
                node.right = transform_node(node.right)
                return node

            elif isinstance(node, UnaryOp):
                node.operand = transform_node(node.operand)
                return node

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
