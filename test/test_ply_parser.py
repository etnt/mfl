"""Unit tests for the PLY parser implementation."""

import unittest
import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# add ../mfl to the Python path
sys.path.insert(0, os.path.join(current_dir, '../mfl'))

from mfl_ply_parser import parse
from mfl_type_checker import BinOp, Int, Bool, If, infer_j
from mfl_ast import execute_ast
from mfl_transform import ASTTransformer


class TestPLYParser(unittest.TestCase):

    def setUp(self):
        self.transformer = ASTTransformer()

    def test_greater_than(self):
        ast = parse("5 > 3")
        self.assertIsInstance(ast, BinOp)
        self.assertEqual(ast.op, ">")
        self.assertEqual(ast.left.value, 5)
        self.assertEqual(ast.right.value, 3)

    def test_less_than(self):
        ast = parse("2 < 7")
        self.assertIsInstance(ast, BinOp)
        self.assertEqual(ast.op, "<")
        self.assertEqual(ast.left.value, 2)
        self.assertEqual(ast.right.value, 7)

    def test_equality(self):
        ast = parse("4 == 4")
        self.assertIsInstance(ast, BinOp)
        self.assertEqual(ast.op, "==")
        self.assertEqual(ast.left.value, 4)
        self.assertEqual(ast.right.value, 4)

    def test_less_than_or_equal(self):
        ast = parse("3 <= 3")
        self.assertIsInstance(ast, BinOp)
        self.assertEqual(ast.op, "<=")
        self.assertEqual(ast.left.value, 3)
        self.assertEqual(ast.right.value, 3)

    def test_greater_than_or_equal(self):
        ast = parse("8 >= 5")
        self.assertIsInstance(ast, BinOp)
        self.assertEqual(ast.op, ">=")
        self.assertEqual(ast.left.value, 8)
        self.assertEqual(ast.right.value, 5)

    def test_type_checking(self):
        ctx = {}
        for expr in [parse("5 > 3"), parse("2 < 7"), parse("4 == 4"), parse("3 <= 3"), parse("8 >= 5")]:
            type_result = infer_j(expr, ctx)
            self.assertEqual(str(type_result), "bool")

    def test_let_expression_double(self):
        ast = parse("let double = λx.(x*2) in (double 21)")
        result = execute_ast(self.transformer.multiple_bindings_to_let(ast), False)
        self.assertEqual(result, 42)

    def test_let_expression_add(self):
        ast = parse("let add = λx.λy.(x+y) in (add 3 4)")
        result = execute_ast(self.transformer.multiple_bindings_to_let(ast), False)
        self.assertEqual(result, 7)

    def test_let_expression_compose(self):
        ast = parse("let compose = λf.λg.λx.(f (g x)), add1 = λx.(x+1), double = λx.(x+x) in ((compose double add1) 2)")
        result = execute_ast(self.transformer.multiple_bindings_to_let(ast), False)
        self.assertEqual(result, 6)

    def test_basic_if_structure(self):
        ast = parse("if True then 1 else 0")
        self.assertIsInstance(ast, If)
        self.assertIsInstance(ast.cond, Bool)
        self.assertTrue(ast.cond.value)
        self.assertIsInstance(ast.then_expr, Int)
        self.assertEqual(ast.then_expr.value, 1)
        self.assertIsInstance(ast.else_expr, Int)
        self.assertEqual(ast.else_expr.value, 0)

    def test_if_with_comparison(self):
        ast = parse("if 5 > 3 then 1 else 0")
        self.assertIsInstance(ast, If)
        self.assertIsInstance(ast.cond, BinOp)
        self.assertEqual(ast.cond.op, ">")
        self.assertEqual(ast.cond.left.value, 5)
        self.assertEqual(ast.cond.right.value, 3)

    def test_if_with_arithmetic(self):
        ast = parse("if True then 2 + 3 else 5 - 1")
        self.assertIsInstance(ast, If)
        self.assertIsInstance(ast.then_expr, BinOp)
        self.assertEqual(ast.then_expr.op, "+")
        self.assertIsInstance(ast.else_expr, BinOp)
        self.assertEqual(ast.else_expr.op, "-")

    def test_nested_if(self):
        ast = parse("if True then if False then 1 else 2 else 3")
        self.assertIsInstance(ast, If)
        self.assertIsInstance(ast.then_expr, If)
        self.assertIsInstance(ast.else_expr, Int)

    def test_if_execution_true_branch(self):
        ast = parse("if 5 > 3 then 42 else 0")
        result = execute_ast(self.transformer.multiple_bindings_to_let(ast), False)
        self.assertEqual(result, 42)

    def test_if_execution_false_branch(self):
        ast = parse("if 2 > 3 then 42 else 0")
        result = execute_ast(self.transformer.multiple_bindings_to_let(ast), False)
        self.assertEqual(result, 0)

    def test_if_type_checking(self):
        ctx = {}
        # Test that condition must be boolean
        ast = parse("if True then 1 else 0")
        type_result = infer_j(ast, ctx)
        self.assertEqual(str(type_result), "int")

        # Test that branches must have same type
        ast = parse("if 5 > 3 then 42 else 0")
        type_result = infer_j(ast, ctx)
        self.assertEqual(str(type_result), "int")

if __name__ == "__main__":
    unittest.main()
