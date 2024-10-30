"""Unit tests for the functional parser with comparison operators."""

import unittest
import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the 'mfl' directory to the Python path
sys.path.insert(0, os.path.join(parent_dir, 'mfl'))

from mfl_parser import FunctionalParser
from mfl_type_checker import BinOp, Int, Bool, infer_j
from mfl_ast import execute_ast

class TestComparisonOperators(unittest.TestCase):

    def setUp(self):
        self.parser = FunctionalParser([], {})

    def test_greater_than(self):
        ast = self.parser.parse("5 > 3")
        self.assertIsInstance(ast, BinOp)
        self.assertEqual(ast.op, ">")
        self.assertEqual(ast.left.value, 5)
        self.assertEqual(ast.right.value, 3)

    def test_less_than(self):
        ast = self.parser.parse("2 < 7")
        self.assertIsInstance(ast, BinOp)
        self.assertEqual(ast.op, "<")
        self.assertEqual(ast.left.value, 2)
        self.assertEqual(ast.right.value, 7)

    def test_equality(self):
        ast = self.parser.parse("4 == 4")
        self.assertIsInstance(ast, BinOp)
        self.assertEqual(ast.op, "==")
        self.assertEqual(ast.left.value, 4)
        self.assertEqual(ast.right.value, 4)

    def test_less_than_or_equal(self):
        ast = self.parser.parse("3 <= 3")
        self.assertIsInstance(ast, BinOp)
        self.assertEqual(ast.op, "<=")
        self.assertEqual(ast.left.value, 3)
        self.assertEqual(ast.right.value, 3)

    def test_greater_than_or_equal(self):
        ast = self.parser.parse("8 >= 5")
        self.assertIsInstance(ast, BinOp)
        self.assertEqual(ast.op, ">=")
        self.assertEqual(ast.left.value, 8)
        self.assertEqual(ast.right.value, 5)

    def test_type_checking(self):
        ctx = {}
        for expr in [self.parser.parse("5 > 3"), self.parser.parse("2 < 7"), self.parser.parse("4 == 4"), self.parser.parse("3 <= 3"), self.parser.parse("8 >= 5")]:
            type_result = infer_j(expr, ctx)
            self.assertEqual(str(type_result), "bool")

    def test_let_expression_double(self):
        ast = self.parser.parse("let double = λx.(x*2) in (double 21)")
        result = execute_ast(ast, False)
        self.assertEqual(result, 42)

    def test_let_expression_add(self):
        ast = self.parser.parse("let add = λx.λy.(x+y) in (add 3 4)")
        result = execute_ast(ast, False)
        self.assertEqual(result, 7)

    def test_let_expression_compose(self):
        ast = self.parser.parse("let compose = λf.λg.λx.(f (g x)) in let add1 = λx.(x+1) in let double = λx.(x+x) in ((compose double add1) 2)")
        result = execute_ast(ast, False)
        self.assertEqual(result, 6)


if __name__ == "__main__":
    unittest.main()
