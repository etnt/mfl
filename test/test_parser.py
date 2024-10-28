"""Unit tests for the functional parser with comparison operators."""

import unittest
from mfl_parser import FunctionalParser
from mfl_type_checker import BinOp, Int, Bool, infer_j

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


if __name__ == "__main__":
    unittest.main()
