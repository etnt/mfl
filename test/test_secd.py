"""Unit tests for mfl_secd.py"""

import unittest
import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# add ../mfl to the Python path
sys.path.insert(0, os.path.join(current_dir, '../mfl'))

from mfl_secd import SECDMachine
from mfl_parser import FunctionalParser
from mfl_type_checker import infer_j
from mfl_secd import execute_ast

class TestSECDMachine(unittest.TestCase):

    def setUp(self):
        self.parser = FunctionalParser([], {})
        self.machine = SECDMachine()

    def test_let_expression(self):
        ast = self.parser.parse("let x = 5 in x")
        infer_j(ast, {}) #type check
        result = execute_ast(ast, False)
        self.assertEqual(result, 5)

    def test_lambda_application(self):
        ast = self.parser.parse("let double = λx.(x * 2) in (double 5)")
        infer_j(ast, {}) #type check
        result = execute_ast(ast, False)
        self.assertEqual(result, 10)

    def test_nested_lambda_application(self):
        ast = self.parser.parse("let add = λx.λy.(x + y) in (add 3 4)")
        infer_j(ast, {}) #type check
        result = execute_ast(ast, False)
        self.assertEqual(result, 7)

    def test_complex_nested_lambda_application(self):
        ast = self.parser.parse("let add = λx.λy.(x + y) in (add 3 (add 4 5))")
        infer_j(ast, {}) #type check
        result = execute_ast(ast, False)
        self.assertEqual(result, 12)

    def test_composition(self):
        ast = self.parser.parse("let compose = λf.λg.λx.(f (g x)) in let add1 = λx.(x + 1) in let double = λx.(x + x) in ((compose double add1) 2)")
        infer_j(ast, {}) #type check
        result = execute_ast(ast, False)
        self.assertEqual(result, 6)

    def test_less_than(self):
        ast = self.parser.parse("3 < 5")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertTrue(result)

    def test_less_than_equal(self):
        ast = self.parser.parse("3 <= 3")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertTrue(result)

    def test_greater_than(self):
        ast = self.parser.parse("5 > 3")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertTrue(result)

    def test_greater_than_equal(self):
        ast = self.parser.parse("5 >= 5")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertTrue(result)

    def test_equals(self):
        ast = self.parser.parse("5 == 5")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertTrue(result)

    def test_comparison_in_function(self):
        ast = self.parser.parse("let isPositive = λx.(x > 0) in (isPositive 5)")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertTrue(result)

    def test_complex_comparison(self):
        # Test that checks if a number is between two other numbers
        ast = self.parser.parse("let between = λx.λy.λz.((x >= y) & (y >= z)) in (between 10 5 3)")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertTrue(result)

    def test_equality_with_expressions(self):
        ast = self.parser.parse("let x = 5 in ((x + 3) == 8)")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertTrue(result)

    def test_comparison_false_cases(self):
        # Test cases where comparisons should return false
        ast = self.parser.parse("3 > 5")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertFalse(result)

        ast = self.parser.parse("5 < 3")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertFalse(result)

        ast = self.parser.parse("4 == 5")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertFalse(result)

    def test_comparison_with_arithmetic(self):
        # Test comparison operators with arithmetic expressions
        ast = self.parser.parse("(2 + 3) > (1 + 2)")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertTrue(result)

        ast = self.parser.parse("(3 * 2) == (2 * 3)")
        infer_j(ast, {})
        result = execute_ast(ast, False)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
