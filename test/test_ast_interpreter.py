import unittest
import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the 'mfl' directory to the Python path
sys.path.insert(0, os.path.join(parent_dir, 'mfl'))

from mfl_ast import execute_ast
from mfl_type_checker import Int, Bool, BinOp, Let, Var, Function, Apply
from mfl_parser import FunctionalParser

class TestASTInterpreter(unittest.TestCase):

    def setUp(self):
        self.parser = FunctionalParser([], {})

    def test_greater_than(self):
        # Test 5 > 3 (should be True)
        ast = BinOp(">", Int(5), Int(3))
        result = execute_ast(ast, False)
        self.assertTrue(result.value)

        # Test 3 > 5 (should be False)
        ast = BinOp(">", Int(3), Int(5))
        result = execute_ast(ast, False)
        self.assertFalse(result.value)

        # Test 5 > 5 (should be False)
        ast = BinOp(">", Int(5), Int(5))
        result = execute_ast(ast, False)
        self.assertFalse(result.value)

    def test_less_than(self):
        # Test 3 < 5 (should be True)
        ast = BinOp("<", Int(3), Int(5))
        result = execute_ast(ast, False)
        self.assertTrue(result.value)

        # Test 5 < 3 (should be False)
        ast = BinOp("<", Int(5), Int(3))
        result = execute_ast(ast, False)
        self.assertFalse(result.value)

        # Test 5 < 5 (should be False)
        ast = BinOp("<", Int(5), Int(5))
        result = execute_ast(ast, False)
        self.assertFalse(result.value)

    def test_equals(self):
        # Test 5 == 5 (should be True)
        ast = BinOp("==", Int(5), Int(5))
        result = execute_ast(ast, False)
        self.assertTrue(result.value)

        # Test 5 == 3 (should be False)
        ast = BinOp("==", Int(5), Int(3))
        result = execute_ast(ast, False)
        self.assertFalse(result.value)

    def test_less_than_equals(self):
        # Test 3 <= 5 (should be True)
        ast = BinOp("<=", Int(3), Int(5))
        result = execute_ast(ast, False)
        self.assertTrue(result.value)

        # Test 5 <= 3 (should be False)
        ast = BinOp("<=", Int(5), Int(3))
        result = execute_ast(ast, False)
        self.assertFalse(result.value)

        # Test 5 <= 5 (should be True)
        ast = BinOp("<=", Int(5), Int(5))
        result = execute_ast(ast, False)
        self.assertTrue(result.value)

    def test_greater_than_equals(self):
        # Test 5 >= 3 (should be True)
        ast = BinOp(">=", Int(5), Int(3))
        result = execute_ast(ast, False)
        self.assertTrue(result.value)

        # Test 3 >= 5 (should be False)
        ast = BinOp(">=", Int(3), Int(5))
        result = execute_ast(ast, False)
        self.assertFalse(result.value)

        # Test 5 >= 5 (should be True)
        ast = BinOp(">=", Int(5), Int(5))
        result = execute_ast(ast, False)
        self.assertTrue(result.value)

    def test_comparison_in_function(self):
        # Test a function that checks if a number is positive (> 0)
        is_positive = Let(
            Var("is_positive"),
            Function(Var("x"), BinOp(">", Var("x"), Int(0))),
            Apply(Var("is_positive"), Int(5))
        )
        result = execute_ast(is_positive, False)
        self.assertTrue(result.value)

        # Test with a negative number
        is_positive = Let(
            Var("is_positive"),
            Function(Var("x"), BinOp(">", Var("x"), Int(0))),
            Apply(Var("is_positive"), Int(-5))
        )
        result = execute_ast(is_positive, False)
        self.assertFalse(result.value)

    def test_complex_comparison(self):
        # Test a more complex expression: let max = λx.λy.if (x > y) then x else y
        # Since we don't have if-then-else, we'll test just the comparison part
        max_func = Let(
            Var("max"),
            Function(Var("x"), Function(Var("y"), BinOp(">", Var("x"), Var("y")))),
            Apply(Apply(Var("max"), Int(5)), Int(3))
        )
        result = execute_ast(max_func, False)
        self.assertTrue(result.value)

    def test_let_expression_add(self):
        ast = self.parser.parse("let add = λx.λy.(x+y) in (add 3 4)")
        result = execute_ast(ast, False)
        self.assertEqual(result, 7)

    def test_let_expression_double(self):
        ast = self.parser.parse("let double = λx.(x*2) in (double 21)")
        result = execute_ast(ast, False)
        self.assertEqual(result, 42)

    def test_let_expression_compose(self):
        ast = self.parser.parse("let compose = λf.λg.λx.(f (g x)) in let add1 = λx.(x+1) in let double = λx.(x+x) in ((compose double add1) 2)")
        result = execute_ast(ast, False)
        self.assertEqual(result, 6)


if __name__ == "__main__":
    unittest.main()
