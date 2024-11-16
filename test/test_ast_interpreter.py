import unittest
import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# add ../mfl to the Python path
sys.path.insert(0, os.path.join(current_dir, '../mfl'))

from mfl_ply_parser import parser
from mfl_ast import execute_ast
from mfl_type_checker import Int, Bool, BinOp, Let, Var, Function, Apply, If
from mfl_transform import ASTTransformer

class TestASTInterpreter(unittest.TestCase):

    def setUp(self):
        self.parser = parser

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
        max_func = Let(
            Var("max"),
            Function(Var("x"), Function(Var("y"), 
                If(BinOp(">", Var("x"), Var("y")), Var("x"), Var("y")))),
            Apply(Apply(Var("max"), Int(5)), Int(3))
        )
        result = execute_ast(max_func, False)
        self.assertEqual(result.value, 5)

    def test_if_expression_basic(self):
        # Test if true then 1 else 0
        ast = If(Bool(True), Int(1), Int(0))
        result = execute_ast(ast, False)
        self.assertEqual(result.value, 1)

        # Test if false then 1 else 0
        ast = If(Bool(False), Int(1), Int(0))
        result = execute_ast(ast, False)
        self.assertEqual(result.value, 0)

    def test_if_expression_with_comparison(self):
        # Test if 5 > 3 then 1 else 0
        ast = If(BinOp(">", Int(5), Int(3)), Int(1), Int(0))
        result = execute_ast(ast, False)
        self.assertEqual(result.value, 1)

        # Test if 3 > 5 then 1 else 0
        ast = If(BinOp(">", Int(3), Int(5)), Int(1), Int(0))
        result = execute_ast(ast, False)
        self.assertEqual(result.value, 0)

    def test_if_expression_in_function(self):
        # Test let abs = λx.if (x < 0) then (0 - x) else x
        abs_func = Let(
            Var("abs"),
            Function(Var("x"), 
                If(BinOp("<", Var("x"), Int(0)),
                   BinOp("-", Int(0), Var("x")),
                   Var("x"))),
            Apply(Var("abs"), Int(-5))
        )
        result = execute_ast(abs_func, False)
        self.assertEqual(result.value, 5)

        # Test with positive number
        abs_func = Let(
            Var("abs"),
            Function(Var("x"), 
                If(BinOp("<", Var("x"), Int(0)),
                   BinOp("-", Int(0), Var("x")),
                   Var("x"))),
            Apply(Var("abs"), Int(5))
        )
        result = execute_ast(abs_func, False)
        self.assertEqual(result.value, 5)

    def test_letrec_to_let(self):
        # Test recursive function using letrec when transformed to let + y-combinator
        ast = self.parser.parse("letrec fac = λx.(if (x == 0) then 1 else (x * (fac (x - 1)))) in (fac 5)")
        ast = ASTTransformer.transform_letrec_to_let(ast)
        result = execute_ast(ast)
        self.assertEqual(result.value, 120)

    def test_letrec_to_let_fibonacci(self):
        # Test recursive function using letrec when transformed to let + y-combinator
        ast = self.parser.parse("letrec fibonacci = λx.(if (x == 0) then 0 else (if (x == 1) then 1 else (fibonacci (x - 1) + (fibonacci (x - 2))))) in (fibonacci 10)")
        ast = ASTTransformer.transform_letrec_to_let(ast)
        result = execute_ast(ast)
        self.assertEqual(result.value, 34)


if __name__ == "__main__":
    unittest.main()
