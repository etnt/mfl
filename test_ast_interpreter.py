"""
Unit tests for the AST interpreter, focusing on comparison operators.
"""

import unittest
from mfl_ast import ASTInterpreter
from mfl_type_checker import Int, Bool, BinOp, Let, Var, Function, Apply

class TestASTInterpreter(unittest.TestCase):
    def setUp(self):
        self.interpreter = ASTInterpreter()

    def test_greater_than(self):
        # Test 5 > 3 (should be True)
        ast = BinOp(">", Int(5), Int(3))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertTrue(result.value)

        # Test 3 > 5 (should be False)
        ast = BinOp(">", Int(3), Int(5))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertFalse(result.value)

        # Test 5 > 5 (should be False)
        ast = BinOp(">", Int(5), Int(5))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertFalse(result.value)

    def test_less_than(self):
        # Test 3 < 5 (should be True)
        ast = BinOp("<", Int(3), Int(5))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertTrue(result.value)

        # Test 5 < 3 (should be False)
        ast = BinOp("<", Int(5), Int(3))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertFalse(result.value)

        # Test 5 < 5 (should be False)
        ast = BinOp("<", Int(5), Int(5))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertFalse(result.value)

    def test_equals(self):
        # Test 5 == 5 (should be True)
        ast = BinOp("==", Int(5), Int(5))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertTrue(result.value)

        # Test 5 == 3 (should be False)
        ast = BinOp("==", Int(5), Int(3))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertFalse(result.value)

    def test_less_than_equals(self):
        # Test 3 <= 5 (should be True)
        ast = BinOp("<=", Int(3), Int(5))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertTrue(result.value)

        # Test 5 <= 3 (should be False)
        ast = BinOp("<=", Int(5), Int(3))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertFalse(result.value)

        # Test 5 <= 5 (should be True)
        ast = BinOp("<=", Int(5), Int(5))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertTrue(result.value)

    def test_greater_than_equals(self):
        # Test 5 >= 3 (should be True)
        ast = BinOp(">=", Int(5), Int(3))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertTrue(result.value)

        # Test 3 >= 5 (should be False)
        ast = BinOp(">=", Int(3), Int(5))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertFalse(result.value)

        # Test 5 >= 5 (should be True)
        ast = BinOp(">=", Int(5), Int(5))
        result = self.interpreter.eval(ast)
        self.assertIsInstance(result, Bool)
        self.assertTrue(result.value)

    def test_comparison_in_function(self):
        # Test a function that checks if a number is positive (> 0)
        is_positive = Let(
            Var("is_positive"),
            Function(Var("x"), BinOp(">", Var("x"), Int(0))),
            Apply(Var("is_positive"), Int(5))
        )
        result = self.interpreter.eval(is_positive)
        self.assertIsInstance(result, Bool)
        self.assertTrue(result.value)

        # Test with a negative number
        is_positive = Let(
            Var("is_positive"),
            Function(Var("x"), BinOp(">", Var("x"), Int(0))),
            Apply(Var("is_positive"), Int(-5))
        )
        result = self.interpreter.eval(is_positive)
        self.assertIsInstance(result, Bool)
        self.assertFalse(result.value)

    def test_complex_comparison(self):
        # Test a more complex expression: let max = λx.λy.if (x > y) then x else y
        # Since we don't have if-then-else, we'll test just the comparison part
        max_func = Let(
            Var("max"),
            Function(Var("x"), Function(Var("y"), BinOp(">", Var("x"), Var("y")))),
            Apply(Apply(Var("max"), Int(5)), Int(3))
        )
        result = self.interpreter.eval(max_func)
        self.assertIsInstance(result, Bool)
        self.assertTrue(result.value)

if __name__ == '__main__':
    unittest.main()
