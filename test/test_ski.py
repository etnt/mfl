"""Unit tests for the SKI combinator machine implementation."""

import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# add ../mfl to the Python path
sys.path.insert(0, os.path.join(current_dir, '../mfl'))


import unittest
from mfl_ply_parser import parser
from mfl_ski import SKIMachine, execute_ast
from mfl_transform import ASTTransformer

class TestSKIMachine(unittest.TestCase):

    def setUp(self):
        self.parser = parser
        self.machine = SKIMachine(verbose=False)

    def test_basic_combinators(self):
        # Test I combinator: I x → x
        ast = self.parser.parse("let id = λx.x in (id 42)")
        result = execute_ast(ast)
        self.assertEqual(result.value, 42)

        # Test K combinator: K x y → x
        ast = self.parser.parse("let const = λx.λy.x in ((const 42) 7)")
        result = execute_ast(ast)
        self.assertEqual(result.value, 42)

        ast = self.parser.parse("let s = λf.λg.λx.(f (g x)) in let add1 = λx.(x+1) in let double = λx.(x*2) in (((s add1) double) 3)")
        result = execute_ast(ast)
        self.assertEqual(result.value, 7)  # (3+1) + (3*2) = 4 + 6 = 10

    def test_optimization_combinators(self):
        # Test B combinator: B f g x → f (g x)
        ast = self.parser.parse("let compose = λf.λg.λx.(f (g x)) in let add1 = λx.(x+1) in let double = λx.(x*2) in (((compose double) add1) 3)")
        result = execute_ast(ast)
        self.assertEqual(result.value, 8)  # double(add1(3)) = double(4) = 8

        # Test C combinator: C f g x → f x g
        ast = self.parser.parse("let flip = λf.λg.λx.((f x) g) in let div = λx.λy.(x/y) in (((flip div) 2) 10)")
        result = execute_ast(ast)
        self.assertEqual(result.value, 5)  # flip div 2 10 = div 10 2 = 5

    def test_arithmetic_operations(self):
        # Test addition
        ast = self.parser.parse("5 + 3")
        result = execute_ast(ast)
        self.assertEqual(result.value, 8)

        # Test subtraction
        ast = self.parser.parse("10 - 4")
        result = execute_ast(ast)
        self.assertEqual(result.value, 6)

        # Test multiplication
        ast = self.parser.parse("6 * 7")
        result = execute_ast(ast)
        self.assertEqual(result.value, 42)

        # Test division
        ast = self.parser.parse("15 / 3")
        result = execute_ast(ast)
        self.assertEqual(result.value, 5)

    def test_boolean_operations(self):
        # Test greater than
        ast = self.parser.parse("5 > 3")
        result = execute_ast(ast)
        self.assertTrue(result.value)

        # Test less than
        ast = self.parser.parse("2 < 7")
        result = execute_ast(ast)
        self.assertTrue(result.value)

        # Test equality
        ast = self.parser.parse("4 == 4")
        result = execute_ast(ast)
        self.assertTrue(result.value)

        # Test greater than or equal
        ast = self.parser.parse("8 >= 5")
        result = execute_ast(ast)
        self.assertTrue(result.value)

        # Test less than or equal
        ast = self.parser.parse("3 <= 3")
        result = execute_ast(ast)
        self.assertTrue(result.value)

    def test_complex_expressions(self):
        # Test composition of arithmetic and boolean operations
        ast = self.parser.parse("let compose = λf.λg.λx.(f (g x)) in let add1 = λx.(x+1) in let isEven = λx.((x/2)*2 == x) in ((compose isEven add1) 5)")
        result = execute_ast(ast)
        self.assertTrue(result.value)  # add1(5) = 6, isEven(6) = true

        # Test higher-order function with arithmetic
        ast = self.parser.parse("let twice = λf.λx.(f (f x)) in let add3 = λx.(x+3) in ((twice add3) 7)")
        result = execute_ast(ast)
        self.assertEqual(result.value, 13)  # add3(add3(7)) = add3(10) = 13

        # Test conditional expression using boolean operations
        ast = self.parser.parse("let max = λx.λy.(((if x > y then 1 else 0) * x) + ((if x <= y then 1 else 0) * y)) in ((max 15) 10)")
        result = execute_ast(ast)
        self.assertEqual(result.value, 15)

    def test_letrec_to_let(self):
        # Test recursive function using letrec when transformed to let + y-combinator
        ast = self.parser.parse("letrec fac = λx.(if (x == 0) then 1 else (x * (fac (x - 1)))) in (fac 5)")
        ast = ASTTransformer.transform_letrec_to_let(ast)
        result = execute_ast(ast)
        self.assertEqual(result.value, 120)

if __name__ == "__main__":
    unittest.main()
