"""Unit tests for mfl_secd.py"""

import unittest
from mfl_secd import SECDMachine
from mfl_parser import FunctionalParser
from mfl_type_checker import infer_j
from mfl_secd import execute_ast

class TestSECDMachine(unittest.TestCase):

    def setUp(self):
        self.parser = FunctionalParser([], {})
        self.machine = SECDMachine()

    #def test_simple_expression(self):
    #    ast = self.parser.parse("21")
    #    infer_j(ast, {}) #type check
    #    result = execute_ast(ast, False)
    #    self.assertEqual(result, 21)

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


if __name__ == "__main__":
    unittest.main()
