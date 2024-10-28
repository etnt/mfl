import unittest
from mfl_parser import FunctionalParser
from mfl_ast import execute_ast

class TestASTInterpreter(unittest.TestCase):

    def setUp(self):
        self.parser = FunctionalParser([], {})

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
