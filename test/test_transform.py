import unittest
import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# add ../mfl to the Python path
sys.path.insert(0, os.path.join(current_dir, '../mfl'))

from mfl_ast import (
    ASTNode, Var, Function, Apply, Let, Lets, LetBinding,
    BinOp, If, LetRec, Int
)
from mfl_transform import ASTTransformer

class TestASTTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = ASTTransformer()

    def test_multiple_bindings_to_let(self):
        # Create the test AST: let x = 3, y = 4 in x + y
        lets_ast = Lets(
            [  # Note: bindings are wrapped in a list as per the AST structure
                [
                    LetBinding(Var("x"), Int(3)),
                    LetBinding(Var("y"), Int(4))
                ]
            ],
            BinOp("+", Var("x"), Var("y"))
        )

        # Transform the AST
        result = self.transformer.multiple_bindings_to_let(lets_ast)

        # Expected structure:
        # Let(Var("x"),
        #     Int(3),
        #     Let(Var("y"),
        #         Int(4),
        #         BinOp("+", Var("x"), Var("y"))))

        # Verify the outer Let
        self.assertIsInstance(result, Let)
        self.assertEqual(result.name.name, "x")
        self.assertEqual(result.value.value, 3)

        # Verify the inner Let
        inner_let = result.body
        self.assertIsInstance(inner_let, Let)
        self.assertEqual(inner_let.name.name, "y")
        self.assertEqual(inner_let.value.value, 4)

        # Verify the innermost expression
        body = inner_let.body
        self.assertIsInstance(body, BinOp)
        self.assertEqual(body.op, "+")
        self.assertEqual(body.left.name, "x")
        self.assertEqual(body.right.name, "y")

    def test_letrec_for_core_erlang(self):
        # Create the factorial example AST:
        # letrec fac = λx.(if (x == 0) then 1 else (x * (fac (x - 1)))) in (fac 5)
        fac_ast = LetRec(
            Var("fac"),
            Function(
                Var("x"),
                If(
                    BinOp("==", Var("x"), Int(0)),
                    Int(1),
                    BinOp(
                        "*",
                        Var("x"),
                        Apply(
                            Var("fac"),
                            BinOp("-", Var("x"), Int(1))
                        )
                    )
                )
            ),
            Apply(Var("fac"), Int(5))
        )

        # Transform the AST
        result = self.transformer.letrec_for_core_erlang(fac_ast)

        # Verify the structure matches the expected transformation
        self.assertIsInstance(result, Let)
        self.assertEqual(result.name.name, "fac")

        # Verify the LetRec part
        letrec_part = result.value
        self.assertIsInstance(letrec_part, LetRec)
        self.assertTrue(letrec_part.name.name.startswith("'V"))
        self.assertTrue(letrec_part.name.name.endswith("/1"))

        # Verify the function structure
        func = letrec_part.value
        self.assertIsInstance(func, Function)
        self.assertEqual(func.arg.name, "x")

        # Verify the body contains a Let with the generated variable
        let_body = func.body
        self.assertIsInstance(let_body, Let)
        self.assertTrue(let_body.name.name.startswith("V"))

if __name__ == "__main__":
    unittest.main()
