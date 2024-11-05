import unittest
import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# add ../mfl to the Python path
sys.path.insert(0, os.path.join(current_dir, '../mfl'))

from mfl_type_checker import (
    Var, Int, Bool, Function, Apply, Let, BinOp, UnaryOp,
    Forall, IntType, BoolType, infer_j, If, LetRec
)

class TestTypeChecker(unittest.TestCase):
    def test_var(self):
        """Test type inference for variables"""
        ctx = {"x": Forall([], IntType)}  # Declare x as an integer
        expr = Var("x")
        inferred_type = infer_j(expr, ctx)
        self.assertEqual(str(inferred_type), "int")
        self.assertEqual(expr.typed_structure(), 'Var<int>("x")')

    def test_int(self):
        """Test type inference for integer literals"""
        ctx = {}
        expr = Int(42)
        inferred_type = infer_j(expr, ctx)
        self.assertEqual(str(inferred_type), "int")
        self.assertEqual(expr.typed_structure(), 'Int<int>(42)')

    def test_bool(self):
        """Test type inference for boolean literals"""
        ctx = {}
        expr = Bool(True)
        inferred_type = infer_j(expr, ctx)
        self.assertEqual(str(inferred_type), "bool")
        self.assertEqual(expr.typed_structure(), 'Bool<bool>(True)')

    def test_identity_function(self):
        """
        Test type inference for the identity function.
        The identity function λx.x should have type ∀a.a -> a,
        meaning it works for any type.
        """
        ctx = {}
        expr = Function(Var("x"), Var("x"))  # λx.x
        inferred_type = infer_j(expr, ctx)
        self.assertTrue(str(inferred_type).startswith("->"))
        self.assertEqual(expr.arg.type, expr.body.type)

    def test_function_application(self):
        """
        Test type inference for function application.
        Applying the identity function to an integer should yield an integer.
        """
        ctx = {}
        func = Function(Var("x"), Var("x"))  # λx.x
        expr = Apply(func, Int(42))  # (λx.x)(42)
        inferred_type = infer_j(expr, ctx)
        self.assertEqual(str(inferred_type), "int")
        self.assertEqual(expr.arg.type, IntType)

    def test_let_binding(self):
        """
        Test type inference for let bindings.
        This demonstrates how we can bind the identity function and use it.
        """
        ctx = {}
        expr = Let(Var("id"), Function(Var("x"), Var("x")), Apply(Var("id"), Int(42)))
        inferred_type = infer_j(expr, ctx)
        self.assertEqual(str(inferred_type), "int")
        self.assertEqual(expr.body.type, IntType)


    def test_fibonacci(self):
        """
        Test type inference for the Fibonacci function.
        """
        ctx = {}
        expr = LetRec(Var("fibonacci"), Function(Var("x"), If(BinOp("==", Var("x"), Int(0)), Int(0), If(BinOp("==", Var("x"), Int(1)), Int(1), BinOp("+", Apply(Var("fibonacci"), BinOp("-", Var("x"), Int(1))), Apply(Var("fibonacci"), BinOp("-", Var("x"), Int(2))))))), Apply(Var("fibonacci"), Int(5)))
        inferred_type = infer_j(expr, ctx)
        self.assertEqual(str(inferred_type), "int")
        self.assertEqual(expr.body.type, IntType)

    def test_double_function(self):
        """
        Test type inference for the double function.
        Expression: let double = λx.(x*2) in (double 21)
        """
        ctx = {}
        # Create the double function: λx.(x*2)
        double_func = Function(
            Var("x"),
            BinOp("*", Var("x"), Int(2))
        )
        # Create the full expression: let double = λx.(x*2) in (double 21)
        expr = Let(
            Var("double"),
            double_func,
            Apply(Var("double"), Int(21))
        )

        inferred_type = infer_j(expr, ctx)

        # Check the overall type
        self.assertEqual(str(inferred_type), "int")

        # Check the function's type
        self.assertEqual(str(expr.value.type), "->(int, int)")

        # Check that all nodes have proper type annotations
        self.assertEqual(
            expr.typed_structure(),
            'Let<int>(Var<->(int, int)>("double"), ' +
            'Function<->(int, int)>(Var<int>("x"), BinOp<int>("*", Var<int>("x"), Int<int>(2))), ' +
            'Apply<int>(Var<->(int, int)>("double"), Int<int>(21)))'
        )

        # Check specific parts of the expression
        self.assertEqual(str(expr.name.type), "->(int, int)")  # double's type
        self.assertEqual(str(expr.value.arg.type), "int")      # x's type
        self.assertEqual(str(expr.value.body.type), "int")     # (x*2)'s type
        self.assertEqual(str(expr.body.type), "int")           # (double 21)'s type

    def test_arithmetic(self):
        """Test type inference for arithmetic operations"""
        ctx = {}

        # Test addition
        expr1 = BinOp("+", Int(5), Int(3))
        type1 = infer_j(expr1, ctx)
        self.assertEqual(str(type1), "int")
        self.assertEqual(expr1.left.type, IntType)
        self.assertEqual(expr1.right.type, IntType)

        # Test multiplication with a more complex expression
        expr2 = BinOp("*", 
                     BinOp("+", Int(2), Int(3)),
                     BinOp("-", Int(10), Int(5)))
        type2 = infer_j(expr2, ctx)
        self.assertEqual(str(type2), "int")
        self.assertEqual(expr2.left.type, IntType)
        self.assertEqual(expr2.right.type, IntType)

        # Test division
        expr3 = BinOp("/", Int(10), Int(2))
        type3 = infer_j(expr3, ctx)
        self.assertEqual(str(type3), "int")
        self.assertEqual(expr3.left.type, IntType)
        self.assertEqual(expr3.right.type, IntType)

    def test_boolean(self):
        """Test type inference for boolean operations"""
        ctx = {}

        # Test and
        expr1 = BinOp("&", Bool(True), Bool(False))
        type1 = infer_j(expr1, ctx)
        self.assertEqual(str(type1), "bool")
        self.assertEqual(expr1.left.type, BoolType)
        self.assertEqual(expr1.right.type, BoolType)

        # Test or
        expr2 = BinOp("|", Bool(True), Bool(False))
        type2 = infer_j(expr2, ctx)
        self.assertEqual(str(type2), "bool")
        self.assertEqual(expr2.left.type, BoolType)
        self.assertEqual(expr2.right.type, BoolType)

        # Test not
        expr3 = UnaryOp("!", Bool(True))
        type3 = infer_j(expr3, ctx)
        self.assertEqual(str(type3), "bool")
        self.assertEqual(expr3.operand.type, BoolType)

        # Test complex boolean expression
        expr4 = BinOp("|",
                     BinOp("&", Bool(True), Bool(False)),
                     UnaryOp("!", Bool(True)))
        type4 = infer_j(expr4, ctx)
        self.assertEqual(str(type4), "bool")
        self.assertEqual(expr4.left.type, BoolType)
        self.assertEqual(expr4.right.type, BoolType)

    def test_comparison(self):
        """Test type inference for comparison operations"""
        ctx = {}

        # Test greater than
        expr1 = BinOp(">", Int(5), Int(3))
        type1 = infer_j(expr1, ctx)
        self.assertEqual(str(type1), "bool")
        self.assertEqual(expr1.left.type, IntType)
        self.assertEqual(expr1.right.type, IntType)

        # Test less than
        expr2 = BinOp("<", Int(2), Int(7))
        type2 = infer_j(expr2, ctx)
        self.assertEqual(str(type2), "bool")
        self.assertEqual(expr2.left.type, IntType)
        self.assertEqual(expr2.right.type, IntType)

        # Test equality
        expr3 = BinOp("==", Int(4), Int(4))
        type3 = infer_j(expr3, ctx)
        self.assertEqual(str(type3), "bool")
        self.assertEqual(expr3.left.type, IntType)
        self.assertEqual(expr3.right.type, IntType)

        # Test less than or equal
        expr4 = BinOp("<=", Int(3), Int(3))
        type4 = infer_j(expr4, ctx)
        self.assertEqual(str(type4), "bool")
        self.assertEqual(expr4.left.type, IntType)
        self.assertEqual(expr4.right.type, IntType)

        # Test greater than or equal
        expr5 = BinOp(">=", Int(8), Int(5))
        type5 = infer_j(expr5, ctx)
        self.assertEqual(str(type5), "bool")
        self.assertEqual(expr5.left.type, IntType)
        self.assertEqual(expr5.right.type, IntType)

if __name__ == '__main__':
    unittest.main()
