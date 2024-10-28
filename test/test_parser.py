"""
Test cases for the functional parser with comparison operators.
"""

from mfl_parser import FunctionalParser
from mfl_type_checker import BinOp, Int, Bool, infer_j

def test_comparison_operators():
    parser = FunctionalParser([], {})

    # Test greater than
    ast = parser.parse("5 > 3")
    assert isinstance(ast, BinOp)
    assert ast.op == ">"
    assert ast.left.value == 5
    assert ast.right.value == 3

    # Test less than
    ast = parser.parse("2 < 7")
    assert isinstance(ast, BinOp)
    assert ast.op == "<"
    assert ast.left.value == 2
    assert ast.right.value == 7

    # Test equality
    ast = parser.parse("4 == 4")
    assert isinstance(ast, BinOp)
    assert ast.op == "=="
    assert ast.left.value == 4
    assert ast.right.value == 4

    # Test less than or equal
    ast = parser.parse("3 <= 3")
    assert isinstance(ast, BinOp)
    assert ast.op == "<="
    assert ast.left.value == 3
    assert ast.right.value == 3

    # Test greater than or equal
    ast = parser.parse("8 >= 5")
    assert isinstance(ast, BinOp)
    assert ast.op == ">="
    assert ast.left.value == 8
    assert ast.right.value == 5

    # Test type checking of comparison expressions
    ctx = {}
    for expr in [ast]:  # We could test all expressions, using last one for example
        type_result = infer_j(expr, ctx)
        assert str(type_result) == "bool"  # All comparison operators should return bool type

    print("All comparison operator tests passed!")

if __name__ == "__main__":
    test_comparison_operators()
