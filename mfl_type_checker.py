"""
Hindley-Milner Type Inference System Implementation

This module implements a basic Hindley-Milner type inference system, which is a classical
algorithm for inferring types in functional programming languages. The system can automatically
deduce the types of expressions without requiring explicit type annotations.

Key Concepts:

1. Type Variables: Represented by letters (a1, a2, etc.), these are placeholders for unknown
   types that get resolved through unification.

2. Monotypes: These are either:
   - Basic types (like int, bool)
   - Type variables
   - Function types (T1 -> T2)

3. Type Schemes (Polytypes): These are types with universal quantification, allowing
   polymorphic functions like the identity function (∀a.a -> a).

4. Unification: The process of making two types equal by finding substitutions for
   type variables.

The implementation follows these key principles:
- Every expression has a type, which can be inferred from context
- Functions can be polymorphic (work with multiple types)
- Type inference is done through constraint solving (unification)
"""

import dataclasses
from typing import Dict, Any

@dataclasses.dataclass
class MonoType:
    """
    Base class for monomorphic types (types without quantifiers).
    MonoTypes can be:
    - Type variables (TyVar)
    - Type constructors (TyCon)
    """
    def find(self) -> 'MonoType':
        """
        Find the ultimate type that this type resolves to.
        Used in the unification process.
        """
        return self

    def __str__(self):
        return self.__repr__()

@dataclasses.dataclass
class TyVar(MonoType):
    """
    Represents a type variable (e.g., 'a', 'b') that can be unified with any type.
    Type variables are the core mechanism that enables polymorphic typing.

    Attributes:
        name: The name of the type variable (e.g., 'a1', 'a2')
        forwarded: If this type variable has been unified with another type,
                  this points to that type
    """
    name: str
    forwarded: MonoType = None

    def find(self) -> 'MonoType':
        """
        Follow the chain of forwarded references to find the ultimate type.
        This is crucial during unification to handle chains of type variable
        assignments.
        """
        result = self
        while isinstance(result, TyVar) and result.forwarded:
            result = result.forwarded
        return result

    def make_equal_to(self, other: MonoType):
        """
        Unify this type variable with another type.
        This is a key operation in type inference.
        """
        chain_end = self.find()
        assert isinstance(chain_end, TyVar)
        chain_end.forwarded = other

    def __repr__(self):
        if self.forwarded:
            return str(self.find())
        return self.name

@dataclasses.dataclass
class TyCon(MonoType):
    """
    Represents a type constructor, which can be:
    - A basic type (like int, bool)
    - A compound type (like list, or function types)

    Attributes:
        name: The name of the type constructor (e.g., 'int', '->' for functions)
        args: List of type arguments (empty for basic types, populated for compound types)
    """
    name: str
    args: list

    def __repr__(self):
        if not self.args:
            return self.name
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"

@dataclasses.dataclass
class Forall:
    """
    Represents a polymorphic type scheme (∀a.T).
    This allows us to express types like the identity function: ∀a.a -> a

    Attributes:
        vars: List of universally quantified type variables
        ty: The type in which these variables are quantified
    """
    vars: list
    ty: MonoType

    def __repr__(self):
        if not self.vars:
            return str(self.ty)
        vars_str = " ".join(self.vars)
        return f"∀{vars_str}.{self.ty}"

def occurs_in(var: TyVar, ty: MonoType) -> bool:
    """
    Check if a type variable occurs within a type.
    This is used to prevent infinite types during unification.
    For example, we can't unify 'a' with 'List[a]' as it would create
    an infinite type.
    """
    ty = ty.find()
    if ty == var:
        return True
    if isinstance(ty, TyCon):
        return any(occurs_in(var, arg) for arg in ty.args)
    return False

def unify_j(ty1: MonoType, ty2: MonoType):
    """
    Unify two types, making them equal by finding appropriate substitutions
    for type variables. This is the core of type inference.

    The unification algorithm:
    1. If either type is a variable, unify it with the other type
    2. If both are type constructors, they must match in name and arity
    3. Recursively unify their arguments

    Raises:
        Exception: If types cannot be unified (type error in the program)
    """
    ty1 = ty1.find()
    ty2 = ty2.find()
    if isinstance(ty1, TyVar):
        if occurs_in(ty1, ty2):
            raise Exception(f"Recursive type found: {ty1} and {ty2}")
        ty1.make_equal_to(ty2)
        return
    if isinstance(ty2, TyVar):
        return unify_j(ty2, ty1)
    if isinstance(ty1, TyCon) and isinstance(ty2, TyCon):
        if ty1.name != ty2.name or len(ty1.args) != len(ty2.args):
            raise Exception(f"Type mismatch: {ty1} and {ty2}")
        for l, r in zip(ty1.args, ty2.args):
            unify_j(l, r)

def infer_j(expr, ctx: Dict[str, Forall]) -> MonoType:
    """
    Infer the type of an expression in a given typing context.
    This is the main type inference function that handles different
    kinds of expressions.

    Args:
        expr: The expression to type check
        ctx: Typing context mapping variables to their type schemes

    Returns:
        The inferred MonoType for the expression

    The type inference rules:
    1. Variables: Look up their type scheme in the context
    2. Integers: Have type 'int'
    3. Booleans: Have type 'bool'
    4. Functions: Create fresh type variable for argument,
                 infer body type in extended context
    5. Applications: Infer function and argument types,
                    ensure function type matches argument
    6. Let bindings: Infer value type, extend context, infer body
    7. Binary operations: Handle both arithmetic and boolean operations
    8. Unary operations: Handle boolean not operation
    """
    result = fresh_tyvar()

    if isinstance(expr, Var):
        # Variables get their type from the context
        scheme = ctx.get(expr.name)
        if scheme is None:
            raise Exception(f"Unbound variable {expr.name}")
        unify_j(result, scheme.ty)

    elif isinstance(expr, Int):
        # Integer literals always have type 'int'
        unify_j(result, IntType)

    elif isinstance(expr, Bool):
        # Boolean literals always have type 'bool'
        unify_j(result, BoolType)

    elif isinstance(expr, Function):
        # For a function λx.e:
        # 1. Create fresh type variable for the argument
        arg_type = fresh_tyvar()
        # 2. Add x:arg_type to context and infer the body
        body_ctx = ctx.copy()
        body_ctx[expr.arg.name] = Forall([], arg_type)
        body_type = infer_j(expr.body, body_ctx)
        # 3. Create function type arg_type → body_type
        func_type = TyCon("->", [arg_type, body_type])
        unify_j(result, func_type)

    elif isinstance(expr, Apply):
        # For function application (f x):
        # 1. Infer types for function and argument
        func_type = infer_j(expr.func, ctx)
        arg_type = infer_j(expr.arg, ctx)
        # 2. Create fresh type variable for the result
        ret_type = fresh_tyvar()
        # 3. Unify func_type with arg_type → ret_type
        expected_func_type = TyCon("->", [arg_type, ret_type])
        unify_j(func_type, expected_func_type)
        unify_j(result, ret_type)

    elif isinstance(expr, Let):
        # For let x = e1 in e2:
        # 1. Infer type of e1
        value_type = infer_j(expr.value, ctx)
        # 2. Extend context with x:value_type and infer e2
        body_ctx = ctx.copy()
        body_ctx[expr.name.name] = Forall([], value_type)
        body_type = infer_j(expr.body, body_ctx)
        unify_j(result, body_type)

    elif isinstance(expr, BinOp):
        # For binary operations:
        # 1. Infer types of both operands
        left_type = infer_j(expr.left, ctx)
        right_type = infer_j(expr.right, ctx)

        # 2. Handle arithmetic vs boolean operations
        if expr.op in ["+", "-", "*", "/"]:
            # Arithmetic operations require integers
            unify_j(left_type, IntType)
            unify_j(right_type, IntType)
            unify_j(result, IntType)
        elif expr.op in ["&", "|"]:
            # Boolean operations require booleans
            unify_j(left_type, BoolType)
            unify_j(right_type, BoolType)
            unify_j(result, BoolType)
        else:
            raise Exception(f"Unknown binary operator: {expr.op}")

    elif isinstance(expr, UnaryOp):
        # For unary operations (currently only boolean not):
        # 1. Infer type of operand
        operand_type = infer_j(expr.operand, ctx)

        # 2. Handle not operation
        if expr.op == "!":
            # Not operation requires boolean operand
            unify_j(operand_type, BoolType)
            unify_j(result, BoolType)
        else:
            raise Exception(f"Unknown unary operator: {expr.op}")

    else:
        raise Exception(f"Unknown expression type: {type(expr)}")

    return result.find()

# Expression Classes
# These classes represent the abstract syntax tree of our simple language

class ASTNode:
    """Base class for all AST nodes with raw structure printing capability"""
    def raw_structure(self):
        """Return the raw AST structure as a string"""
        if isinstance(self, Var):
            return f'Var("{self.name}")'
        elif isinstance(self, Int):
            return f'Int({self.value})'
        elif isinstance(self, Bool):
            return f'Bool({self.value})'
        elif isinstance(self, Function):
            return f'Function({self.arg.raw_structure()}, {self.body.raw_structure()})'
        elif isinstance(self, Apply):
            return f'Apply({self.func.raw_structure()}, {self.arg.raw_structure()})'
        elif isinstance(self, Let):
            return f'Let({self.name.raw_structure()}, {self.value.raw_structure()}, {self.body.raw_structure()})'
        elif isinstance(self, BinOp):
            return f'BinOp("{self.op}", {self.left.raw_structure()}, {self.right.raw_structure()})'
        elif isinstance(self, UnaryOp):
            return f'UnaryOp("{self.op}", {self.operand.raw_structure()})'
        return str(self)

@dataclasses.dataclass
class Var(ASTNode):
    """
    Represents a variable reference in the program.
    Example: x
    """
    name: str

    def __repr__(self):
        return self.name

@dataclasses.dataclass
class Int(ASTNode):
    """
    Represents an integer literal.
    Example: 42
    """
    value: int

    def __repr__(self):
        return str(self.value)

@dataclasses.dataclass
class Bool(ASTNode):
    """
    Represents a boolean literal.
    Example: True, False
    """
    value: bool

    def __repr__(self):
        return str(self.value)

@dataclasses.dataclass
class Function(ASTNode):
    """
    Represents a lambda function.
    Example: λx.x (the identity function)
    """
    arg: Var
    body: Any  # Expression for the body

    def __repr__(self):
        return f"λ{self.arg}.{self.body}"

@dataclasses.dataclass
class Apply(ASTNode):
    """
    Represents function application.
    Example: (f x) applies function f to argument x
    """
    func: Any  # The function being applied
    arg: Any   # The argument being passed

    def __repr__(self):
        return f"({self.func} {self.arg})"

@dataclasses.dataclass
class Let(ASTNode):
    """
    Represents let bindings.
    Example: let x = e1 in e2
    Allows local variable definitions
    """
    name: Var
    value: Any  # Value expression
    body: Any   # Body expression where the value is bound

    def __repr__(self):
        return f"let {self.name} = {self.value} in {self.body}"

@dataclasses.dataclass
class BinOp(ASTNode):
    """
    Represents binary operations (+, -, *, /, &, |).
    Example: x + y, a & b
    """
    op: str  # One of: '+', '-', '*', '/', '&', '|'
    left: Any
    right: Any

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"

@dataclasses.dataclass
class UnaryOp(ASTNode):
    """
    Represents unary operations (!).
    Example: !x
    """
    op: str  # Currently only '!'
    operand: Any

    def __repr__(self):
        return f"{self.op}{self.operand}"

# Type Variable Generation

tyvar_counter = 0

def fresh_tyvar(prefix="a"):
    """
    Generate a fresh type variable with a unique name.
    This ensures each type variable in our inference is unique.
    """
    global tyvar_counter
    tyvar_counter += 1
    return TyVar(name=f"{prefix}{tyvar_counter}")

# Primitive Types
IntType = TyCon("int", [])    # The integer type
BoolType = TyCon("bool", [])  # The boolean type


# ---------------------------------------
# Test Cases demonstrating type inference
# ---------------------------------------

def test_var():
    """Test type inference for variables"""
    ctx = {"x": Forall([], IntType)}  # Declare x as an integer
    expr = Var("x")
    inferred_type = infer_j(expr, ctx)
    print(f"Variable 'x': {expr}")
    print(f"Raw AST structure: {expr.raw_structure()}")
    print(expr)
    print(f"Inferred type: {inferred_type}\n")

def test_int():
    """Test type inference for integer literals"""
    ctx = {}
    expr = Int(42)
    inferred_type = infer_j(expr, ctx)
    print(f"Integer literal: {expr}")
    print(f"Raw AST structure: {expr.raw_structure()}")
    print(f"Inferred type: {inferred_type}\n")

def test_bool():
    """Test type inference for boolean literals"""
    ctx = {}
    expr = Bool(True)
    inferred_type = infer_j(expr, ctx)
    print(f"Boolean literal: {expr}")
    print(f"Raw AST structure: {expr.raw_structure()}")
    print(f"Inferred type: {inferred_type}\n")

def test_identity_function():
    """
    Test type inference for the identity function.
    The identity function λx.x should have type ∀a.a -> a,
    meaning it works for any type.
    """
    ctx = {}
    expr = Function(Var("x"), Var("x"))  # λx.x
    inferred_type = infer_j(expr, ctx)
    print(f"Identity function: {expr}")
    print(f"Raw AST structure: {expr.raw_structure()}")
    print(f"Inferred type: {inferred_type}\n")

def test_function_application():
    """
    Test type inference for function application.
    Applying the identity function to an integer should yield an integer.
    """
    ctx = {}
    func = Function(Var("x"), Var("x"))  # λx.x
    expr = Apply(func, Int(42))  # (λx.x)(42)
    inferred_type = infer_j(expr, ctx)
    print(f"Function application: {expr}")
    print(f"Raw AST structure: {expr.raw_structure()}")
    print(f"Inferred type: {inferred_type}\n")

def test_let_binding():
    """
    Test type inference for let bindings.
    This demonstrates how we can bind the identity function and use it.
    """
    ctx = {}
    expr = Let(Var("id"), Function(Var("x"), Var("x")), Apply(Var("id"), Int(42)))
    inferred_type = infer_j(expr, ctx)
    print(f"Let binding: {expr}")
    print(f"Raw AST structure: {expr.raw_structure()}")
    print(f"Inferred type: {inferred_type}\n")
    

def test_arithmetic():
    """
    Test type inference for arithmetic operations.
    All arithmetic operations should work with integers and produce integers.
    """
    ctx = {}
    # Test addition
    expr1 = BinOp("+", Int(5), Int(3))
    type1 = infer_j(expr1, ctx)
    print(f"Addition: {expr1}")
    print(f"Raw AST structure: {expr1.raw_structure()}")
    print(f"Inferred type: {type1}\n")

    # Test multiplication with a more complex expression
    expr2 = BinOp("*", 
                  BinOp("+", Int(2), Int(3)),
                  BinOp("-", Int(10), Int(5)))
    type2 = infer_j(expr2, ctx)
    print(f"Complex arithmetic: {expr2}")
    print(f"Raw AST structure: {expr2.raw_structure()}")
    print(f"Inferred type: {type2}\n")

    # Test division
    expr3 = BinOp("/", Int(10), Int(2))
    type3 = infer_j(expr3, ctx)
    print(f"Division: {expr3}")
    print(f"Raw AST structure: {expr3.raw_structure()}")
    print(f"Inferred type: {type3}\n")

def test_boolean():
    """
    Test type inference for boolean operations.
    All boolean operations should work with booleans and produce booleans.
    """
    ctx = {}
    # Test and
    expr1 = BinOp("&", Bool(True), Bool(False))
    type1 = infer_j(expr1, ctx)
    print(f"And: {expr1}")
    print(f"Raw AST structure: {expr1.raw_structure()}")
    print(f"Inferred type: {type1}\n")

    # Test or
    expr2 = BinOp("|", Bool(True), Bool(False))
    type2 = infer_j(expr2, ctx)
    print(f"Or: {expr2}")
    print(f"Raw AST structure: {expr2.raw_structure()}")
    print(f"Inferred type: {type2}\n")

    # Test not
    expr3 = UnaryOp("!", Bool(True))
    type3 = infer_j(expr3, ctx)
    print(f"Not: {expr3}")
    print(f"Raw AST structure: {expr3.raw_structure()}")
    print(f"Inferred type: {type3}\n")

    # Test complex boolean expression
    expr4 = BinOp("|",
                  BinOp("&", Bool(True), Bool(False)),
                  UnaryOp("!", Bool(True)))
    type4 = infer_j(expr4, ctx)
    print(f"Complex boolean: {expr4}")
    print(f"Raw AST structure: {expr4.raw_structure()}")
    print(f"Inferred type: {type4}\n")

if __name__ == "__main__":
    # Run all test cases
    print("Type Inference Examples:\n")
    test_var()
    test_int()
    test_bool()
    test_identity_function()
    test_function_application()
    test_let_binding()
    test_arithmetic()
    test_boolean()
