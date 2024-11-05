"""
This module defines the Abstract Syntax Tree (AST) nodes and provides an interpreter
for a simple functional language.
"""
import dataclasses
from typing import Union, Dict, Any, Optional, TypeVar

# Type variable for monomorphic types
MonoType = TypeVar('MonoType')

# AST Node Classes
class ASTNode:
    """Base class for all AST nodes with raw structure printing capability"""
    def __init__(self):
        # To hold type information
        self.type: Optional['MonoType'] = None

        # To hold LLVM IR information
        self.llvm: Dict[str, str] = {}

    # The method allows you to add or change an attribute of an object using
    # strings to specify the attribute's name.
    def _set_attr(self, name, value):
        self.__dict__[name] = value

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
        elif isinstance(self, LetRec):
            return f'LetRec({self.name.raw_structure()}, {self.value.raw_structure()}, {self.body.raw_structure()})'
        elif isinstance(self, If):
            return f'If({self.cond.raw_structure()}, {self.then_expr.raw_structure()}, {self.else_expr.raw_structure()})'
        elif isinstance(self, BinOp):
            return f'BinOp("{self.op}", {self.left.raw_structure()}, {self.right.raw_structure()})'
        elif isinstance(self, UnaryOp):
            return f'UnaryOp("{self.op}", {self.operand.raw_structure()})'
        return str(self)

    def typed_structure(self):
        """Return the raw AST structure as a string with type annotations"""
        type_str = f"<{self.type}>" if self.type else "<untyped>"
        if isinstance(self, Var):
            return f'Var{type_str}("{self.name}")'
        elif isinstance(self, Int):
            return f'Int{type_str}({self.value})'
        elif isinstance(self, Bool):
            return f'Bool{type_str}({self.value})'
        elif isinstance(self, Function):
            return f'Function{type_str}({self.arg.typed_structure()}, {self.body.typed_structure()})'
        elif isinstance(self, Apply):
            return f'Apply{type_str}({self.func.typed_structure()}, {self.arg.typed_structure()})'
        elif isinstance(self, Let):
            return f'Let{type_str}({self.name.typed_structure()}, {self.value.typed_structure()}, {self.body.typed_structure()})'
        elif isinstance(self, LetRec):
            return f'LetRec{type_str}({self.name.typed_structure()}, {self.value.typed_structure()}, {self.body.typed_structure()})'
        elif isinstance(self, If):
            return f'If{type_str}({self.cond.typed_structure()}, {self.then_expr.typed_structure()}, {self.else_expr.typed_structure()})'
        elif isinstance(self, BinOp):
            return f'BinOp{type_str}("{self.op}", {self.left.typed_structure()}, {self.right.typed_structure()})'
        elif isinstance(self, UnaryOp):
            return f'UnaryOp{type_str}("{self.op}", {self.operand.typed_structure()})'
        return str(self)

@dataclasses.dataclass
class Var(ASTNode):
    """
    Represents a variable reference in the program.
    Example: x
    """
    name: str

    def __post_init__(self):
        super().__init__()

    def __repr__(self):
        return self.name

@dataclasses.dataclass
class Int(ASTNode):
    """
    Represents an integer literal.
    Example: 42
    """
    value: int

    def __post_init__(self):
        super().__init__()

    def __eq__(self, other):  # Overload the equality operator
        return self.value == other

    def __repr__(self):
        return str(self.value)

@dataclasses.dataclass
class Bool(ASTNode):
    """
    Represents a boolean literal.
    Example: True, False
    """
    value: bool

    def __post_init__(self):
        super().__init__()

    def __eq__(self, other):  # Overload the equality operator
        return self.value == other

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

    def __post_init__(self):
        super().__init__()

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

    def __post_init__(self):
        super().__init__()

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

    def __post_init__(self):
        super().__init__()

    def __repr__(self):
        return f"let {self.name} = {self.value} in {self.body}"

@dataclasses.dataclass
class LetRec(ASTNode):
    """
    Represents letrec bindings.
    Example: letrec x = e1 in e2
    Allows recursive function definitions
    """
    name: Var
    value: Any  # Value expression
    body: Any   # Body expression where the value is bound

    def __post_init__(self):
        super().__init__()

    def __repr__(self):
        return f"letrec {self.name} = {self.value} in {self.body}"

@dataclasses.dataclass
class If(ASTNode):
    """
    Represents a conditional expression.
    Example: if x > 0 then x else 0
    """
    cond: Any      # Condition expression
    then_expr: Any # Expression for the 'then' branch
    else_expr: Any # Expression for the 'else' branch

    def __post_init__(self):
        super().__init__()

    def __repr__(self):
        return f"if {self.cond} then {self.then_expr} else {self.else_expr}"

@dataclasses.dataclass
class BinOp(ASTNode):
    """
    Represents binary operations (+, -, *, /, &, |, >, <, ==, <=, >=).
    Example: x + y, a & b, x > y
    """
    op: str  # One of: '+', '-', '*', '/', '&', '|', '>', '<', '==', '<=', '>='
    left: Any
    right: Any

    def __post_init__(self):
        super().__init__()

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

    def __post_init__(self):
        super().__init__()

    def __repr__(self):
        return f"{self.op}{self.operand}"

# AST Interpreter
class ASTInterpreter:
    """
    An interpreter for the AST. It maintains an environment to store variable bindings.
    """
    def __init__(self, verbose: bool = False):
        """Initializes the interpreter with an empty environment."""
        self.verbose = verbose
        self.env: Dict[str, ASTNode] = {}

    def eval(self, node: ASTNode) -> ASTNode:
        """
        Evaluates the given AST node.
        """
        if self.verbose:
            print(f"Evaluating {type(node).__name__}: {node}")

        # Variable lookup
        if isinstance(node, Var):
            if self.verbose:
                print(f"Looking up variable {node.name}")
            result = self.env.get(node.name, node)
            if self.verbose:
                print(f"Variable lookup result: {result}")
            return result

        # Function handling - functions are self-evaluating
        elif isinstance(node, Function):
            if self.verbose:
                print(f"Function is self-evaluating: {node}")
            return node

        # Function application
        elif isinstance(node, Apply):
            if self.verbose:
                print(f"Evaluating function: {node.func}")
            func = self.eval(node.func)
            if self.verbose:
                print(f"Evaluating argument: {node.arg}")
            arg = self.eval(node.arg)

            if isinstance(func, Function):
                # Curried function handling
                if isinstance(func.body, Function):
                    result = self.substitute(func.body, func.arg, arg)
                    if self.verbose:
                        print(f"Curried function result: {result}")
                    return result

                # Regular function application
                if self.verbose:
                    print(f"Substituting {arg} for {func.arg} in {func.body}")
                substituted = self.substitute(func.body, func.arg, arg)
                if self.verbose:
                    print(f"After substitution: {substituted}")
                return self.eval(substituted)

            return Apply(func, arg)

        # Let expression handling
        elif isinstance(node, Let):
            if self.verbose:
                print(f"Evaluating let binding for {node.name.name}")
            value = self.eval(node.value)
            self.env[node.name.name] = value
            if self.verbose:
                print(f"Evaluating let body: {node.body}")
            return self.eval(node.body)

        # LetRec expression handling
        elif isinstance(node, LetRec):
            if self.verbose:
                print(f"Evaluating letrec binding for {node.name.name}")

            # First ensure we have a function value
            if not isinstance(node.value, Function):
                raise ValueError("LetRec value must be a function")

            # Create a recursive closure by adding the function to the environment
            # before evaluating its body
            func = node.value
            self.env[node.name.name] = func

            if self.verbose:
                print(f"Evaluating letrec body: {node.body}")
            return self.eval(node.body)

        # If expression handling
        elif isinstance(node, If):
            if self.verbose:
                print(f"Evaluating if condition: {node.cond}")
            cond_result = self.eval(node.cond)
            if isinstance(cond_result, Bool):
                if cond_result.value:
                    if self.verbose:
                        print(f"Condition is true, evaluating then branch: {node.then_expr}")
                    return self.eval(node.then_expr)
                else:
                    if self.verbose:
                        print(f"Condition is false, evaluating else branch: {node.else_expr}")
                    return self.eval(node.else_expr)
            return If(cond_result, node.then_expr, node.else_expr)

        # Binary operation handling
        elif isinstance(node, BinOp):
            if self.verbose:
                print(f"Evaluating left operand: {node.left}")
            left = self.eval(node.left)
            if self.verbose:
                print(f"Evaluating right operand: {node.right}")
            right = self.eval(node.right)
            if isinstance(left, Int) and isinstance(right, Int):
                # Arithmetic operations
                if node.op == "+":
                    result = Int(left.value + right.value)
                elif node.op == "*":
                    result = Int(left.value * right.value)
                elif node.op == "-":
                    result = Int(left.value - right.value)
                elif node.op == "/":
                    result = Int(left.value // right.value)  # Using integer division
                # Comparison operations
                elif node.op == ">":
                    result = Bool(left.value > right.value)
                elif node.op == "<":
                    result = Bool(left.value < right.value)
                elif node.op == "==":
                    result = Bool(left.value == right.value)
                elif node.op == "<=":
                    result = Bool(left.value <= right.value)
                elif node.op == ">=":
                    result = Bool(left.value >= right.value)
                if self.verbose:
                    print(f"Binary operation result: {result}")
                return result
            # Boolean operations
            elif isinstance(left, Bool) and isinstance(right, Bool):
                if node.op == "&":
                    result = Bool(left.value and right.value)
                elif node.op == "|":
                    result = Bool(left.value or right.value)
                if self.verbose:
                    print(f"Boolean operation result: {result}")
                return result
            return BinOp(node.op, left, right)

        # Unary operation handling
        elif isinstance(node, UnaryOp):
            if self.verbose:
                print(f"Evaluating unary operand: {node.operand}")
            operand = self.eval(node.operand)
            if isinstance(operand, Bool) and node.op == "!":
                result = Bool(not operand.value)
                if self.verbose:
                    print(f"Unary operation result: {result}")
                return result
            return UnaryOp(node.op, operand)

        if self.verbose:
            print(f"Default return: {node}")
        return node

    def substitute(self, expr: ASTNode, var: Var, value: ASTNode) -> ASTNode:
        """
        Substitutes 'value' for 'var' in 'expr'.
        """
        if isinstance(expr, Var):
            return value if expr.name == var.name else expr
        elif isinstance(expr, Apply):
            return Apply(
                self.substitute(expr.func, var, value),
                self.substitute(expr.arg, var, value)
            )
        elif isinstance(expr, Function):
            if expr.arg.name == var.name:
                return expr
            if isinstance(expr.body, Function) and expr.body.arg.name == var.name:
                return expr
            return Function(
                expr.arg,
                self.substitute(expr.body, var, value)
            )
        elif isinstance(expr, BinOp):
            return BinOp(
                expr.op,
                self.substitute(expr.left, var, value),
                self.substitute(expr.right, var, value)
            )
        elif isinstance(expr, Let):
            if expr.name.name == var.name:
                return expr
            new_value = self.substitute(expr.value, var, value)
            new_body = self.substitute(expr.body, var, value)
            return Let(expr.name, new_value, new_body)
        elif isinstance(expr, If):
            return If(
                self.substitute(expr.cond, var, value),
                self.substitute(expr.then_expr, var, value),
                self.substitute(expr.else_expr, var, value)
            )
        return expr

def execute_ast(ast: ASTNode, verbose: bool = False) -> ASTNode:
    """
    Executes an AST using the AST interpreter.
    """
    machine = ASTInterpreter(verbose)
    if verbose:
        print(f"Executing AST: {ast}")
    result = machine.eval(ast)
    if verbose:
        print(f"Final result: {result}")
    return result
