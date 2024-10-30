# Import the ASTNode classes
from typing import Union
from mfl_type_checker import ASTNode, Var, Function, Apply, Let, Int, Bool, BinOp, UnaryOp

class Combinator(ASTNode):
    """Base class for combinators."""
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name

class S(Combinator):
    def __init__(self):
        super().__init__("S")

class K(Combinator):
    def __init__(self):
        super().__init__("K")

class I(Combinator):
    def __init__(self):
        super().__init__("I")

class B(Combinator):
    """B combinator: B f g x → f (g x)"""
    def __init__(self):
        super().__init__("B")

class C(Combinator):
    """C combinator: C f g x → f x g"""
    def __init__(self):
        super().__init__("C")

class SPrime(Combinator):
    """S' combinator: S' c f g x → c (f x) (g x)"""
    def __init__(self):
        super().__init__("S'")

class BStar(Combinator):
    """B* combinator: B* c f g x → c (f (g x))"""
    def __init__(self):
        super().__init__("B*")

class CPrime(Combinator):
    """C' combinator: C' c f g x → c (f x) g"""
    def __init__(self):
        super().__init__("C'")

class Plus(Combinator):
    """Primitive addition combinator."""
    def __init__(self):
        super().__init__("+")

class Minus(Combinator):
    """Primitive subtraction combinator."""
    def __init__(self):
        super().__init__("-")

class Multiply(Combinator):
    """Primitive multiplication combinator."""
    def __init__(self):
        super().__init__("*")

class Divide(Combinator):
    """Primitive division combinator."""
    def __init__(self):
        super().__init__("/")

class GreaterThan(Combinator):
    """Greater than comparison combinator."""
    def __init__(self):
        super().__init__(">")

class LessThan(Combinator):
    """Less than comparison combinator."""
    def __init__(self):
        super().__init__("<")

class GreaterEqual(Combinator):
    """Greater than or equal comparison combinator."""
    def __init__(self):
        super().__init__(">=")

class LessEqual(Combinator):
    """Less than or equal comparison combinator."""
    def __init__(self):
        super().__init__("<=")

class Equal(Combinator):
    """Equality comparison combinator."""
    def __init__(self):
        super().__init__("==")

class And(Combinator):
    """Logical AND combinator."""
    def __init__(self):
        super().__init__("&&")

class Or(Combinator):
    """Logical OR combinator."""
    def __init__(self):
        super().__init__("||")

class Not(Combinator):
    """Logical NOT combinator."""
    def __init__(self):
        super().__init__("!")

class SKIMachine:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.s = S()
        self.k = K()
        self.i = I()
        self.b = B()
        self.c = C()
        self.s_prime = SPrime()
        self.b_star = BStar()
        self.c_prime = CPrime()
        self.plus = Plus()
        self.minus = Minus()
        self.multiply = Multiply()
        self.divide = Divide()
        self.gt = GreaterThan()
        self.lt = LessThan()
        self.ge = GreaterEqual()
        self.le = LessEqual()
        self.eq = Equal()
        self.and_ = And()
        self.or_ = Or()
        self.not_ = Not()

    def eval(self, node: ASTNode) -> ASTNode:
        """Evaluates the AST node by first converting to SKI combinators and then reducing."""
        if isinstance(node, (Int, Bool)):
            return node

        print("\nTranslating to SKI combinators...")

        # Convert to SKI combinators
        ski_term = self.to_ski(node)
        print(f"SKI term: {ski_term}")

        if self.verbose:
            print("\nReducing SKI term...")
        return self.reduce(ski_term)

    def to_ski(self, node: ASTNode) -> ASTNode:
        """Converts lambda expressions to SKI combinators."""
        if isinstance(node, (Int, Bool)):
            return node

        if isinstance(node, Var):
            return node

        if isinstance(node, Function):
            return self.abstract_var(node.arg.name, self.to_ski(node.body))

        if isinstance(node, Apply):
            return Apply(self.to_ski(node.func), self.to_ski(node.arg))

        if isinstance(node, Let):
            # Convert let to lambda application
            return self.to_ski(Apply(Function(node.name, node.body), node.value))

        if isinstance(node, BinOp):
            left = self.to_ski(node.left)
            right = self.to_ski(node.right)
            
            # Map operators to combinators
            op_map = {
                "+": self.plus,
                "-": self.minus,
                "*": self.multiply,
                "/": self.divide,
                ">": self.gt,
                "<": self.lt,
                ">=": self.ge,
                "<=": self.le,
                "==": self.eq,
                "&&": self.and_,
                "||": self.or_
            }
            
            if node.op in op_map:
                return Apply(Apply(op_map[node.op], left), right)

        if isinstance(node, UnaryOp) and node.op == "!":
            return Apply(self.not_, self.to_ski(node.expr))

        return node

    def abstract_var(self, var: str, body: ASTNode) -> ASTNode:
        """Performs bracket abstraction for a single variable."""
        if not self.occurs_free(var, body):
            return Apply(self.k, body)

        if isinstance(body, Var):
            if body.name == var:
                return self.i
            return Apply(self.k, body)

        if isinstance(body, Apply):
            return Apply(
                Apply(self.s, self.abstract_var(var, body.func)),
                self.abstract_var(var, body.arg)
            )

        return body

    def reduce(self, node: ASTNode) -> ASTNode:
        """Reduces SKI expressions to normal form."""
        steps = 0
        while True:
            if self.verbose:
                print(f"Step {steps}: {node}")
            new_node = self.reduce_step(node)
            if new_node == node:
                break
            node = new_node
            steps += 1
            if steps > 100:  # Prevent infinite loops
                if self.verbose:
                    print("Maximum reduction steps reached")
                break
        return node

    def reduce_step(self, node: ASTNode) -> ASTNode:
        """Performs one step of reduction."""
        if not isinstance(node, Apply):
            return node

        # First reduce the function and argument
        func = self.reduce_step(node.func)
        arg = self.reduce_step(node.arg)

        # Basic SKI reduction rules
        if isinstance(func, I):
            return arg

        if isinstance(func, Apply) and isinstance(func.func, K):
            return func.arg

        if isinstance(func, Apply) and isinstance(func.func, Apply) and isinstance(func.func.func, S):
            f = func.func.arg
            g = func.arg
            x = arg
            return Apply(Apply(f, x), Apply(g, x))

        # B combinator reduction: B f g x → f (g x)
        if isinstance(func, Apply) and isinstance(func.func, Apply) and isinstance(func.func.func, B):
            f = func.func.arg
            g = func.arg
            x = arg
            return Apply(f, Apply(g, x))

        # C combinator reduction: C f g x → f x g
        if isinstance(func, Apply) and isinstance(func.func, Apply) and isinstance(func.func.func, C):
            f = func.func.arg
            g = func.arg
            x = arg
            return Apply(Apply(f, x), g)

        # S' combinator reduction: S' c f g x → c (f x) (g x)
        if isinstance(func, Apply) and isinstance(func.func, Apply) and isinstance(func.func.func, Apply) and isinstance(func.func.func.func, SPrime):
            c = func.func.func.arg
            f = func.func.arg
            g = func.arg
            x = arg
            return Apply(Apply(c, Apply(f, x)), Apply(g, x))

        # B* combinator reduction: B* c f g x → c (f (g x))
        if isinstance(func, Apply) and isinstance(func.func, Apply) and isinstance(func.func.func, Apply) and isinstance(func.func.func.func, BStar):
            c = func.func.func.arg
            f = func.func.arg
            g = func.arg
            x = arg
            return Apply(c, Apply(f, Apply(g, x)))

        # C' combinator reduction: C' c f g x → c (f x) g
        if isinstance(func, Apply) and isinstance(func.func, Apply) and isinstance(func.func.func, Apply) and isinstance(func.func.func.func, CPrime):
            c = func.func.func.arg
            f = func.func.arg
            g = func.arg
            x = arg
            return Apply(Apply(c, Apply(f, x)), g)

        # Handle arithmetic operations
        if isinstance(func, Apply):
            if isinstance(func.arg, Int) and isinstance(arg, Int):
                if isinstance(func.func, Plus):
                    return Int(func.arg.value + arg.value)
                elif isinstance(func.func, Minus):
                    return Int(func.arg.value - arg.value)
                elif isinstance(func.func, Multiply):
                    return Int(func.arg.value * arg.value)
                elif isinstance(func.func, Divide) and arg.value != 0:
                    return Int(func.arg.value // arg.value)

            # Handle comparison operations
            if isinstance(func.arg, Int) and isinstance(arg, Int):
                if isinstance(func.func, GreaterThan):
                    return Bool(func.arg.value > arg.value)
                elif isinstance(func.func, LessThan):
                    return Bool(func.arg.value < arg.value)
                elif isinstance(func.func, GreaterEqual):
                    return Bool(func.arg.value >= arg.value)
                elif isinstance(func.func, LessEqual):
                    return Bool(func.arg.value <= arg.value)
                elif isinstance(func.func, Equal):
                    return Bool(func.arg.value == arg.value)

            # Handle boolean operations
            if isinstance(func.arg, Bool) and isinstance(arg, Bool):
                if isinstance(func.func, And):
                    return Bool(func.arg.value and arg.value)
                elif isinstance(func.func, Or):
                    return Bool(func.arg.value or arg.value)

        # Handle unary operations
        if isinstance(func, Not) and isinstance(arg, Bool):
            return Bool(not arg.value)

        # If no reduction rule applies, reconstruct with reduced parts
        if func != node.func or arg != node.arg:
            return Apply(func, arg)

        return node

    def occurs_free(self, var_name: str, node: ASTNode) -> bool:
        """Checks if variable occurs free in expression."""
        if isinstance(node, Var):
            return node.name == var_name
        if isinstance(node, Function):
            return node.arg.name != var_name and self.occurs_free(var_name, node.body)
        if isinstance(node, Apply):
            return self.occurs_free(var_name, node.func) or self.occurs_free(var_name, node.arg)
        if isinstance(node, BinOp):
            return self.occurs_free(var_name, node.left) or self.occurs_free(var_name, node.right)
        return False

def execute_ast(ast: ASTNode, verbose: bool = False) -> ASTNode:
    """Execute an AST using the SKI combinator machine."""
    machine = SKIMachine(verbose)
    if verbose:
        print(f"Input AST: {ast}")
    result = machine.eval(ast)
    if verbose:
        print(f"Final result: {result}")
    return result