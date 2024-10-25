"""
SECD Machine Implementation

The SECD machine is a virtual machine designed to evaluate lambda calculus expressions.
It consists of four main registers:

S (Stack): Holds intermediate results during computation
E (Environment): Stores variable bindings
C (Control): Contains the sequence of instructions to be executed
D (Dump): Used to save/restore machine state during function calls

The machine executes instructions one by one, modifying the S, E, C, and D components
as needed. A function call pushes the current state onto the D, sets up a new
environment E, and continues execution with the function's code in C.
A function return, pops a state from D, restoring the previous environment and
continuing execution. The process continues until C is empty. The final result
is usually found on the top of the stack S.

Instructions:
- NIL: Push empty list onto stack
- LDC: Load constant onto stack
- LD: Load variable value from environment
- CONS: Create pair from top two stack elements
- CAR: Get first element of pair
- CDR: Get second element of pair
- ATOM: Check if value is atomic
- ADD/SUB/MUL/DIV: Arithmetic operations
- EQ/LE/LT: Comparison operations
- SEL: Conditional branch
- JOIN: Return from conditional branch
- LDF: Create closure (lambda)
- AP: Apply function
- RET: Return from function call
- DUM: Create dummy environment for recursion
- RAP: Recursive apply
"""

from mfl_type_checker import (
    Int, Bool, Var, Function, Apply, Let, BinOp, UnaryOp
)

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import operator

# Type aliases for clarity
Value = Any  # Values that can be manipulated by the machine
Env = List[List[Value]]  # Environment: list of frames, each frame is a list of values
Control = List[Any]  # Control: list of instructions to execute
Dump = List[Tuple[List[Value], Env, Control]]  # Dump: saved machine states

@dataclass
class Closure:
    """
    Represents a function closure in the SECD machine.

    Attributes:
        body: The control sequence (instructions) of the function
        env: The environment captured when the closure was created
    """
    body: Control
    env: Env

class SECDMachine:
    """
    Implementation of the SECD (Stack, Environment, Control, Dump) machine.

    The machine executes instructions by maintaining and updating these four registers:
    - stack: Holds computation results and intermediate values
    - env: Current environment containing variable bindings
    - control: Current sequence of instructions being executed
    - dump: Stack of saved machine states for returning from function calls
    """

    def __init__(self, verbose: bool = False):
        """Initialize SECD machine with empty registers."""
        self.verbose = verbose
        self.stack: List[Value] = []
        self.env: Env = []
        self.control: Control = []
        self.dump: Dump = []

    def debug_print(self, message: str) -> None:
        """Print debug message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def run(self, control: Control) -> Optional[Value]:
        """
        Execute a sequence of SECD machine instructions.

        Args:
            control: List of instructions to execute

        Returns:
            The final value on the stack after execution, or None if stack is empty
        """
        self.control = control

        while self.control:
            instruction = self.control.pop(0)
            self.execute(instruction)

        return self.stack[0] if self.stack else None

    def execute(self, instruction: Union[str, Tuple[str, Any]]) -> None:
        """
        Execute a single SECD machine instruction.

        Args:
            instruction: Either a string representing the instruction name,
                       or a tuple of (instruction_name, argument)
        """
        if isinstance(instruction, tuple):
            op, arg = instruction
        else:
            op = instruction
            arg = None

        self.debug_print(f"\nExecuting: {op} {arg if arg else ''}")
        self.debug_print(f"Stack before: {self.stack}")
        self.debug_print(f"Env before: {self.env}")
        self.debug_print(f"Control before: {self.control}")
        self.debug_print(f"Dump before: {self.dump}")

        if op == "NIL":
            # Push empty list onto stack
            self.stack.append([])

        elif op == "LDC":
            # Load constant: Push constant value onto stack
            self.stack.append(arg)

        elif op == "LD":
            # Load variable: Push value from environment onto stack
            i, j = arg  # Environment coordinates (frame, position)
            self.stack.append(self.env[i][j])

        elif op == "CONS":
            # Create pair: Pop two values and push (value1, value2)
            b = self.stack.pop()
            a = self.stack.pop()
            self.stack.append([a, b])

        elif op == "CAR":
            # Get first element: Pop pair and push first element
            pair = self.stack.pop()
            self.stack.append(pair[0])

        elif op == "CDR":
            # Get second element: Pop pair and push second element
            pair = self.stack.pop()
            self.stack.append(pair[1])

        elif op == "ATOM":
            # Check if atomic: Push True if top of stack is atomic (not a pair)
            value = self.stack.pop()
            self.stack.append(not isinstance(value, list))

        elif op in ["ADD", "SUB", "MUL", "DIV", "EQ", "LE", "LT"]:
            # Arithmetic and comparison operations
            b = self.stack.pop()
            a = self.stack.pop()
            # If we get a list, extract the actual value
            if isinstance(a, list) and len(a) > 0:
                a = a[0]
            if isinstance(b, list) and len(b) > 0:
                b = b[0]
            ops = {
                "ADD": operator.add,
                "SUB": operator.sub,
                "MUL": operator.mul,
                "DIV": operator.truediv,
                "EQ": operator.eq,
                "LE": operator.le,
                "LT": operator.lt
            }
            self.stack.append(ops[op](a, b))

        elif op == "SEL":
            # Conditional branch: Pop condition and select then/else branch
            then_branch, else_branch = arg
            condition = self.stack.pop()
            self.dump.append((self.stack.copy(), self.env, self.control))
            self.control = then_branch if condition else else_branch

        elif op == "JOIN":
            # Return from conditional: Restore state from dump
            self.stack, self.env, self.control = self.dump.pop()

        elif op == "LDF":
            # Create closure: Push new closure with current environment
            self.stack.append(Closure(arg, self.env))

        elif op == "AP":
            # Apply function: Call closure with arguments
            closure = self.stack.pop()
            args = self.stack.pop()

            # Save current state
            self.dump.append((self.stack.copy(), self.env, self.control))

            # Set up new state for function execution
            # Extract argument value from the args list structure
            arg_value = args[1] if isinstance(args, list) and len(args) > 1 else args
            new_frame = [arg_value]  # Create new environment frame with the argument
            self.stack = []
            self.env = [new_frame] + closure.env
            self.control = closure.body

        elif op == "RET":
            # Return from function: Restore state and push result
            result = self.stack.pop()
            self.stack, self.env, self.control = self.dump.pop()
            self.stack.append(result)

        elif op == "DUM":
            # Create dummy environment for recursive functions
            self.env = [[]] + self.env

        elif op == "RAP":
            # Recursive apply: Similar to AP but updates recursive environment
            closure = self.stack.pop()
            args = self.stack.pop()

            self.dump.append((self.stack.copy(), self.env[1:], self.control))
            self.stack = []
            self.env[0] = args
            self.control = closure.body

        elif op == "LET":
            # Create new environment frame with binding
            value = self.stack.pop()
            bind_idx = arg
            new_frame = [None] * (bind_idx + 1)
            new_frame[bind_idx] = value
            self.env = [new_frame] + self.env

def compile_ast(ast, env_map=None, level=0, verbose=False):
    """
    Compile AST to SECD machine instructions.

    Args:
        ast: AST node from the parser
        env_map: Dictionary mapping variable names to (level, index) pairs
        level: Current nesting level for environment indexing

    Returns:
        List of SECD machine instructions
    """
    if env_map is None:
        env_map = {}

    if verbose:
        print(f"\nCompiling node: {type(ast).__name__}")
        print(f"Current env_map: {env_map}")
        print(f"Current level: {level}")

    if isinstance(ast, Int):
        # Load constant onto stack
        return [("LDC", ast.value)]

    elif isinstance(ast, Bool):
        # Load constant onto stack
        return [("LDC", ast.value)]

    elif isinstance(ast, Var):
        if ast.name not in env_map:
            raise ValueError(f"Unbound variable: {ast.name}")
        # Load variable value from environment
        return [("LD", env_map[ast.name])]

    elif isinstance(ast, Function):
        # Create new environment for function body
        new_env_map = env_map.copy()
        param_idx = 0

        if verbose:
            print(f"Function AST: {ast}")
            print(f"Function AST dir: {dir(ast)}")
            print(f"Function AST vars: {vars(ast)}")
        new_env_map[ast.arg.name] = (0, param_idx)

        # Shift existing variables one level up
        for var in env_map:
            lvl, idx = env_map[var]
            new_env_map[var] = (lvl + 1, idx)

        # Compile function body with new environment
        body_code = compile_ast(ast.body, new_env_map, level + 1)
        # Create closure with function body and add Return instruction
        return [("LDF", body_code + ["RET"])]

    elif isinstance(ast, Apply):
        # Compile function and argument
        func_code = compile_ast(ast.func, env_map, level)
        arg_code = compile_ast(ast.arg, env_map, level)
        # 1. Load argument onto stack
        # 2. Load function onto stack
        # 3. Apply function to argument
        return [
            "NIL",
            *arg_code,
            "CONS",
            *func_code,
            "AP"
        ]

    elif isinstance(ast, Let):
        # The Let AST node structure is: Let(name, value, body) where name is a Var node
        var_name = ast.name.name  # First get the Var node, then its name attribute

        # Compile the value to be bound
        value_code = compile_ast(ast.value, env_map, level)

        # Create new environment for let body
        new_env_map = env_map.copy()
        bind_idx = len(env_map)
        new_env_map[var_name] = (0, bind_idx)

        # Compile body with new binding
        body_code = compile_ast(ast.body, new_env_map, level)

        # 1. Compile the value to be bound
        # 2. Create new environment frame with binding
        # 3. Compile the body of the let expression with the new binding
        return [
            *value_code,
            ("LET", bind_idx),  # Create new environment frame with binding
            *body_code
        ]

    elif isinstance(ast, BinOp):
        # Compile operands
        left_code = compile_ast(ast.left, env_map, level)
        right_code = compile_ast(ast.right, env_map, level)

        # Map operators to SECD instructions
        op_map = {
            "+": "ADD",
            "-": "SUB", 
            "*": "MUL",
            "/": "DIV",
            "&": "AND",
            "|": "OR"
        }

        return [
            *left_code,
            *right_code,
            op_map[ast.op]
        ]

    elif isinstance(ast, UnaryOp):
        if ast.op == "!":
            expr_code = compile_ast(ast.expr)
            # Compile expression and apply NOT operation
            return [
                *expr_code,
                "NOT"
            ]

    raise ValueError(f"Unknown AST node type: {type(ast)}")

def execute_ast(ast, verbose=False):
    """
    Execute an AST using the SECD machine.

    Args:
        ast: AST node from the parser

    Returns:
        Final value from the stack
    """
    # Compile AST to instructions
    instructions = compile_ast(ast, {}, 0, verbose)
    print(f"SECD instructions: {instructions}")

    # Create and run SECD machine
    machine = SECDMachine(verbose)
    return machine.run(instructions)

if __name__ == "__main__":
    # Example usage
    from mfl_type_checker import Int, BinOp

    # Create AST for: 5 + 3
    ast = BinOp("+", Int(5), Int(3))

    result = execute_ast(ast)
    print(f"Result: {result}")  # Output: 8
