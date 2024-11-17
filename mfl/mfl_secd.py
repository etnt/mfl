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

from mfl_ast import (
    Int, Bool, Var, Function, Apply, Let, BinOp, UnaryOp, If
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
        self.control = control.copy()  # Make a copy to avoid modifying the original

        while self.control:
            instruction = self.control.pop(0)
            self.execute(instruction)

            # After executing all instructions, if we have a nested list result
            # extract the actual value
            if len(self.stack) == 1 and isinstance(self.stack[0], list):
                while isinstance(self.stack[0], list) and len(self.stack[0]) > 1:
                    self.stack[0] = self.stack[0][1]

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

        # Helper function to safely extract values from nested lists
        def extract_value(val):
            if isinstance(val, list):
                if len(val) == 0:
                    return None
                elif len(val) == 1:
                    return extract_value(val[0])
                else:
                    # For pairs, we want the second element as it contains the actual value
                    return extract_value(val[1])
            return val

        if op == "NIL":
            # Push empty list onto stack
            self.stack.append([])

        elif op == "LDC":
            # Load constant: Push constant value onto stack
            self.stack.append(arg)

        elif op == "LD":
            # Load variable: Push value from environment onto stack
            i, j = arg  # Environment coordinates (frame, position)
            value = self.env[i][j]
            # If the value is a nested list, extract the actual value
            if isinstance(value, list):
                value = extract_value(value)
            self.stack.append(value)

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
            b = extract_value(self.stack.pop())
            a = extract_value(self.stack.pop())
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
            condition = extract_value(self.stack.pop())
            self.dump.append((self.stack.copy(), self.env, self.control))
            self.control = then_branch if condition else else_branch

        elif op == "JOIN":
            # Return from conditional: Restore state from dump and push top of stack back on stack
            top = self.stack.pop()
            # Return from conditional: Restore state from dump
            self.stack, self.env, self.control = self.dump.pop()
            # Push top of stack back on stack
            self.stack.append(top)

        elif op == "LDF":
            # Create closure: Push new closure with current environment
            self.stack.append(Closure(arg, self.env))

        elif op == "AP":
            # Apply function: Call closure with arguments
            closure = extract_value(self.stack.pop())
            args = extract_value(self.stack.pop())

            if closure is None:
                raise ValueError("Cannot apply None as a function")

            if not isinstance(closure, Closure):
                raise ValueError(f"Cannot apply non-closure value: {closure}")

            # Save current state
            self.dump.append((self.stack.copy(), self.env, self.control))

            # Set up new state for function execution
            # Extract argument value from the args list structure
            arg_value = extract_value(args)

            # Create new environment frame with the argument
            new_frame = [arg_value]

            self.stack = []
            self.env = [new_frame] + closure.env
            # Make sure we copy the closure body to avoid modifying it
            self.control = closure.body.copy()

        elif op == "RET":
            # Return from function: Restore state and push result
            result = extract_value(self.stack.pop())
            self.stack, self.env, self.control = self.dump.pop()
            self.stack.append(result)

        elif op == "DUM":
            # Create dummy environment for recursive functions
            self.env = [[]] + self.env

        elif op == "RAP":
            # Recursive apply: Similar to AP but updates recursive environment
            closure = extract_value(self.stack.pop())
            args = extract_value(self.stack.pop())

            self.dump.append((self.stack.copy(), self.env[1:], self.control))
            self.stack = []
            self.env[0] = args
            self.control = closure.body

        elif op == "LET":
            # Create new environment frame with binding
            value = extract_value(self.stack.pop())
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
        # 1. Create empty list for argument
        # 2. Load argument onto stack
        # 3. Create pair (empty list, argument)
        # 4. Load function onto stack
        # 5. Apply function to argument pair
        return [
            "NIL",
            *arg_code,
            "CONS",
            *func_code,
            "AP"
        ]

    elif isinstance(ast, Let):
        # Compile the value to be bound
        value_code = compile_ast(ast.value, env_map, level)

        # Create new environment for let body
        new_env_map = env_map.copy()
        bind_idx = len(env_map)
        new_env_map[ast.name.name] = (0, bind_idx)

        # Shift existing variables one level up
        for var in env_map:
            lvl, idx = env_map[var]
            new_env_map[var] = (lvl + 1, idx)

        # Compile body with new binding
        body_code = compile_ast(ast.body, new_env_map, level)

        return [
            *value_code,
            ("LET", bind_idx),
            *body_code
        ]

    elif isinstance(ast, If):
        # Compile condition
        cond_code = compile_ast(ast.cond, env_map, level)

        # Compile then and else branches.  IMPORTANT: Add JOIN at the end of each branch.
        then_code = compile_ast(ast.then_expr, env_map, level) + ["JOIN"]
        else_code = compile_ast(ast.else_expr, env_map, level) + ["JOIN"]

        return [
            *cond_code,
            ("SEL",(then_code, else_code)),
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
            "|": "OR",
            "==": "EQ",
            "<=": "LE",
            "<": "LT",
            ">": "LT",  # x > y is equivalent to y < x, so swap operands
            ">=": "LE"  # x >= y is equivalent to y <= x, so swap operands
        }

        # For > and >=, we need to swap the operands
        if ast.op in [">", ">="]:
            return [
                *right_code,  # Note the swap here
                *left_code,
                op_map[ast.op]
            ]
        return [
            *left_code,
            *right_code,
            op_map[ast.op]
        ]

    elif isinstance(ast, UnaryOp):
        if ast.op == "!":
            expr_code = compile_ast(ast.expr, env_map, level)
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

    # Create AST for: 5 + 3
    #ast = BinOp("+", Int(5), Int(3))

    # let max = λx.λy.if (x > y) then x else y in (max 3 5)
    ast = Let(Var("max"), Function(Var("x"), Function(Var("y"), If(BinOp(">", Var("x"), Var("y")), Var("x"), Var("y")))), Apply(Apply(Var("max"), Int(3)), Int(5)))

    result = execute_ast(ast)
    print(f"Result: {result}") 
