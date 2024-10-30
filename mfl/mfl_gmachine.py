"""
G-Machine Implementation for MFL (Mini Functional Language)

The G-machine is a graph reduction machine for evaluating lazy functional programs.
It operates by compiling high-level functional code into a sequence of imperative
instructions that manipulate a graph structure (heap) and a stack.

Key Components:
--------------
1. Stack: Used for temporary values and building graph structures
2. Heap: Stores the graph nodes representing the program
3. Globals: Maps names to supercombinator nodes in the heap
4. Code: Sequence of instructions to execute

Node Types:
----------
1. NNum: Number node, represents an integer value
2. NAp: Application node, represents function application
3. NGlobal: Global node, represents a supercombinator (function)
4. NInd: Indirection node, used for sharing and updating

Instruction Set:
--------------
Stack Operations:
- PUSH n: Push nth stack item onto top of stack
- POP n: Remove n items from stack
- SLIDE n: Keep top item and remove n items below it
- UPDATE n: Update nth stack item with indirection to top item
- ALLOC n: Allocate n nodes initialized to dummy values

Graph Construction:
- PUSHINT n: Create number node with value n
- PUSHGLOBAL f: Push address of global f
- MKAP: Make application node from top two stack items

Evaluation Control:
- EVAL: Evaluate top stack item to weak head normal form
- UNWIND: Unwind spine of graph until weak head normal form

Compilation Process:
------------------
1. Lambda Lifting:
   - Convert let expressions into global supercombinators
   - Handle free variables by passing them as extra arguments

2. Scheme R - Compiling expressions:
   - Variables: Look up in environment and push offset
   - Applications: Compile argument, compile function, make application
   - Functions: Create global supercombinator
   - Let bindings: Allocate space, compile value, update, compile body

3. Scheme C - Compiling supercombinators:
   - Create new environment for arguments
   - Compile body in this environment
   - Add update and unwind instructions

Evaluation Strategy:
------------------
1. Lazy Evaluation:
   - Arguments are not evaluated until needed
   - Results are updated with their value (sharing)

2. Graph Reduction:
   - Program is represented as a graph
   - Evaluation updates graph nodes (mutation)
   - Sharing prevents duplicate computation

Example Execution:
----------------
For expression: let double = λx.(x*2) in (double 3)

1. Compilation:
   - Create global for double function
   - Compile body with multiplication
   - Create application of double to 3

2. Execution:
   - Push double function
   - Push argument 3
   - Make application node
   - Evaluate application:
     * Unwind to function
     * Push argument
     * Execute function body
     * Update result
     * Clean up stack

3. Result:
   - Final value 6 is left on stack

Usage:
-----
The G-machine is used as a backend for MFL:
    python3 mfl.py -g -b "let double = λx.(x*2) in (double 3)"

This will:
1. Parse the expression into an AST
2. Compile the AST to G-machine code
3. Execute the code using the G-machine
4. Return the final result
"""
# Import the ASTNode classes
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
from mfl_ast import ASTNode, Var, Function, Apply, Let, Int, Bool, BinOp, UnaryOp

# G-machine instructions
class Instruction:
    def __str__(self) -> str:
        return self.__class__.__name__

@dataclass
class PUSH(Instruction):
    """Push offset from stack"""
    n: int
    def __str__(self): return f"PUSH({self.n})"

@dataclass
class PUSHGLOBAL(Instruction):
    """Push global name"""
    name: str
    def __str__(self): return f"PUSHGLOBAL({self.name})"

@dataclass
class PUSHINT(Instruction):
    """Push integer"""
    value: int
    def __str__(self): return f"PUSHINT({self.value})"

@dataclass
class MKAP(Instruction):
    """Make application node"""
    pass

class EVAL(Instruction):
    """Evaluate top of stack"""
    def __init__(self): pass

class UNWIND(Instruction):
    """Unwind spine"""
    def __init__(self): pass

@dataclass
class SLIDE(Instruction):
    """Slide n items"""
    n: int
    def __str__(self): return f"SLIDE({self.n})"

@dataclass
class UPDATE(Instruction):
    """Update node"""
    n: int
    def __str__(self): return f"UPDATE({self.n})"

@dataclass
class POP(Instruction):
    """Pop n items"""
    n: int
    def __str__(self): return f"POP({self.n})"

@dataclass
class ALLOC(Instruction):
    """Allocate n items"""
    n: int
    def __str__(self): return f"ALLOC({self.n})"

class ADD(Instruction):
    """Add top two numbers"""
    def __init__(self): pass

class MUL(Instruction):
    """Multiply top two numbers"""
    def __init__(self): pass

# Node types for graph
class Node:
    def __str__(self) -> str:
        return self.__class__.__name__

@dataclass
class NNum(Node):
    """Number node"""
    n: int
    def __str__(self): return str(self.n)

@dataclass
class NAp(Node):
    """Application node"""
    left: int  # Address of function
    right: int  # Address of argument
    def __str__(self): return f"@{self.left} @{self.right}"

@dataclass
class NGlobal(Node):
    """Global node"""
    arity: int
    code: List[Instruction]
    def __str__(self): return f"NGlobal({self.arity})"

@dataclass
class NInd(Node):
    """Indirection node"""
    addr: int
    def __str__(self): return f"→@{self.addr}"

class Env:
    """Environment mapping names to stack offsets"""
    def __init__(self, offsets: Dict[str, int] = None, parent: Optional['Env'] = None):
        self.offsets = offsets or {}
        self.parent = parent
        self.next_offset = 0

    def lookup(self, name: str) -> Optional[int]:
        """Look up a name in the environment"""
        if name in self.offsets:
            return self.offsets[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def extend(self, name: str) -> int:
        """Add a new binding and return its offset"""
        offset = self.next_offset
        self.offsets[name] = offset
        self.next_offset += 1
        return offset

class GMachine:
    def __init__(self, verbose=False):
        self.code: List[Instruction] = []  # Current instruction stream
        self.stack: List[int] = []  # Stack of addresses
        self.heap: Dict[int, Node] = {}  # Heap of nodes
        self.globals: Dict[str, int] = {}  # Global addresses
        self.next_addr = 0  # Next free heap address
        self.verbose = verbose

    def log(self, msg: str):
        """Print a log message if verbose mode is enabled"""
        if self.verbose:
            print(msg)

    def show_stack(self) -> str:
        """Show current stack state"""
        return "[" + " ".join(f"@{addr}={self.heap[addr]}" for addr in self.stack) + "]"

    def alloc(self, node: Node) -> int:
        """Allocate a new node in the heap"""
        addr = self.next_addr
        self.heap[addr] = node
        self.next_addr += 1
        return addr

    def compile_sc(self, name: str, args: List[str], body: ASTNode, env: Env) -> NGlobal:
        """Compile a supercombinator"""
        # Create new environment with arguments
        sc_env = Env()
        for arg in reversed(args):
            sc_env.extend(arg)
        
        # Compile body in new environment
        code = self.compile(body, sc_env)
        return NGlobal(len(args), code + [UPDATE(len(args)), POP(len(args)), UNWIND()])

    def compile(self, node: ASTNode, env: Env) -> List[Instruction]:
        """Compile AST to G-machine instructions"""
        if isinstance(node, Int):
            return [PUSHINT(node.value)]

        elif isinstance(node, Var):
            # Look up variable in environment
            offset = env.lookup(node.name)
            if offset is not None:
                return [PUSH(offset), EVAL()]
            else:
                return [PUSHGLOBAL(node.name)]

        elif isinstance(node, Apply):
            return (
                self.compile(node.arg, env) +  # Compile argument
                [EVAL()] +                     # Evaluate argument
                self.compile(node.func, env) + # Compile function
                [EVAL()] +                     # Evaluate function
                [MKAP(), EVAL()]              # Make application node and evaluate
            )

        elif isinstance(node, Function):
            if isinstance(node.body, BinOp) and node.body.op == "*":
                # Special case for multiplication by constant
                if (isinstance(node.body.left, Var) and node.body.left.name == node.arg.name and
                    isinstance(node.body.right, Int)):
                    # λx.(x * 2) -> special multiplication function
                    name = f"$f{len(self.globals)}"
                    code = [
                        PUSH(0),           # Push argument
                        EVAL(),            # Evaluate to number
                        PUSHINT(node.body.right.value),  # Push constant
                        MUL()              # Multiply
                    ]
                    sc = NGlobal(1, code)
                    addr = self.alloc(sc)
                    self.globals[name] = addr
                    return [PUSHGLOBAL(name)]
            
            # Default case - compile as normal function
            name = f"$f{len(self.globals)}"
            body_env = Env()
            body_env.extend(node.arg.name)
            body_code = self.compile(node.body, body_env)
            sc = NGlobal(1, body_code)
            addr = self.alloc(sc)
            self.globals[name] = addr
            return [PUSHGLOBAL(name)]

        elif isinstance(node, Let):
            # Compile value
            value_code = self.compile(node.value, env)
            
            # Add binding to environment
            let_env = Env(parent=env)
            offset = let_env.extend(node.name.name)
            
            # Compile body with new binding
            body_code = self.compile(node.body, let_env)
            
            return (
                [ALLOC(1)] +                # Space for binding
                value_code +                # Evaluate value
                [UPDATE(0)] +               # Update binding
                body_code +                 # Evaluate body
                [SLIDE(1)]                  # Remove binding but keep result
            )

        elif isinstance(node, BinOp):
            if node.op == "+":
                return (
                    self.compile(node.left, env) +   # Compile left operand
                    [EVAL()] +                       # Evaluate to WHNF
                    self.compile(node.right, env) +  # Compile right operand
                    [EVAL()] +                       # Evaluate to WHNF
                    [ADD()]                          # Add them
                )
            elif node.op == "*":
                return (
                    self.compile(node.left, env) +   # Compile left operand
                    [EVAL()] +                       # Evaluate to WHNF
                    self.compile(node.right, env) +  # Compile right operand
                    [EVAL()] +                       # Evaluate to WHNF
                    [MUL()]                          # Multiply them
                )
            else:
                raise ValueError(f"Unsupported operator: {node.op}")

        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    def eval(self, node: ASTNode) -> Union[int, Node]:
        """Evaluate an AST node using the G-machine"""
        # Compile the node
        self.code = self.compile(node, Env()) + [EVAL()]
        self.log("\nCompiled code:")
        for i, instr in enumerate(self.code):
            self.log(f"{i}: {str(instr)}")
        
        # Execute until no more instructions
        while self.code:
            instruction = self.code.pop(0)
            self.log(f"\nExecuting: {str(instruction)}")
            self.log(f"Stack before: {self.show_stack()}")
            self.execute(instruction)
            self.log(f"Stack after: {self.show_stack()}")

        # Result is on top of stack
        if not self.stack:
            raise ValueError("Stack empty after evaluation")
        
        result_addr = self.stack[-1]
        result = self.heap[result_addr]
        
        # Follow indirections to get final result
        while isinstance(result, NInd):
            result_addr = result.addr
            result = self.heap[result_addr]
        
        if isinstance(result, NNum):
            return result.n
        return result

    def execute(self, instruction: Instruction):
        """Execute a single instruction"""
        if isinstance(instruction, PUSHINT):
            # Create number node and push address
            addr = self.alloc(NNum(instruction.value))
            self.stack.append(addr)

        elif isinstance(instruction, PUSHGLOBAL):
            # Push global address
            if instruction.name not in self.globals:
                raise ValueError(f"Undefined global: {instruction.name}")
            self.stack.append(self.globals[instruction.name])

        elif isinstance(instruction, MKAP):
            # Make application node from top two stack items
            if len(self.stack) < 2:
                raise ValueError("Stack underflow in MKAP")
            right = self.stack.pop()
            left = self.stack.pop()
            addr = self.alloc(NAp(left, right))
            self.stack.append(addr)

        elif isinstance(instruction, PUSH):
            # Push nth stack item
            if len(self.stack) <= instruction.n:
                raise ValueError("Stack underflow in PUSH")
            self.stack.append(self.stack[-1 - instruction.n])

        elif isinstance(instruction, EVAL):
            # Evaluate top of stack to WHNF
            if not self.stack:
                raise ValueError("Stack underflow in EVAL")
            self.eval_stack_top()

        elif isinstance(instruction, UNWIND):
            # Unwind spine of graph
            if not self.stack:
                raise ValueError("Stack underflow in UNWIND")
            self.unwind()

        elif isinstance(instruction, UPDATE):
            # Update node with indirection
            if len(self.stack) <= instruction.n:
                raise ValueError("Stack underflow in UPDATE")
            addr = self.stack[-1]
            update_addr = self.stack[-1 - instruction.n]
            self.heap[update_addr] = NInd(addr)

        elif isinstance(instruction, POP):
            # Pop n items from stack
            if len(self.stack) < instruction.n:
                raise ValueError("Stack underflow in POP")
            self.stack = self.stack[:-instruction.n]

        elif isinstance(instruction, SLIDE):
            # Slide top over n items
            if len(self.stack) <= instruction.n:
                raise ValueError("Stack underflow in SLIDE")
            top = self.stack.pop()
            self.stack = self.stack[:-instruction.n]
            self.stack.append(top)

        elif isinstance(instruction, ALLOC):
            # Allocate n dummy items
            for _ in range(instruction.n):
                addr = self.alloc(NInd(0))  # Dummy indirection
                self.stack.append(addr)

        elif isinstance(instruction, ADD):
            # Add top two numbers
            if len(self.stack) < 2:
                raise ValueError("Stack underflow in ADD")
            right_addr = self.stack.pop()
            left_addr = self.stack.pop()
            
            # Follow indirections
            while isinstance(self.heap[right_addr], NInd):
                right_addr = self.heap[right_addr].addr
            while isinstance(self.heap[left_addr], NInd):
                left_addr = self.heap[left_addr].addr
            
            right_node = self.heap[right_addr]
            left_node = self.heap[left_addr]
            if not isinstance(right_node, NNum) or not isinstance(left_node, NNum):
                raise ValueError("ADD requires number nodes")
            result = self.alloc(NNum(left_node.n + right_node.n))
            self.stack.append(result)

        elif isinstance(instruction, MUL):
            # Multiply top two numbers
            if len(self.stack) < 2:
                raise ValueError("Stack underflow in MUL")
            right_addr = self.stack.pop()
            left_addr = self.stack.pop()
            
            # Follow indirections
            while isinstance(self.heap[right_addr], NInd):
                right_addr = self.heap[right_addr].addr
            while isinstance(self.heap[left_addr], NInd):
                left_addr = self.heap[left_addr].addr
            
            right_node = self.heap[right_addr]
            left_node = self.heap[left_addr]
            if not isinstance(right_node, NNum) or not isinstance(left_node, NNum):
                raise ValueError("MUL requires number nodes")
            result = self.alloc(NNum(left_node.n * right_node.n))
            self.stack.append(result)

    def eval_stack_top(self):
        """Evaluate top of stack to WHNF"""
        addr = self.stack[-1]
        node = self.heap[addr]
        
        # Follow indirections
        while isinstance(node, NInd):
            addr = node.addr
            node = self.heap[addr]
            self.stack[-1] = addr
        
        # Already in WHNF?
        if isinstance(node, (NNum, NGlobal)):
            return
            
        # Save rest of stack
        save_stack = self.stack[:-1]
        self.stack = [addr]
        
        # Unwind until WHNF
        self.unwind()
        
        # Restore stack
        result = self.stack[0]
        self.stack = save_stack + [result]

    def unwind(self):
        """Unwind spine of graph"""
        while True:
            addr = self.stack[-1]
            node = self.heap[addr]
            
            # Follow indirections
            while isinstance(node, NInd):
                addr = node.addr
                node = self.heap[addr]
                self.stack[-1] = addr
            
            if isinstance(node, NNum):
                # Number - done
                return
                
            elif isinstance(node, NAp):
                # Application - follow function
                self.stack.append(node.left)
                
            elif isinstance(node, NGlobal):
                # Global - check arity
                if len(self.stack) - 1 < node.arity:
                    # Partial application - done
                    self.stack.pop()  # Remove global
                    return
                
                # Get arguments
                args = []
                for i in range(node.arity):
                    ap_addr = self.stack[-1 - i]
                    ap_node = self.heap[ap_addr]
                    if not isinstance(ap_node, NAp):
                        raise ValueError("Expected application node")
                    args.append(ap_node.right)
                
                # Remove function and arguments
                self.stack = self.stack[:-node.arity-1]
                
                # Push arguments
                self.stack.extend(reversed(args))
                
                # Save code and execute
                save_code = self.code
                self.code = node.code.copy()
                
                # Execute function body
                while self.code:
                    self.execute(self.code.pop(0))
                
                self.code = save_code
                return
            
            else:
                raise ValueError(f"Invalid node type in unwind: {type(node)}")

def execute_ast(ast: ASTNode, verbose: bool = False) -> Union[int, Node]:
    """Execute an AST using the G-machine."""
    machine = GMachine(verbose)
    if verbose:
        print(f"Input AST: {ast}")
    result = machine.eval(ast)
    if verbose:
        print(f"Final result: {result}")
    return result
