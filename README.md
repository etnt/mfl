# MFL: Mini Functional Language

This repository contains the implementation of a simple functional programming language called MFL (Mini Functional Language).
MFL is just a learning project, playing around with typical concepts of functional programming languages.

## Features

- **Lambda abstractions:** Define anonymous functions using `λ` (lambda)
- **Function application:** Apply functions to arguments using parentheses
- **Let bindings:** Introduce local variables using `let`
- **Arithmetic operations:** `+`, `-`, `*`, `/`
- **Boolean operations:** `&` (and), `|` (or), `!` (not)
- **Type inference:** The language uses a Hindley-Milner type inference system to automatically deduce the types of expressions.
- **Core Erlang generation:** MFL expressions can be compiled into Core Erlang, a functional intermediate language used in the Erlang compiler pipeline, and then further compiled into BEAM bytecode using the Erlang compiler.
- **SECD machine execution:** MFL expressions can also be executed directly using a SECD machine, a virtual machine designed for evaluating lambda calculus expressions.
- **LLVM IR generation:** MFL expressions can be compiled into LLVM IR, allowing for compilation to native machine code.

## Programs

### 1. MFL - Type Inference System (mfl_type_checker.py)
An educational implementation of the Hindley-Milner type inference system, demonstrating how programming language type systems work. Includes polymorphic type inference, unification, and type checking with detailed documentation explaining core concepts.

### 2. MFL - Parser (mfl_parser.py)
A shift-reduce parser for a simple functional programming language that supports lambda abstractions, function applications, let bindings, and arithmetic expressions. Integrates with the type inference system to provide static typing for parsed expressions.

### 3. MFL - SECD Machine (mfl_secd.py)
An implementation of the SECD (Stack, Environment, Control, Dump) virtual machine for evaluating lambda calculus expressions.
The SECD machine serves as one of the execution backends for the MFL language, allowing direct interpretation of MFL expressions without compilation via Erlang Core.

### 4. MFL - Core Erlang Generator (mfl_core_erlang_generator.py)
A code generator that translates parsed and type-checked expressions into Erlang Core language code. Supports lambda abstractions, function applications, let bindings, and arithmetic expressions.

### 5. MFL - LLVM IR Generator (mfl_llvm.py)
A code generator that translates parsed and type-checked expressions into LLVM IR.  This allows for compilation to native machine code using the LLVM compiler toolchain (clang).

### 6. MFL - AST Interpreter (mfl_ast.py)
A direct interpreter for the Abstract Syntax Tree (AST) that provides another execution backend for MFL expressions. The interpreter evaluates expressions by traversing the AST, handling:
- Variable bindings and lookups
- Function application with proper variable substitution
- Let expressions for local variable definitions
- Arithmetic operations
- Curried functions
This interpreter serves as a reference implementation, making it easier to understand how MFL expressions are evaluated without the complexity of compilation or virtual machine execution.

## Requirements

The programs may require various Python packages. Install them using:

```bash
make
```

Then setup the virtual environment:

```bash
source ./venv/bin/activate
```

## Usage

The `mfl.py` can be run with an argument or without, where in the latter case it will run the default test cases for the parser and type checker.
It also takes a `-v`/`--verbose` flag to print the parsing / code generation steps. By default the code will be compiled down to BEAM code,
via Erlang Core; with the `-s`/`--secd` flag it will instead be executed in the SECD machine; and with the `--llvm` flag it will generate LLVM IR.

```bash
❯ python3 mfl.py --help
usage: mfl.py [-h] [-v] [-b] [-o OUTPUT] [-s] [-k] [-a] [-g] [expression]

Parse and type-check functional programming expressions.

positional arguments:
  expression            Expression to parse and type-check

options:
  -h, --help            show this help message and exit
  -v, --verbose         Enable verbose output from all modules
  -b, --backend-verbose
                        Enable verbose output from backend
  -o, --output OUTPUT   Output file name
  -s, --secd            Execute using SECD machine
  -k, --ski             Execute using SKI combinator machine
  -a, --ast             Execute using AST interpreter
  -g, --gmachine        Execute using G-machine
  -l, --llvm            Generate LLVM IR and compile to binary code 
```

Note: The SKI and G-machine backends are not working atm.

## Examples
Example, generating BEAM code:

```bash
❯ ./venv/bin/python3 mfl.py "let double = λx.(x*2) in (double 21)"
Successfully parsed!
AST: let double = λx.(x * 2) in (double 21)
AST(raw): Let(Var("double"), Function(Var("x"), BinOp("*", Var("x"), Int(2))), Apply(Var("double"), Int(21)))
Inferred type: int
Output written to: mfl.core ,compiling to BEAM as: erlc +from_core mfl.core
Compilation successful!

❯ erl
1> mfl:main().
42
```

Another example, this time running our code in the SECD machine:

```bash
❯ ./venv/bin/python3 mfl.py -s "let add = λx.λy.(x+y) in (add 3 4)"
Successfully parsed!
AST: let add = λx.λy.(x+y) in ((add 3) 4)
AST(raw): Let(Var("add"), Function(Var("x"), Function(Var("y"), BinOp("+", Var("x"), Var("y")))), Apply(Apply(Var("add"), Int(3)), Int(4)))
Inferred type: int
SECD instructions: [('LDF', [('LDF', [('LD', (1, 0)), ('LD', (0, 0)), 'ADD', 'RET']), 'RET']), ('LET', 0), 'NIL', ('LDC', 4), 'CONS', 'NIL', ('LDC', 3), 'CONS', ('LD', (0, 0)), 'AP', 'AP']
SECD machine result: 7
```

Testing the type checker:

```bash
❯ ./venv/bin/python3 mfl.py -s "let add = λx.λy.(x+y) in (add 3 True)"
Successfully parsed!
AST: let add = λx.λy.(x + y) in ((add 3) True)
AST(raw): Let(Var("add"), Function(Var("x"), Function(Var("y"), BinOp("+", Var("x"), Var("y")))), Apply(Apply(Var("add"), Int(3)), Bool(True))) 
Error during type checking: Type mismatch: int and bool
 ```

Test Currying:

```bash
python3 mfl.py -s  "let inc = let add1 = λx.λy.(x+y) in (add1 1) in (inc 4)"
Successfully parsed!
AST: let inc = let add1 = λx.λy.(x + y) in (add1 1) in (inc 4)
AST(raw): Let(Var("inc"), Let(Var("add1"), Function(Var("x"), Function(Var("y"), BinOp("+", Var("x"), Var("y")))), Apply(Var("add1"), Int(1))), Apply(Var("inc"), Int(4)))
Inferred type: int
SECD instructions: [('LDF', [('LDF', [('LD', (1, 0)), ('LD', (0, 0)), 'ADD', 'RET']), 'RET']), ('LET', 0), 'NIL', ('LDC', 1), 'CONS', ('LD', (0, 0)), 'AP', ('LET', 0), 'NIL', ('LDC', 4), 'CONS', ('LD', (0, 0)), 'AP']
SECD machine result: 5
```

Function composition, using the SKI machine:
```bash
python3 mfl.py -k  "let compose = λf.λg.λx.(f (g x)) in let add1 = λx.(x+1) in let double = λx.(x+x) in ((compose double add1) 2)"
Successfully parsed!
AST: let compose = λf.λg.λx.(f (g x)) in let add1 = λx.(x + 1) in let double = λx.(x + x) in (((compose double) add1) 2)
AST(raw): Let(Var("compose"), Function(Var("f"), Function(Var("g"), Function(Var("x"), Apply(Var("f"), Apply(Var("g"), Var("x")))))), Let(Var("add1"), Function(Var("x"), BinOp("+", Var("x"), Int(1))), Let(Var("double"), Function(Var("x"), BinOp("+", Var("x"), Var("x"))), Apply(Apply(Apply(Var("compose"), Var("double")), Var("add1")), Int(2)))))
Inferred type: int
Translating to SKI combinators...
SKI term: (((S ((S ((S (K S)) ((S ((S (K S)) ((S (K (S (K S)))) ((S ((S (K S)) ((S (K K)) ((S (K S)) ((S ((S (K S)) ((S (K K)) I))) (K I)))))) (K ((S (K K)) I)))))) (K (K (K 2)))))) (K (K ((S ((S (K +)) I)) I))))) (K ((S ((S (K +)) I)) (K 1)))) ((S ((S (K S)) ((S (K K)) ((S (K S)) ((S (K K)) I))))) (K ((S ((S (K S)) ((S (K K)) I))) (K I)))))
SKI machine result: 6
 ```

Example, using the LLVM backend:

```bash
❯ python3 mfl.py -o double -l "let double = λx.(x*2) in (double 21)"
Successfully parsed! 
AST: let double = λx.(x * 2) in (double 21) 
AST(raw): Let(Var("double"), Function(Var("x"), BinOp("*", Var("x"), Int(2))), Apply(Var("double"), Int(21))) 
Inferred type: int  
LLVM IR written to: mfl.ll 
Compiling as: clang -o double mfl.ll 
Compilation successful!

❯ ./double
42
```