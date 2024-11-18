# MFL: Mini Functional Language

This repository contains the implementation of a simple functional programming language called MFL (Mini Functional Language).
MFL is just a learning project, playing around with typical concepts of functional programming languages.

## Features

- **Lambda abstractions:** Define anonymous functions using `λ` (lambda)
- **Function application:** Apply functions to arguments using parentheses
- **Let bindings:** Introduce local variables using `let`
- **LetRec bindings:** Introduce recursive local variables using `letrec`
- **Arithmetic operations:** `+`, `-`, `*`, `/`
- **Boolean operations:** `&` (and), `|` (or), `!` (not)
- **Type inference:** The language uses a Hindley-Milner type inference system to automatically deduce the types of expressions.
- **Core Erlang generation:** MFL expressions can be compiled into Core Erlang, a functional intermediate language used in the Erlang compiler pipeline, and then further compiled into BEAM bytecode using the Erlang compiler.
- **SECD machine execution:** MFL expressions can also be executed directly using a SECD machine, a virtual machine designed for evaluating lambda calculus expressions.
- **LLVM IR generation:** MFL expressions can be compiled into LLVM IR, allowing for compilation to native machine code.

## Programs

### 1. MFL - Type Inference System (mfl_type_checker.py)
An educational implementation of the Hindley-Milner type inference system, demonstrating how programming language type systems work. Includes polymorphic type inference, unification, and type checking with detailed documentation explaining core concepts. This implementation was made after reading: [Damas-Hindley-Milner inference two ways](https://bernsteinbear.com/blog/type-inference/) and implements the Algorithm J.

### 2. MFL - PLY Parser and Lexer (mfl_ply_parser.py, mfl_ply_lexer.py)
A modern parser implementation using PLY (Python Lex-Yacc) that replaced the older parser. The lexer defines tokens for all language features including lambda abstractions, let bindings, arithmetic operations, and boolean operations. The parser implements the grammar rules and builds an Abstract Syntax Tree (AST) from the token stream. This implementation provides better error handling and a more maintainable structure compared to the previous parser.

### 3. MFL - SECD Machine (mfl_secd.py)
An implementation of the SECD (Stack, Environment, Control, Dump) virtual machine for evaluating lambda calculus expressions.
The SECD machine serves as one of the execution backends for the MFL language, allowing direct interpretation of MFL expressions without compilation via Erlang Core.

### 4. MFL - Core Erlang Generator (mfl_core_erlang_generator.py)
A code generator that translates parsed and type-checked expressions into Erlang Core language code. Supports lambda abstractions, function applications, let bindings, and arithmetic expressions. See also [Core Erlang by Example](https://www.erlang.org/blog/core-erlang-by-example/) and [Core Erlang](https://www2.it.uu.se/research/group/hipe/cerl/doc/core_erlang-1.0.3.pdf).

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

### 7. MFL - SKI Machine (mfl_ski.py)
An implementation of the SKI combinator calculus machine that provides another execution backend for MFL expressions. The machine works by:
- Converting lambda expressions to SKI combinators through bracket abstraction
- Reducing SKI expressions to normal form using combinator reduction rules
- Supporting arithmetic, comparison, and boolean operations
This implementation demonstrates how complex lambda expressions can be reduced to a minimal set of combinators while preserving their computational meaning.

### 8. MFL - Transform (mfl_transform.py)
A simple transformation that converts `letrec` constructs to `let` constructs plus using a `Y-combinator` to achieve recursion. This transformation is useful for languages that do not support `letrec` directly.

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
  -s, --secd           Execute using SECD machine
  -k, --ski            Execute using SKI combinator machine
  -a, --ast            Execute using AST interpreter
  -g, --gmachine        Execute using G-machine
  -l, --llvm           Generate LLVM IR and compile to binary code 
```

Note: The LLVM- and G-machine backends are not working atm.

## Examples
Example, generating BEAM code:

```bash
❯ python3 ./mfl/mfl.py "let double = λx.(x*2) in (double 21)"
Successfully parsed!
AST: let double = λx.(x * 2) in (double 21)
AST(typed): Let<int>(Var<->(int, int)>("double"), Function<->(int, int)>(Var<int>("x"), BinOp<int>("*", Var<int>("x"), Int<int>(2))), Apply<int>(Var<->(int, int)>("double"), Int<int>(21)))
Inferred type: int
Output written to: mfl.core ,compiling to BEAM as: erlc +from_core mfl.core
Compilation successful!

❯ erl
1> mfl:main().
42
```

Another example, this time running our code in the SECD machine:

```bash
❯ python3 ./mfl/mfl.py -s "let add = λx.λy.(x+y) in (add 3 4)"
Successfully parsed!
AST: let add = λx.λy.(x+y) in ((add 3) 4)
AST(typed): Let<int>(Var<->(int, ->(int, int))>("add"), Function<->(int, ->(int, int))>(Var<int>("x"), Function<->(int, int)>(Var<int>("y"), BinOp<int>("+", Var<int>("x"), Var<int>("y")))), Apply<int>(Apply<->(int, int)>(Var<->(int, ->(int, int))>("add"), Int<int>(3)), Int<int>(4)))
Inferred type: int
SECD instructions: [('LDF', [('LDF', [('LD', (1, 0)), ('LD', (0, 0)), 'ADD', 'RET']), 'RET']), ('LET', 0), 'NIL', ('LDC', 4), 'CONS', 'NIL', ('LDC', 3), 'CONS', ('LD', (0, 0)), 'AP', 'AP']
SECD machine result: 7
```

Testing the type checker:

```bash
❯ python3 ./mfl/mfl.py -s "let add = λx.λy.(x+y) in (add 3 True)"
Successfully parsed!
AST: let add = λx.λy.(x + y) in ((add 3) True)
AST(raw): Let(Var("add"), Function(Var("x"), Function(Var("y"), BinOp("+", Var("x"), Var("y")))), Apply(Apply(Var("add"), Int(3)), Bool(True))) 
Error during type checking: Type mismatch: int and bool
 ```

Test Currying:

```bash
python3 ./mfl/mfl.py -s  "let inc = let add1 = λx.λy.(x+y) in (add1 1) in (inc 4)"
Successfully parsed!
AST: let inc = let add1 = λx.λy.(x + y) in (add1 1) in (inc 4)
AST(typed): Let<int>(Var<->(int, int)>("inc"), Let<->(int, int)>(Var<->(int, ->(int, int))>("add1"), Function<->(int, ->(int, int))>(Var<int>("x"), Function<->(int, int)>(Var<int>("y"), BinOp<int>("+", Var<int>("x"), Var<int>("y")))), Apply<->(int, int)>(Var<->(int, ->(int, int))>("add1"), Int<int>(1))), Apply<int>(Var<->(int, int)>("inc"), Int<int>(4)))
Inferred type: int
SECD instructions: [('LDF', [('LDF', [('LD', (1, 0)), ('LD', (0, 0)), 'ADD', 'RET']), 'RET']), ('LET', 0), 'NIL', ('LDC', 1), 'CONS', ('LD', (0, 0)), 'AP', ('LET', 0), 'NIL', ('LDC', 4), 'CONS', ('LD', (0, 0)), 'AP']
SECD machine result: 5
```

Function composition, using the SKI machine:
```bash
python3 ./mfl/mfl.py -k  "let compose = λf.λg.λx.(f (g x)) in let add1 = λx.(x+1) in let double = λx.(x+x) in ((compose double add1) 2)"
Successfully parsed!
AST: let compose = λf.λg.λx.(f (g x)) in let add1 = λx.(x + 1) in let double = λx.(x + x) in (((compose double) add1) 2)
AST(typed): Let<int>(Var<->(->(int, int), ->(->(int, int), ->(int, int)))>("compose"), Function<->(->(int, int), ->(->(int, int), ->(int, int)))>(Var<->(int, int)>("f"), Function<->(->(int, int), ->(int, int))>(Var<->(int, int)>("g"), Function<->(int, int)>(Var<int>("x"), Apply<int>(Var<->(int, int)>("f"), Apply<int>(Var<->(int, int)>("g"), Var<int>("x")))))), Let<int>(Var<->(int, int)>("add1"), Function<->(int, int)>(Var<int>("x"), BinOp<int>("+", Var<int>("x"), Int<int>(1))), Let<int>(Var<->(int, int)>("double"), Function<->(int, int)>(Var<int>("x"), BinOp<int>("+", Var<int>("x"), Var<int>("x"))), Apply<int>(Apply<->(int, int)>(Apply<->(->(int, int), ->(int, int))>(Var<->(->(int, int), ->(->(int, int), ->(int, int)))>("compose"), Var<->(int, int)>("double")), Var<->(int, int)>("add1")), Int<int>(2))))) 
Inferred type: int
Translating to SKI combinators...
SKI term: (((S ((S ((S (K S)) ((S ((S (K S)) ((S (K (S (K S)))) ((S ((S (K S)) ((S (K K)) ((S (K S)) ((S ((S (K S)) ((S (K K)) I))) (K I)))))) (K ((S (K K)) I)))))) (K (K (K 2)))))) (K (K ((S ((S (K +)) I)) I))))) (K ((S ((S (K +)) I)) (K 1)))) ((S ((S (K S)) ((S (K K)) ((S (K S)) ((S (K K)) I))))) (K ((S ((S (K S)) ((S (K K)) I))) (K I)))))
SKI machine result: 6
 ```

Some fun with `if`:

```bash
python3 ./mfl/mfl.py -a "let between = λx.if (x < 0) then False else if (x > 10) then False else True in (between 2)"
Successfully parsed!
AST(pretty): let between = λx.if (x < 0) then False else if (x > 10) then False else True in (between 2)
AST(typed): Let<bool>(Var<->(int, bool)>("between"), Function<->(int, bool)>(Var<int>("x"), If<bool>(BinOp<bool>("<", Var<int>("x"), Int<int>(0)), Bool<bool>(False), If<bool>(BinOp<bool>(">", Var<int>("x"), Int<int>(10)), Bool<bool>(False), Bool<bool>(True)))), Apply<bool>(Var<->(int, bool)>("between"), Int<int>(2)))
Inferred final type: bool
AST interpreter result: True

python3 ./mfl/mfl.py -a "let between = λx.if (x < 0) then False else if (x > 10) then False else True in (between 11)"
 ...
AST interpreter result: False

# Note: unary minus not supported atm...
python3 ./mfl/mfl.py -a "let between = λx.if (x < 0) then False else if (x > 10) then False else True in (between (0 - 1))"
AST interpreter result: False
```

Example, using the LLVM backend:

```bash
python ./mfl/mfl.py -l -o add "let add = λx.λy.(x + y) in (add 6 9)" 
Successfully parsed!
AST(pretty): let add = λx.λy.(x + y) in ((add 6) 9)
AST(typed): Let<int>(Var<->(int, ->(int, int))>("add"), Function<->(int, ->(int, int))>(Var<int>("x"), Function<->(int, int)>(Var<int>("y"), BinOp<int>("+", Var<int>("x"), Var<int>("y")))), Apply<int>(Apply<->(int, int)>(Var<->(int, ->(int, int))>("add"), Int<int>(6)), Int<int>(9)))
Inferred final type: int  
Module verification successful!
Generated LLVM IR code written to: mfl.ll
Compiling as: clang -O3 -o add mfl.ll 
Compilation successful!

❯ ./add
15
```

Example, using recursion with `letrec` and the SKI backend:

```bash
$ python ./mfl/mfl.py -k  "letrec fac = λx.(if (x == 0) then 1 else (x * (fac (x - 1)))) in (fac 5)"
Successfully parsed!
AST(pretty): letrec fac = λx.if (x == 0) then 1 else (x * (fac (x - 1))) in (fac 5)
AST(typed): LetRec<int>(Var<->(int, int)>("fac"), Function<->(int, int)>(Var<int>("x"), If<int>(BinOp<bool>("==", Var<int>("x"), Int<int>(0)), Int<int>(1), BinOp<int>("*", Var<int>("x"), Apply<int>(Var<->(int, int)>("fac"), BinOp<int>("-", Var<int>("x"), Int<int>(1)))))), Apply<int>(Var<->(int, int)>("fac"), Int<int>(5)))
Inferred final type: int
AST(transformed): let Y = λf.(λx.(f (x x)) λx.(f (x x))) in let fac = (Y λfac.λx.if (x == 0) then 1 else (x * (fac (x - 1)))) in (fac 5)

Translating to SKI combinators...
SKI term: (((S (K ((S I) (K 5)))) ((S I) (K ((S (K (S ((S ((S (K if)) ((S ((S (K ==)) I)) (K 0)))) (K 1))))) ((S (K (S ((S (K *)) I)))) ((S ((S (K S)) ((S (K K)) I))) (K ((S ((S (K -)) I)) (K 1))))))))) ((S ((S ((S (K S)) ((S (K K)) I))) (K ((S I) I)))) ((S ((S (K S)) ((S (K K)) I))) (K ((S I) I)))))
SKI machine result: 120
```

## LLVM IR Generation

Compiling to native machine code, on Mac:

```bash
clang -O3 -o add mfl.ll
```

An alternative way, where we can inspect the optimized IR code:

```bash
# First: brew install llvm@14
# (version 14 since that's what llvmlite wants)

$ /opt/homebrew/opt/llvm/bin/opt -S -O3 mfl.ll -o mfl_optimized.ll
$ /opt/homebrew/opt/llvm/bin/llc -filetype=obj mfl_optimized.ll -o mfl.o
$ clang -o add mfl.o
```
