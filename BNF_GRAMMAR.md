# Mini Functional Language BNF Grammar

This document defines the formal grammar for MFL (Mini Functional Language) using Backus-Naur Form (BNF).

## Lexical Elements

```bnf
<letter>     ::= "a" | "b" | ... | "z" | "A" | "B" | ... | "Z"
<digit>      ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
<identifier> ::= <letter> (<letter> | <digit> | "_")*
<integer>    ::= <digit>+
<boolean>    ::= "True" | "False"
```

## Expressions

```bnf
<expr> ::= <var>
         | <literal>
         | <lambda>
         | <application>
         | <let>
         | <letrec>
         | <if>
         | <binary_op>
         | <unary_op>
         | "(" <expr> ")"

<var>    ::= <identifier>
<literal> ::= <integer> | <boolean>

<lambda> ::= "λ" <identifier> "." <expr>
           | "\" <identifier> "." <expr>    (* Alternative lambda syntax *)

<application> ::= "(" <expr> <expr> ")"

<let> ::= "let" <identifier> "=" <expr> "in" <expr>

<letrec> ::= "letrec" <identifier> "=" <expr> "in" <expr>

<if> ::= "if" <expr> "then" <expr> "else" <expr>

<binary_op> ::= <expr> <operator> <expr>

<operator> ::= "+" | "-" | "*" | "/"       (* Arithmetic operators *)
             | "&" | "|"                    (* Boolean operators *)
             | ">" | "<" | "==" | "<=" | ">=" (* Comparison operators *)

<unary_op> ::= "!" <expr>                  (* Logical not *)
```

## Operator Precedence (from highest to lowest)

1. Function application
2. Unary operators (!)
3. Multiplicative operators (*, /)
4. Additive operators (+, -)
5. Comparison operators (>, <, ==, <=, >=)
6. Boolean AND (&)
7. Boolean OR (|)
8. Lambda abstraction
9. Let expressions
10. If expressions

## Examples

```
# Variable
x

# Integer literal
42

# Boolean literal
True

# Lambda function (identity function)
λx.x
\x.x

# Function application
(f x)

# Let binding
let x = 1 in x

# Letrec binding
letrec fac = λn.(if (n == 0) then 1 else (n * (fac (n - 1)))) in (fac 5)

# If expression
if x > 0 then x else 0

# Binary operations
x + y
x * y
x & y
x > y

# Unary operation
!x

# Complex expressions
let f = λx.(x + 1) in (f 5)
if (x > 0) then (f x) else 0
