# MFL Abstract Syntax Tree Reference

This document describes the Abstract Syntax Tree (AST) nodes used in the MFL (Mini Functional Language) implementation.

## AST Node Types

All nodes inherit from `ASTNode` base class which provides `raw_structure()` method for debugging.

### Var
Represents a variable reference
```python
Var(name: str)
```
Example: `x` → `Var("x")`

### Int
Represents an integer literal
```python
Int(value: int)
```
Example: `42` → `Int(42)`

### Bool
Represents a boolean literal
```python
Bool(value: bool)
```
Example: `True` → `Bool(True)`

### Function
Represents a lambda function
```python
Function(arg: Var, body: ASTNode)
```
Example: `λx.x` → `Function(Var("x"), Var("x"))`

### Apply
Represents function application
```python
Apply(func: ASTNode, arg: ASTNode)
```
Example: `(f x)` → `Apply(Var("f"), Var("x"))`

### Let
Represents let bindings
```python
Let(name: Var, value: ASTNode, body: ASTNode)
```
Example: `let x = 1 in x` → `Let(Var("x"), Int(1), Var("x"))`

### BinOp
Represents binary operations
```python
BinOp(op: str, left: ASTNode, right: ASTNode)
```
Supported operators:
- Arithmetic: `+`, `-`, `*`, `/`
- Boolean: `&`, `|`

Example: `x + y` → `BinOp("+", Var("x"), Var("y"))`

### UnaryOp
Represents unary operations
```python
UnaryOp(op: str, operand: ASTNode)
```
Supported operators:
- Boolean: `!` (logical not)

Example: `!x` → `UnaryOp("!", Var("x"))`

## Type System

The type system uses:
- `MonoType`: Base class for types
- `TyVar`: Type variables (e.g., `a1`, `a2`)
- `TyCon`: Type constructors (`int`, `bool`, function types)
- `Forall`: Universal quantification for polymorphic types

Built-in types:
- `IntType = TyCon("int", [])`
- `BoolType = TyCon("bool", [])`
- Function types: `TyCon("->", [arg_type, return_type])`
