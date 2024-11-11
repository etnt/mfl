# AST to LLVM IR Translation Strategy

This document outlines the strategy for translating our MFL AST directly to LLVM IR code without using llvmlite. The strategy is based on analyzing the current implementation and the generated LLVM IR output.

## 1. Module Structure

Every generated LLVM IR module should have this basic structure:

```llvm
; ModuleID = "MFL Generated Module"
target triple = "arm64-apple-darwin23.6.0"
target datalayout = ""

; Type definitions
%"lambda_state" = type {i32, i32, i32, i32, i32, i32, i32, i32}

; String constants for printing
@".str.int" = private constant [3 x i8] c"%d\00"
@".str.bool" = private constant [3 x i8] c"%s\00"
@".str.true" = private constant [5 x i8] c"true\00"
@".str.false" = private constant [6 x i8] c"false\00"

; External declarations
declare i32 @"printf"(i8* %".1", ...)
```

## 2. Lambda State Management

For handling closures and captured variables, we use a lambda state structure:

```llvm
%"lambda_state" = type {i32, i32, i32, i32, i32, i32, i32, i32}
```

This structure holds up to 8 captured variables, each as a 32-bit integer. The state is passed between functions to maintain closure context.

## 3. Translation Rules

### 3.1 Integer Literals (Int)
```python
class Int:
    value: int
```
Translation:
```llvm
i32 <value>  ; Direct integer constant
```

### 3.2 Variables (Var)
```python
class Var:
    name: str
```
Translation:
```llvm
; For captured variables in lambda state
%".ptr" = getelementptr %"lambda_state", %"lambda_state"* %state_ptr, i32 0, i32 <index>
%".value" = load i32, i32* %".ptr"
```

### 3.3 Functions (Lambda)
```python
class Function:
    arg: Var
    body: ASTNode
```
Translation:
```llvm
; For a function λx.(x)
define i32 @"compute_1"(%"lambda_state"* %".1", i32 %".2") {
entry:
  ; Store argument in lambda state
  %".5" = getelementptr %"lambda_state", %"lambda_state"* %".1", i32 0, i32 0
  store i32 %".2", i32* %".5"

  ; Load captured variable for body
  %".8" = getelementptr %"lambda_state", %"lambda_state"* %".1", i32 0, i32 0
  %".10" = load i32, i32* %".8"

  ret i32 %".10"
}
```

### 3.4 Function Application (Apply)
```python
class Apply:
    func: ASTNode
    arg: ASTNode
```
Translation:
```llvm
; For (id 8)
%".6" = call i32 @"compute_1"(%"lambda_state"* %"lambda_state_0", i32 8)
```

### 3.5 Let Bindings
```python
class Let:
    name: Var
    value: ASTNode
    body: ASTNode
```
Translation:
```llvm
; For let id = λx.(x) in (id 8)
%".2" = alloca %"lambda_state"
%"lambda_state_0" = alloca %"lambda_state"
; ... function definition ...
; ... body evaluation ...
```

## 4. Main Function Generation

Every module needs a main function that sets up the initial state and handles printing:

```llvm
define i32 @"main"() {
entry:
  ; Allocate lambda state
  %".2" = alloca %"lambda_state"
  %"lambda_state_0" = alloca %"lambda_state"

  ; Evaluate expression
  %".6" = call i32 @"compute_1"(%"lambda_state"* %"lambda_state_0", i32 8)

  ; Print result
  %"str_ptr" = getelementptr [3 x i8], [3 x i8]* @".str.int", i64 0, i64 0
  %".8" = call i32 (i8*, ...) @"printf"(i8* %"str_ptr", i32 %".6")

  ret i32 0
}
```

## 5. Implementation Strategy

1. Create a new class `LLVMIRGenerator` that will handle the translation:
   ```python
   class LLVMIRGenerator:
       def __init__(self):
           self.fresh_counter = 0
           self.output = []
           self.indent_level = 0

       def emit(self, line):
           self.output.append("  " * self.indent_level + line)

       def fresh_name(self, prefix=""):
           name = f"{prefix}_{self.fresh_counter}"
           self.fresh_counter += 1
           return name
   ```

2. Implement visitor methods for each AST node type that emit the corresponding LLVM IR:
   ```python
   def visit_Int(self, node):
       return f"i32 {node.value}"

   def visit_Var(self, node):
       # Generate variable lookup code
       pass

   def visit_Function(self, node):
       # Generate function definition
       pass
   ```

3. Track lambda state indices for captured variables using a symbol table:
   ```python
   class SymbolTable:
       def __init__(self):
           self.scopes = [{}]
           self.current_index = 0

       def add_variable(self, name):
           index = self.current_index
           self.scopes[-1][name] = index
           self.current_index += 1
           return index
   ```

4. Generate unique names for functions and variables using a counter:
   ```python
   def fresh_name(self, prefix=""):
       name = f"{prefix}_{self.fresh_counter}"
       self.fresh_counter += 1
       return name
   ```

## 6. Testing Strategy

1. Start with simple expressions like integers and variables
2. Progress to basic function definitions and applications
3. Test let bindings with increasing complexity
4. Verify output matches expected LLVM IR structure
5. Compile generated IR with clang to verify correctness

## 7. Example Translation

Input AST:
```python
Let(
    Var("id"),
    Function(Var("x"), Var("x")),
    Apply(Var("id"), Int(8))
)
```

Output LLVM IR:
```llvm
; ModuleID = "MFL Generated Module"
target triple = "arm64-apple-darwin23.6.0"
target datalayout = ""

%"lambda_state" = type {i32, i32, i32, i32, i32, i32, i32, i32}
declare i32 @"printf"(i8* %".1", ...)

@".str.int" = private constant [3 x i8] c"%d\00"

define i32 @"main"() {
entry:
  %".2" = alloca %"lambda_state"
  %"lambda_state_0" = alloca %"lambda_state"
  %".6" = call i32 @"compute_1"(%"lambda_state"* %"lambda_state_0", i32 8)
  %"str_ptr" = getelementptr [3 x i8], [3 x i8]* @".str.int", i64 0, i64 0
  %".8" = call i32 (i8*, ...) @"printf"(i8* %"str_ptr", i32 %".6")
  ret i32 0
}

define i32 @"compute_1"(%"lambda_state"* %".1", i32 %".2") {
entry:
  %".5" = getelementptr %"lambda_state", %"lambda_state"* %".1", i32 0, i32 0
  store i32 %".2", i32* %".5"
  %".8" = getelementptr %"lambda_state", %"lambda_state"* %".1", i32 0, i32 0
  %".10" = load i32, i32* %".8"
  ret i32 %".10"
}
