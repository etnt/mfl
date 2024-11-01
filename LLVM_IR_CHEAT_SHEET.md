
# LLVM IR Cheat Sheet

LLVM IR is a low-level programming language used in the LLVM compiler infrastructure for optimization and analysis. This cheat sheet covers the most common constructs in LLVM IR.

---

## Basics

- **Module**: A top-level container for all LLVM IR structures, such as functions, global variables, and metadata.
- **Function**: Defines a single function within a module. Functions consist of basic blocks.
- **Basic Block**: A sequence of instructions with a single entry point and a single exit, ended by a terminator (e.g., `ret`, `br`).
- **Instruction**: Operations within a basic block, such as arithmetic operations, memory access, and branching.

---

## Syntax Essentials

- **Types**: LLVM IR has a strongly typed system.
  - **i32**: 32-bit integer.
  - **i64**: 64-bit integer.
  - **float**: 32-bit floating-point.
  - **double**: 64-bit floating-point.
  - **pointer**: `type*` represents a pointer to `type`.
  - **void**: Indicates no return type for a function.
  - **struct**: Aggregates multiple types together.
- **Values**: Operands in LLVM IR are typed values.
  - **Constants**: Literal values like `i32 10` or `double 3.14`.
  - **Registers**: Virtual registers represented with `%` (e.g., `%1`, `%result`).

---

## Common Instructions

1. **Binary Operations**
   - `add`, `sub`, `mul`, `sdiv`, `udiv`: Arithmetic operations.
     ```llvm
     %result = add i32 %a, %b     ; %result = %a + %b (32-bit integers)
     ```
   - `and`, `or`, `xor`: Bitwise operations.
     ```llvm
     %result = and i32 %x, %y     ; %result = %x & %y
     ```

2. **Memory Access**
   - `alloca`: Allocate memory on the stack.
     ```llvm
     %ptr = alloca i32            ; Allocate 4 bytes (for i32) on stack
     ```
   - `load`: Load a value from memory.
     ```llvm
     %value = load i32, i32* %ptr ; Load the i32 value from address %ptr
     ```
   - `store`: Store a value in memory.
     ```llvm
     store i32 %value, i32* %ptr  ; Store %value at address %ptr
     ```

3. **Control Flow**
   - `br`: Unconditional or conditional branch.
     ```llvm
     br i1 %cond, label %then, label %else  ; Branch based on %cond
     ```
   - `ret`: Return from a function.
     ```llvm
     ret i32 %result               ; Return %result as an i32
     ```

4. **Comparison Operations**
   - `icmp`: Integer comparison, `fcmp`: Floating-point comparison.
     ```llvm
     %is_equal = icmp eq i32 %a, %b  ; %is_equal = %a == %b
     ```
   - Comparison predicates for `icmp`: `eq`, `ne`, `ugt`, `uge`, `ult`, `ule`, `sgt`, `sge`, `slt`, `sle`.
   - Comparison predicates for `fcmp`: `oeq`, `one`, `olt`, `ole`, `ogt`, `oge`, etc.

5. **Casting and Conversion**
   - `trunc`: Truncate to a smaller type.
     ```llvm
     %short = trunc i32 %long to i16 ; Truncate i32 to i16
     ```
   - `zext`: Zero-extend to a larger type.
     ```llvm
     %long = zext i8 %small to i32   ; Zero-extend i8 to i32
     ```
   - `bitcast`: Convert type without changing bits.
     ```llvm
     %float_ptr = bitcast i32* %ptr to float* ; Reinterpret i32* as float*
     ```

6. **Function Calls**
   - `call`: Invoke a function.
     ```llvm
     %result = call i32 @func(i32 %arg)  ; Call function @func with i32 argument
     ```

7. **Phi Node**
   - `phi`: SSA join point for merging values from different control flows.
     ```llvm
     %var = phi i32 [ %value1, %bb1 ], [ %value2, %bb2 ]
     ```

---

## Advanced Constructs

- **Global Variables**
  ```llvm
  @global_var = global i32 42     ; Define a global variable with initial value 42
  ```

- **Structures**
  ```llvm
  %Person = type { i32, i8* }     ; Define a struct with an integer and pointer to i8 (string)
  ```

- **Array**
  ```llvm
  %arr = alloca [10 x i32]        ; Allocate space for an array of 10 i32s
  ```

- **Get Element Pointer (`getelementptr`)**
  - Calculate addresses within arrays or structures.
  ```llvm
  %elem_ptr = getelementptr [10 x i32], [10 x i32]* %arr, i32 0, i32 5 ; Access arr[5]
  ```

- **Attributes and Metadata**
  - **Attributes**: Used for optimizations, such as `noreturn`, `readonly`, and `fast`.
    ```llvm
    declare i32 @foo(i32) #1      ; Function declaration with attribute #1
    ```
  - **Metadata**: Auxiliary information for optimizations/debugging.
    ```llvm
    !dbg !1                       ; Attach debugging metadata
    ```

---

## Example: LLVM IR Function

Here's an example of a simple function that adds two integers and returns the result:

```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
    %sum = add i32 %a, %b          ; Sum %a and %b
    ret i32 %sum                   ; Return the result
}
```

## Key Tips

- **SSA**: All values are assigned once and then used. To merge different values (like in conditional branches), use `phi`.
- **Named Registers**: Use `%` for register names within functions.
- **Instructions and Labels**: Use `label` for branching, with each `br` pointing to a label.

---

This cheat sheet provides a solid foundation for reading and writing basic LLVM IR code.
