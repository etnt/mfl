# Generating LLVM IR from an AST: The `Let` Example

This document details how to generate LLVM IR from an Abstract Syntax Tree (AST), specifically focusing on a `Let` expression.  We'll use the example `Let(Var("one"), Int(1), Var("one"))`.

## 1. `Int(1)` Node

The `Int(1)` node represents a literal integer value.  Its corresponding LLVM IR is:

```llvm
%1 = alloca i32                ; Allocate space for a 32-bit integer on the stack
store i32 1, i32* %1           ; Store the value 1 into the allocated space
```

* `%1`: A temporary variable (register or stack location) representing the memory address.  You'll need a mechanism (like a counter or symbol table) to manage these unique names.
* `alloca i32`: Allocates space for a 32-bit integer (`i32`) on the stack.
* `store i32 1, i32* %1`: Stores the integer value `1` at the address `%1`.


## 2. `Var("one")` Nodes

The `Var("one")` nodes have different meanings depending on their context within the `Let` expression:

* **Declaration (`Var("one")` in the first argument):**  This is a variable declaration. The LLVM IR doesn't generate code *directly* here. Instead, this node updates a symbol table with information about the variable:
    * Name: "one"
    * Type: `i32`
    * Address: `%1` (the address allocated by `alloca` in the `Int(1)` handling)

* **Reference (`Var("one")` in the third argument):** This is a variable reference.  The code generator looks up "one" in the symbol table, retrieves its address (`%1`), and generates a `load` instruction:

```llvm
%2 = load i32, i32* %1          ; Load the value from the address %1 into %2
```

This loads the value stored at address `%1` into a new temporary variable `%2`.


## 3. The `Let` Node

The `Let` node is the orchestrator, combining the IR generated from its sub-nodes:

1. **Processes the declaration:**  Handles the first `Var("one")` argument, populating the symbol table.
2. **Processes the value:** Handles the `Int(1)` argument, generating the `alloca` and `store` instructions.
3. **Processes the body:** Handles the second `Var("one")` argument, generating the `load` instruction.
4. **Combines the IR:** The final LLVM IR for the entire `Let` expression would be:

```llvm
%1 = alloca i32                ; Allocate space for 'one'
store i32 1, i32* %1           ; Store 1 into 'one'
%2 = load i32, i32* %1          ; Load the value of 'one' into %2
; Further instructions using %2 would follow here...
```


## Important Considerations

* **Scope Management:**  Handle variable scopes correctly, especially for nested `Let` expressions.
* **Symbol Table:**  A robust symbol table is essential for tracking variable names and their memory locations.
* **Type Checking:**  Perform type checking before IR generation to catch errors.
* **Optimization:** The LLVM optimizer will further refine the generated IR.
* **Error Handling:** Handle potential errors like undeclared variables.


This detailed explanation shows how a simple `Let` expression translates to multiple LLVM IR instructions, emphasizing the importance of memory management, symbol tables, and recursive AST traversal in code generation.  Remember to adapt register names (`%1`, `%2`, etc.) to prevent collisions in your implementation.

## Symbol Table

You push and pop scopes in your symbol table whenever you enter and exit a new lexical scope in your source code. Lexical scope refers to the region of code where a variable is visible. Here's a breakdown of when to use push_scope() and pop_scope() within a compiler's AST traversal:

`push_scope():`

* `Function Definitions:` When your AST traversal encounters a function definition, you should immediately call push_scope(). This creates a new scope for the function's local variables and parameters. Any variables declared within the function will be added to this new scope.

* `Block Statements (e.g., if, else, while, for):` Similarly, when you encounter a block statement (a sequence of statements enclosed in curly braces {} in many languages), push a new scope. Variables declared within the block are only visible inside the block.

* `let or var declarations (in languages with block scope):` If the language's let or var declarations create a new scope, you would push_scope() before processing the declaration and pop_scope() after. In some languages, this is implicit and handled by the block structure itself.

`pop_scope():`

* `End of Function Definitions:` After completing the traversal of a function's body (all statements within the function), call pop_scope(). This discards the function's local scope, returning to the enclosing scope.

* `End of Block Statements:` After processing all the statements within a block, call pop_scope() to remove the block's scope from the symbol table.

* `Exception Handling Blocks:` If your language has try...catch blocks, you'd generally push a scope at the start of the try and pop it at the end of the catch (or finally if present). This ensures that any variables declared within the exception handling blocks are only accessible within those blocks.

```python
class SymbolTable:
    def __init__(self):
        self.scopes = [{}]  # Start with the global scope

    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        if len(self.scopes) > 1:
            self.scopes.pop()
        else:
            raise Exception("Cannot pop global scope")

    def add_variable(self, name, type, address):
        self.scopes[-1][name] = (type, address)

    def lookup_variable(self, name):
        for scope in reversed(self.scopes):  # Search from inner to outer scopes
            if name in scope:
                return scope[name]
        return None

# Example usage:
symbol_table = SymbolTable()
symbol_table.add_variable("a", "i32", "%1")
symbol_table.push_scope()
symbol_table.add_variable("b", "f64", "%2")
symbol_table.add_variable("a", "i8", "%3") # shadowing 'a' in outer scope
print(symbol_table.lookup_variable("a")) # Output: ('i8', '%3')
symbol_table.pop_scope()
print(symbol_table.lookup_variable("a")) # Output: ('i32', '%1')
print(symbol_table.lookup_variable("b")) # Output: None


```