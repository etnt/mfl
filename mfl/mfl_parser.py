"""
A Shift-Reduce Parser for a simple functional programming language.
"""

from typing import List, Dict, Any
from mfl_ast import (
    Var, Int, Bool, Function, Apply, Let, If, BinOp, UnaryOp
)

class FunctionalParser:
    """
    A shift-reduce parser for functional programming constructs.
    """

    def __init__(self, grammar_rules, terminal_rules, verbose=False):
        self.grammar_rules = grammar_rules
        self.terminal_rules = terminal_rules
        self.stack = []
        self.buffer = []
        self.verbose = verbose

    def debug_print(self, *args, **kwargs):
        """Helper method for conditional printing"""
        if self.verbose:
            print(*args, **kwargs)

    def tokenize(self, input_str: str) -> List[str]:
        """
        Convert input string into tokens.
        Handles special characters and whitespace.
        """
        tokens = []
        i = 0
        while i < len(input_str):
            # Skip whitespace
            if input_str[i].isspace():
                i += 1
                continue

            # Handle two-character operators
            if i < len(input_str) - 1:
                two_chars = input_str[i:i+2]
                if two_chars in ["==", "<=", ">="]:
                    tokens.append(two_chars)
                    i += 2
                    continue

            # Handle single-character operators and other special characters
            if input_str[i] in "()λ.=+*/-&|!<>":
                tokens.append(input_str[i])
                i += 1
                continue

            # Handle words (identifiers, keywords, etc.)
            if input_str[i].isalnum():
                word = ""
                while i < len(input_str) and (input_str[i].isalnum() or input_str[i] == '_'):
                    word += input_str[i]
                    i += 1
                tokens.append(word)
                continue

            i += 1

        return [token for token in tokens if token]

    def try_terminal_reduction(self) -> bool:
        """
        Attempt to reduce terminals (numbers, identifiers, operators).
        """
        if not self.stack:
            return False

        top = self.stack[-1]

        # Skip if already reduced
        if isinstance(top, tuple):
            return False

        # Try to reduce integer literals
        if top.isdigit():
            self.stack.pop()
            self.stack.append(("Expr", Int(int(top))))
            self.debug_print(f"Reduced integer: {top}")
            return True

        # Try to reduce boolean literals
        if top in ["True", "False"]:
            self.stack.pop()
            self.stack.append(("Expr", Bool(top == "True")))
            self.debug_print(f"Reduced boolean: {top}")
            return True

        # Try to reduce identifiers, but not keywords or special chars
        if top.isalnum() and not top.isdigit() and top not in ["let", "in", "if", "then", "else", "λ", "True", "False"]:
            self.stack.pop()
            self.stack.append(("Expr", Var(top)))
            self.debug_print(f"Reduced identifier: {top}")
            return True

        return False

    def reduce_binary_operation(self, start_idx: int) -> bool:
        """Helper method to reduce binary operations"""
        if (isinstance(self.stack[start_idx], tuple) and self.stack[start_idx][0] == "Expr" and
            self.stack[start_idx + 1] in ["+", "-", "*", "/", "&", "|", ">", "<", "==", "<=", ">="] and
            isinstance(self.stack[start_idx + 2], tuple) and self.stack[start_idx + 2][0] == "Expr"):

            _, left = self.stack[start_idx]
            op = self.stack[start_idx + 1]
            _, right = self.stack[start_idx + 2]
            self.stack[start_idx:start_idx + 3] = [("Expr", BinOp(op, left, right))]
            self.debug_print(f"Reduced binary operation: {left} {op} {right}")
            return True
        return False

    def try_reduce_if_expression(self) -> bool:
        """Try to reduce an if expression"""
        if len(self.stack) >= 6:
            if (self.stack[-6] == "if" and
                isinstance(self.stack[-5], tuple) and self.stack[-5][0] == "Expr" and
                self.stack[-4] == "then" and
                isinstance(self.stack[-3], tuple) and self.stack[-3][0] == "Expr" and
                self.stack[-2] == "else" and
                isinstance(self.stack[-1], tuple) and self.stack[-1][0] == "Expr"):

                _, cond = self.stack[-5]
                _, then_expr = self.stack[-3]
                _, else_expr = self.stack[-1]
                self.stack = self.stack[:-6]
                self.stack.append(("Expr", If(cond, then_expr, else_expr)))
                self.debug_print(f"Reduced if: if {cond} then {then_expr} else {else_expr}")
                return True

        return False

    def try_reduce_lambda(self) -> bool:
        """Try to reduce a lambda expression"""
        if len(self.stack) >= 4:
            if (self.stack[-4] == "λ" and
                isinstance(self.stack[-3], tuple) and
                self.stack[-2] == "." and
                isinstance(self.stack[-1], tuple) and self.stack[-1][0] == "Expr"):

                # Extract the variable name
                _, var_expr = self.stack[-3]
                if isinstance(var_expr, str):
                    var_expr = Var(var_expr)
                elif isinstance(var_expr, Var):
                    pass
                else:
                    return False

                _, body = self.stack[-1]
                self.stack = self.stack[:-4]
                self.stack.append(("Expr", Function(var_expr, body)))
                self.debug_print(f"Reduced lambda: λ{var_expr}.{body}")
                return True
        return False

    def try_reduce_application(self) -> bool:
        """Try to reduce a function application"""
        # Look for a sequence of expressions inside parentheses
        if len(self.stack) >= 4:
            if self.stack[-1] == ")":
                # Find the matching opening parenthesis
                depth = 1
                pos = -2
                while depth > 0 and abs(pos) <= len(self.stack):
                    if self.stack[pos] == ")":
                        depth += 1
                    elif self.stack[pos] == "(":
                        depth -= 1
                    pos -= 1
                pos += 1

                if depth == 0:
                    # Extract all expressions between parentheses
                    exprs = []
                    for item in self.stack[pos+1:-1]:
                        if isinstance(item, tuple) and item[0] == "Expr":
                            exprs.append(item[1])

                    if len(exprs) >= 2:
                        # Fold multiple applications from left to right
                        result = exprs[0]
                        for arg in exprs[1:]:
                            result = Apply(result, arg)

                        self.stack = self.stack[:pos] + [("Expr", result)]
                        self.debug_print(f"Reduced application: {result}")
                        return True
        return False

    def try_reduce_let(self) -> bool:
        """Try to reduce a let expression"""
        if len(self.stack) >= 6:
            if (self.stack[-6] == "let" and
                isinstance(self.stack[-5], tuple) and
                self.stack[-4] == "=" and
                isinstance(self.stack[-3], tuple) and self.stack[-3][0] == "Expr" and
                self.stack[-2] == "in" and
                isinstance(self.stack[-1], tuple) and self.stack[-1][0] == "Expr"):

                # Extract the variable
                _, var_expr = self.stack[-5]
                if not isinstance(var_expr, Var):
                    if isinstance(var_expr, str):
                        var_expr = Var(var_expr)
                    else:
                        return False

                _, value = self.stack[-3]
                _, body = self.stack[-1]
                self.stack = self.stack[:-6]
                self.stack.append(("Expr", Let(var_expr, value, body)))
                self.debug_print(f"Reduced let: let {var_expr} = {value} in {body}")
                return True
        return False

    def try_grammar_reduction(self) -> bool:
        """
        Attempt to reduce according to grammar rules.
        """
        if len(self.stack) < 2:
            return False

        # Try reductions in order of precedence
        if self.try_reduce_if_expression():
            return True

        if self.try_reduce_lambda():
            return True

        if self.try_reduce_application():
            return True

        if self.try_reduce_let():
            return True

        # Try to reduce unary not operator: ! e -> UnaryOp
        if len(self.stack) >= 2:
            if (self.stack[-2] == "!" and
                isinstance(self.stack[-1], tuple) and self.stack[-1][0] == "Expr"):
                _, expr = self.stack[-1]
                self.stack = self.stack[:-2]
                self.stack.append(("Expr", UnaryOp("!", expr)))
                self.debug_print(f"Reduced not: !{expr}")
                return True

        # Try to reduce parenthesized expressions
        if len(self.stack) >= 3:
            if (self.stack[-3] == "(" and
                isinstance(self.stack[-2], tuple) and self.stack[-2][0] == "Expr" and
                self.stack[-1] == ")"):
                _, expr = self.stack[-2]
                self.stack = self.stack[:-3]
                self.stack.append(("Expr", expr))
                self.debug_print(f"Reduced parenthesized expression: ({expr})")
                return True

        # Try to reduce arithmetic, boolean, and comparison expressions
        if len(self.stack) >= 3:
            return self.reduce_binary_operation(len(self.stack) - 3)

        return False

    def parse(self, input_str: str) -> Any:
        """
        Parse an input string and return the AST.
        """
        self.buffer = self.tokenize(input_str)
        self.stack = []
        self.debug_print(f"\nParsing: {input_str}")

        while True:
            self.debug_print(f"\nStack: {self.stack}")
            self.debug_print(f"Buffer: {self.buffer}")

            # Try reductions
            reduced = True
            while reduced:
                reduced = False
                # Try all possible reductions
                while self.try_terminal_reduction() or self.try_grammar_reduction():
                    reduced = True

            # If we can't reduce and have items in buffer, shift
            if self.buffer:
                next_token = self.buffer.pop(0)
                self.stack.append(next_token)
                self.debug_print(f"Shifted: {next_token}")
            else:
                # No more input and no reductions possible
                if len(self.stack) == 1 and isinstance(self.stack[0], tuple) and self.stack[0][0] == "Expr":
                    return self.stack[0][1]
                else:
                    raise ValueError(f"Parsing failed. Final stack: {self.stack}")


if __name__ == '__main__':
    parser = FunctionalParser([], {})
    expr = input('Enter an expression: ')
    ast = parser.parse(expr)
    print(f"AST(raw): {ast.raw_structure()}")