#!/usr/bin/env python3
"""
MFL (Mini Functional Language) Parser and Type Checker

This script provides a command-line interface for parsing, type checking,
and compiling functional programming expressions into Core Erlang or LLVM IR.

Example Usage:
    python3 mfl.py "let id = λx.x in (id 42)"
    python3 mfl.py -v "let id = λx.x in (id 42)"  # verbose mode
    python3 mfl.py --llvm "let id = λx.x in (id 42)"  # generate LLVM IR
"""

import argparse
import subprocess
import shlex  # For safe shell command construction
from mfl_parser import FunctionalParser
from mfl_type_checker import infer_j
from mfl_core_erlang_generator import generate_core_erlang
from mfl_llvm import generate_llvm

def main():
    """
    Main function that handles command-line input and runs the parser.
    """
    # Set up argument parser
    arg_parser = argparse.ArgumentParser(description='Parse and type-check functional programming expressions.')
    arg_parser.add_argument('expression', nargs='?', help='Expression to parse and type-check')
    arg_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output from all modules')
    arg_parser.add_argument('-b', '--backend-verbose', action='store_true', help='Enable verbose output from backend')
    arg_parser.add_argument('-o', '--output', default="mfl", help='Output file name (without suffix)')
    arg_parser.add_argument('-s', '--secd', action='store_true', help='Execute using SECD machine')
    arg_parser.add_argument('-k', '--ski', action='store_true', help='Execute using SKI combinator machine')
    arg_parser.add_argument('-a', '--ast', action='store_true', help='Execute using AST interpreter')
    arg_parser.add_argument('-g', '--gmachine', action='store_true', help='Execute using G-machine')
    arg_parser.add_argument('-l', '--llvm', action='store_true', help='Generate LLVM IR and compile to binary code')
    args = arg_parser.parse_args()

    parser = FunctionalParser([], {}, verbose=args.verbose)  # Grammar rules handled in reduction methods

    # If expression is provided as command-line argument, use it
    if args.expression:
        try:
            ast = parser.parse(args.expression)
            print("Successfully parsed!")
            print(f"AST(pretty): {ast}")

            # Type check the parsed expression
            type_ctx = {}  # Empty typing context
            try:
                expr_type = infer_j(ast, type_ctx)
                print(f"AST(typed): {ast.raw_structure()}")
                print(f"Inferred final type: {expr_type}")

                if args.secd:
                    try:
                        # Execute using SECD machine
                        from mfl_secd import execute_ast
                        result = execute_ast(ast, args.backend_verbose)
                        print(f"SECD machine result: {result}")
                    except Exception as e:
                        print(f"Error executing with SECD machine: {e}")
                elif args.ast:
                    try:
                        # Execute using AST interpreter
                        from mfl_ast import ASTInterpreter
                        ast_interpreter = ASTInterpreter(verbose=args.backend_verbose)
                        result = ast_interpreter.eval(ast)
                        print(f"AST interpreter result: {result}")
                    except Exception as e:
                        print(f"Error executing with AST interpreter: {e}")
                elif args.ski:
                    try:
                        # Execute using SKI machine
                        from mfl_ski import execute_ast 
                        if args.verbose or args.backend_verbose:
                            print("\nExecuting with SKI combinator machine...")
                        result = execute_ast(ast, args.backend_verbose)
                        print(f"SKI machine result: {result}")
                    except Exception as e:
                        print(f"Error executing with SKI machine: {e}")
                elif args.gmachine:
                    try:
                        # Execute using G-machine
                        from mfl_gmachine import execute_ast
                        if args.verbose or args.backend_verbose:
                            print("\nExecuting with G-machine...")
                        result = execute_ast(ast, args.backend_verbose)
                        print(f"G-machine result: {result}")
                    except Exception as e:
                        print(f"Error executing with G-machine: {e}")
                elif args.llvm:
                    try:
                        # Generate LLVM IR code
                        llvm_ir = generate_llvm(ast, args.backend_verbose)
                        if args.verbose:
                            print("\nGenerated LLVM IR code:")
                            print(llvm_ir)
                        # Write the generated code to file
                        ll_file = "mfl.ll"
                        with open(ll_file, "w") as f:
                            f.write(llvm_ir)
                        print(f"LLVM IR written to: {ll_file}")
                        print(f"Compiling as: clang -o {args.output} {ll_file}")
                        try:
                            # Use shlex.quote to safely handle filenames with spaces or special characters
                            command = shlex.split(f"clang -o {args.output} {shlex.quote(ll_file)}")
                            result = subprocess.run(command, capture_output=True, text=True, check=True)
                            print("Compilation successful!")
                            print(result.stdout)  # Print compilation output (if any)
                        except subprocess.CalledProcessError as e:
                            print(f"Error compiling with clang: {e}")
                            print(f"Return code: {e.returncode}")
                            print(f"Stdout: {e.stdout}")
                            print(f"Stderr: {e.stderr}")
                        except FileNotFoundError:
                            print("Error: clang command not found. Make sure it's in your PATH.")
                    except Exception as e:
                        print(f"Error during code generation: {str(e)}")
                else:
                    try:
                        # Generate Core Erlang code
                        core_erlang = generate_core_erlang(ast, expr_type, args.output)
                        if args.verbose:
                            print("\nGenerated Core Erlang code:")
                            print(core_erlang)
                        # Write the generated code to file
                        core_file = f"{args.output}.core"
                        with open(core_file, "w") as f:
                            f.write(core_erlang)
                        print(f"Output written to: {core_file}")
                        print(f"Compiling to BEAM as: erlc +from_core {core_file}")
                        try:
                            # Use shlex.quote to safely handle filenames with spaces or special characters
                            command = shlex.split(f"erlc +from_core {shlex.quote(core_file)}")
                            result = subprocess.run(command, capture_output=True, text=True, check=True)
                            print("Compilation successful!")
                            print(result.stdout)  # Print compilation output (if any)
                        except subprocess.CalledProcessError as e:
                            print(f"Error compiling with erlc: {e}")
                            print(f"Return code: {e.returncode}")
                            print(f"Stdout: {e.stdout}")
                            print(f"Stderr: {e.stderr}")
                        except FileNotFoundError:
                            print("Error: erlc command not found. Make sure it's in your PATH.")
                    except Exception as e:
                        print(f"Error during code generation: {str(e)}")

            except Exception as e:
                print(f"Error during type checking: {str(e)}")

        except ValueError as e:
            print(f"Parse error: {str(e)}")
    else:
        # Default test expressions
        test_exprs = [
            "42",
            "λx.x",
            "(λx.x 42)",
            "let id = λx.x in (id 42)",
            "(2 + 3)",
            "let x = 5 in (x * 3)",
            "let double = λx.(x*2) in (double 21)",
            "(!True)",
            "(!False)",
            "(True & True)",
            "(False | False)",
            "(True | True)",
            "(False & False)",
            "let x = True in (x & False)",
            "let y = False in (!y)"
        ]

        print("No expression provided. Running test cases...")
        for expr_str in test_exprs:
            print("\n" + "="*50)
            try:
                ast = parser.parse(expr_str)
                print(f"AST: {ast}")
                print(f"AST(raw): {ast.raw_structure()}")

                # Type check the parsed expression
                type_ctx = {}
                try:
                    expr_type = infer_j(ast, type_ctx)
                    print(f"Inferred type: {expr_type}")
                except Exception as e:
                    print(f"Type error: {str(e)}")

            except ValueError as e:
                print(f"Parse error: {str(e)}")
            print("="*50)

if __name__ == "__main__":
    main()
