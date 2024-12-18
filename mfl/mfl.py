#!/usr/bin/env python3
"""
MFL (Mini Functional Language) Parser and Type Checker
"""

import argparse
import subprocess
import os
import sys
import shlex  # For safe shell command construction
from mfl_parser import FunctionalParser
from mfl_ply_parser import parser as ply_parser
from mfl_type_checker import infer_j
from mfl_core_erlang_generator import generate_core_erlang
from mfl_transform import ASTTransformer



def main():
    """
    Main function that handles command-line input and runs the parser.
    """
    # Set up argument parser
    arg_parser = argparse.ArgumentParser(description='Parse and type-check functional programming expressions.')
    arg_parser.add_argument('expression', nargs='?', help='Expression to parse and type-check')
    arg_parser.add_argument('-v', '--verbose', action='count', help='Increase verbosity (can be specified multiple times)')
    arg_parser.add_argument('-f', '--frontend-only', action='store_true', help='Only run the parser and type checker')
    arg_parser.add_argument('-b', '--backend-verbose', action='store_true', help='Enable verbose output from backend')
    arg_parser.add_argument('-o', '--output', default="output", help='Output file name (without suffix)')
    arg_parser.add_argument('-r', '--run', help='Run the specified file')
    arg_parser.add_argument('-p', '--params', help='Parameters to pass to the program')
    arg_parser.add_argument('-e', '--erlang', action='store_true', help='Compile to BEAM code via Core Erlang')
    arg_parser.add_argument('-s', '--secd', action='store_true', help='Execute using SECD machine')
    arg_parser.add_argument('-k', '--ski', action='store_true', help='Execute using SKI combinator machine')
    arg_parser.add_argument('-a', '--ast', action='store_true', help='Execute using AST interpreter')
    arg_parser.add_argument('-g', '--gmachine', action='store_true', help='Execute using G-machine')
    arg_parser.add_argument('-l', '--llvm', action='store_true', help='Generate LLVM IR and compile to binary code')
    args = arg_parser.parse_args()

    # Set up verbosity levels
    if args.verbose is None:
        info_verbose = False
        frontend_verbose = False
    elif args.verbose == 1:
        info_verbose = True
        frontend_verbose = False
    else:
        info_verbose = True
        frontend_verbose = True

    parser = FunctionalParser([], {}, verbose=args.verbose)  # Grammar rules handled in reduction methods

    if args.run:
        filename, extension = os.path.splitext(args.run)
        extension = extension.lower()
        if extension == ".ski":     # SKI combinator machine
            from mfl_ski import load_and_run_ski_code
            params_ast = None
            if args.params:
                params_ast = ply_parser.parse(args.params)
            result = load_and_run_ski_code(args.run, args.backend_verbose, params_ast)
            print(f"SKI machine result: {result}")
        else:
            print(f"Unknown file extension: {extension}")
            sys.exit(1)
        sys.exit(0)

    # If expression is provided as command-line argument, use it
    if args.expression:
        try:
            ast = ply_parser.parse(args.expression)
            if ast is None:
                print("Parsing failed!")
                sys.exit(1)
            else:
                if info_verbose:
                    print("\nSuccessfully parsed!")
                    print(f"AST(raw): {ast.raw_structure()}")

            # Transform multiple let bindings to a single let's
            transformer = ASTTransformer()
            ast = transformer.multiple_bindings_to_let(ast)
            if info_verbose:
                print("\nTransformed: multiple bindings to single let")
                print(f"AST(transformed): {ast}")
                print(f"AST(transformed, raw): {ast.raw_structure()}")

            # Type check the parsed expression
            type_ctx = {}  # Empty typing context
            try:
                expr_type = infer_j(ast, type_ctx)
                if info_verbose:
                    print("\nType checked!")
                    print(f"AST(typed): {ast.typed_structure()}")
                    print(f"Inferred final type: {expr_type}")

                # Perform program transformations
                transformer = ASTTransformer()
                if args.erlang:
                    ast = transformer.letrec_for_core_erlang(ast)
                else:
                    ast = transformer.transform_letrec_to_let(ast)
                if info_verbose:
                    print("\nTransformed: letrec to: let + Y-combinator")
                    print(f"AST(transformed): {ast}")
                    print(f"AST(transformed, typed): {ast.typed_structure()}")

                # Maybe stop processing here?
                if args.frontend_only:
                    return

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
                        from mfl_ast import execute_ast
                        result = execute_ast(ast, args.backend_verbose)
                        print(f"AST interpreter result: {result}")
                    except Exception as e:
                        print(f"Error executing with AST interpreter: {e}")
                elif args.ski:
                    try:
                        # Execute using SKI machine
                        from mfl_ski import SKIMachine
                        if info_verbose or args.backend_verbose:
                            print("\nExecuting with SKI combinator machine...")
                        ski_machine = SKIMachine(args.backend_verbose)
                        ski_term = ski_machine.to_ski(ast)
                        ski_file = f"{args.output}.ski"
                        ski_machine.save_ski_to_file(ski_term, ski_file)
                        print(f"SKI code saved to: {ski_file}")
                        result = ski_machine.reduce(ski_term)
                        print(f"SKI machine result: {result}")
                    except Exception as e:
                        print(f"Error executing with SKI machine: {e}")
                elif args.gmachine:
                    try:
                        # Execute using G-machine
                        from mfl_gmachine import execute_ast
                        if info_verbose or args.backend_verbose:
                            print("\nExecuting with G-machine...")
                        result = execute_ast(ast, args.backend_verbose)
                        print(f"G-machine result: {result}")
                    except Exception as e:
                        print(f"Error executing with G-machine: {e}")
                elif args.llvm:
                    try:
                        if info_verbose or args.backend_verbose:
                            print("\nGenerating LLVM IR...")
                        from mfl_llvm import LLVMGenerator
                        generator = LLVMGenerator(verbose=args.backend_verbose, generate_comments=True)
                        result = generator.generate(ast)
                        # Verify module
                        llvm_ir = str(generator.module)
                        generator.verify_code(llvm_ir)
                        if info_verbose or args.backend_verbose:
                            print("Module verification successful!")
                        # Write the generated code to file
                        ll_file = f"{args.output}.ll"
                        with open(ll_file, "w") as f:
                            f.write(llvm_ir)
                        print(f"Generated LLVM IR code written to: {ll_file}")
                        print(f"Compiling as: clang -O3 -o {args.output} {ll_file}")
                        try:
                            # Use shlex.quote to safely handle filenames with spaces or special characters
                            command = shlex.split(f"clang -O3 -o {args.output} {shlex.quote(ll_file)}")
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
                elif args.erlang:
                    try:
                        # Generate Core Erlang code
                        core_erlang = generate_core_erlang(ast, args.output)
                        if info_verbose:
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
                else:
                    print("No code generation option specified.")

            except Exception as e:
                print(f"AST(raw): {ast.raw_structure()}")
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
