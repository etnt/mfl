import unittest
import subprocess
import os
import sys
import gc
from llvmlite import binding as llvm
import shlex  # For safe shell command construction


# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# add ../mfl to the Python path
sys.path.insert(0, os.path.join(current_dir, '../mfl'))

# Import your modules
from mfl_ply_parser import parser as ply_parser
from mfl_type_checker import infer_j
from mfl_llvm import LLVMGenerator 


class TestMFLCompilation(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for compilation
        self.temp_dir = "temp_test_dir"
        os.makedirs(self.temp_dir, exist_ok=True)
        


    def tearDown(self):
        # Clean up the temporary directory after each test
        import shutil
        shutil.rmtree(self.temp_dir)


    def run_mfl_test(self, expr_str, expected_output):
        ast = ply_parser.parse(expr_str)
        type_ctx = {}
        infer_j(ast, type_ctx)

        generator = LLVMGenerator(verbose=False, generate_comments=True)
        result = generator.generate(ast)
        llvm_ir = str(generator.module)

        # Explicitly dispose of LLVM resources
        generator.dispose()
        del generator

        # Trigger garbage collection
        gc.collect()

        try:
            llvm.parse_assembly(llvm_ir)
            print("Module verification successful!")
        except RuntimeError as e:
            for i, line in enumerate(llvm_ir.splitlines(), 1):
                print(f"{i:>{2}} | {line}")
            print(f"Module verification failed: {e}")
            # Re-raise the exception to fail the test
            raise

        ll_file = os.path.join(self.temp_dir, "test.ll")
        with open(ll_file, "w") as f:
            f.write(llvm_ir)

        try:
            # Compile and run using clang
            command = shlex.split(f"clang -O3 -o ./foo {ll_file}")
            subprocess.run(command, capture_output=True, text=True, check=True)
            output = subprocess.run(["./foo"], capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(output, expected_output)
        except subprocess.CalledProcessError as e:
            print(f"Compilation or execution failed for '{expr_str}':")
            print(f"Error: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            raise  # Re-raise the exception to fail the test

    def test_simple_let(self):
        self.run_mfl_test("let one = 1 in one", "1")

    def test_binop_let(self):
        self.run_mfl_test("let result = 3 + 4 in result", "7")

    #def test_let_addition(self):
    #    self.run_mfl_test("let add = λx.λy.(x + y) in ((add 6) 9)", "15")

    #def test_nested_let(self):
    #    self.run_mfl_test("let inc = let add1 = λx.λy.(x+y) in (add1 1 2) in (inc 4)", "7")  # Adjust expected output if needed


if __name__ == '__main__':
    unittest.main()

