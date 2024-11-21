"""Unit tests for SKI combinator serialization and file I/O."""

import sys
import os
import tempfile

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# add ../mfl to the Python path
sys.path.insert(0, os.path.join(current_dir, '../mfl'))

import unittest
from mfl_ply_parser import parser
from mfl_ski import SKIMachine, save_ski_code, load_and_run_ski_code
from mfl_ast import Int, Bool, Apply, Var

class TestSKIIO(unittest.TestCase):

    def setUp(self):
        self.parser = parser
        self.machine = SKIMachine(verbose=False)
        self.test_dir = tempfile.mkdtemp()

    def test_serialize_deserialize_basic(self):
        # Test basic combinators
        ast = self.parser.parse("let id = λx.x in (id 42)")
        ski_term = self.machine.to_ski(ast)
        serialized = self.machine.serialize_ski(ski_term)
        deserialized = self.machine.deserialize_ski(serialized)
        result = self.machine.reduce(deserialized)
        self.assertEqual(result.value, 42)

    def test_serialize_deserialize_arithmetic(self):
        # Test arithmetic operations
        ast = self.parser.parse("5 + 3")
        ski_term = self.machine.to_ski(ast)
        serialized = self.machine.serialize_ski(ski_term)
        deserialized = self.machine.deserialize_ski(serialized)
        result = self.machine.reduce(deserialized)
        self.assertEqual(result.value, 8)

    def test_serialize_deserialize_boolean(self):
        # Test boolean operations
        ast = self.parser.parse("5 > 3")
        ski_term = self.machine.to_ski(ast)
        serialized = self.machine.serialize_ski(ski_term)
        deserialized = self.machine.deserialize_ski(serialized)
        result = self.machine.reduce(deserialized)
        self.assertTrue(result.value)

    def test_file_io_basic(self):
        # Test saving and loading basic SKI code
        ast = self.parser.parse("let id = λx.x in (id 42)")
        filename = os.path.join(self.test_dir, "test_basic.ski")
        save_ski_code(ast, filename)
        result = load_and_run_ski_code(filename)
        self.assertEqual(result.value, 42)

    def test_file_io_arithmetic(self):
        # Test saving and loading arithmetic operations
        ast = self.parser.parse("let double = λx.(x*2) in (double 21)")
        filename = os.path.join(self.test_dir, "test_arithmetic.ski")
        save_ski_code(ast, filename)
        result = load_and_run_ski_code(filename)
        self.assertEqual(result.value, 42)

    def test_file_io_complex(self):
        # Test saving and loading complex expressions
        ast = self.parser.parse("""
            let compose = λf.λg.λx.(f (g x)) in 
            let add1 = λx.(x+1) in 
            let double = λx.(x*2) in 
            (((compose double) add1) 3)
        """)
        filename = os.path.join(self.test_dir, "test_complex.ski")
        save_ski_code(ast, filename)
        result = load_and_run_ski_code(filename)
        self.assertEqual(result.value, 8)  # double(add1(3)) = double(4) = 8

    def test_file_io_conditional(self):
        # Test saving and loading conditional expressions
        ast = self.parser.parse("""
            let max = λx.λy.(if x > y then x else y) in 
            ((max 15) 10)
        """)
        filename = os.path.join(self.test_dir, "test_conditional.ski")
        save_ski_code(ast, filename)
        result = load_and_run_ski_code(filename)
        self.assertEqual(result.value, 15)

    def test_file_io_with_argument(self):
        # Test loading SKI code and applying an argument
        # Save a function that doubles its input
        ast = self.parser.parse("let double =λx.(x*2) in double")
        filename = os.path.join(self.test_dir, "test_with_arg.ski")
        save_ski_code(ast, filename)

        # Test with different arguments
        result = load_and_run_ski_code(filename, arg=21)
        self.assertEqual(result.value, 42)

        result = load_and_run_ski_code(filename, arg=5)
        self.assertEqual(result.value, 10)

    def test_file_io_with_boolean_argument(self):
        # Test loading SKI code and applying a boolean argument
        # Save a function that negates its input
        ast = self.parser.parse("let myor =λx.(x | False) in myor")
        filename = os.path.join(self.test_dir, "test_with_bool_arg.ski")
        save_ski_code(ast, filename)

        # Test with different boolean arguments
        result = load_and_run_ski_code(filename, verbose=True, arg=True)
        self.assertTrue(result.value)

        result = load_and_run_ski_code(filename, arg=False)
        self.assertFalse(result.value)

    def tearDown(self):
        # Clean up temporary test files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

if __name__ == "__main__":
    unittest.main()
