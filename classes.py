import os
from typing import Callable, List, Optional
import unittest
from utils import _test_id_to_name

WORD_SIZE = 4

dirname = os.path.dirname(__file__)
file_output_directory: str = os.path.join(dirname, 'output')

class Matrix:
    # TODO: Write class docstring
    # TODO: Write docstrings for all functions
    def __init__(self, data: list[list[int]]):
        assert len(data) > 0, "Must pass in a non-empty list"
        for row in data: assert type(row) == list, "Each row must be a list" 
        assert len(data[0]) > 0, "Columns must not be empty"
        
        row_len = len(data[0])
        for r in range(len(data)):
            assert type(data[r]) == list, "Each row must be a list" 
            assert row_len == len(data[r]), "Length of each row must be the same"
            for c in range(len(data[0])):
                assert type(data[r][c]) == int
        
        self._data: list[list[int]] = data
        self._rows: int = len(data)
        self._cols: int = len(data[0])
        self._elements: int = self._rows * self._cols

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        assert isinstance(other, Matrix), "Can only multiply Matrix with Matrix"
        assert self._cols == other._rows, "Matrix dimensions do not match"

        result = [[0] * other._cols for _ in range(self._rows)]
        for r in range(self._rows):
            for c in range(other._cols):
                for i in range(self._cols):
                    result[r][c] += self._data[r][i] * other._data[i][c]
        
        return Matrix(result)

    def relu(self) -> 'Matrix':
        return self.map(lambda x: max(x, 0))

    def argmax(self) -> 'Matrix':
        """Calls argmax on the given matrix
        """
        return self.to_vector().argmax().to_matrix(self._rows, self._cols)

    def abs(self) -> 'Matrix':
        return self.map(lambda x: abs(x))

    def map(self, fn: Callable[[int], int]) -> 'Matrix':
        result = self.create_blank_matrix()

        for r in range(self._rows):
            for c in range(self._cols):
                result[r][c] = fn(self._data[r][c])

        return result

    def create_blank_matrix(self) -> 'Matrix':
        """Creates a zeroed out matrix of size ROWS * COLS
        """
        return [[0] * self._cols for _ in range(self._rows)]

    def to_bin(self, bin: 'BinFile') -> None:
        """TODO: fill this docstring in 
        
        Arguments:
        filename --
        """
        # TODO: consider whether filepath should include .bin or not
        filepath: str = bin._filename
        with open(filepath, 'wb') as f:
            f.write(self._rows.to_bytes(WORD_SIZE, 'little'))
            f.write(self._cols.to_bytes(WORD_SIZE, 'little'))
            for r in range(self._rows):
                for c in range(self._cols):
                    elem = self._data[r][c].to_bytes(4, 'little', signed=True)
                    f.write(elem)

    def to_vector(self) -> 'Vector':
        """Turns the Matrix into a new Vector instance
        """
        result = [0] * self._elements
        for r in range(self._rows):
            for c in range(self._cols):
                result[r * self._cols + c] = self._data[r][c]
        return result
    
    def __eq__(self, other: 'Matrix') -> bool:
        assert isinstance(other, Matrix), "Must compare a Matrix with another Matrix"
        return self._data == other._data

    def __ne__(self, other: 'Matrix') -> bool:
        assert isinstance(other, Matrix), "Must compare a Matrix with another Matrix"
        return self._data != other._data
    
    def __repr__(self) -> str:
        # TODO: Matrix representation
        assert False

    def __str__(self) -> str:
        # TODO: make it print out the array but pretty (with good formatting?)
        assert False

    # TODO: possibly make a nice __str__ function that's pretty readable in the terminal?
    # TODO: also make a real __repr__ function
    # TODO: make a convenience method to check for equality in matrices using ==
    # TODO: implement classify as a class method? or tell people to chain together some of these other things (imo implement it but that's just me)


# TODO: Consider renaming Vector to Array?
# TODO: Alternatively, make this component invisible to students and just have them deal with matrices
class Vector:
    def __init__(self, data: list[int]):
        # TODO: assert that all elements are a list
        self._data = data
        self._elements = len(data)
    
    def map(self, fn: Callable[[int], int]):
        for i in range(self._elements):
            self._data[i] = fn(self._data[i])

    def argmax(self) -> 'Vector':
        pass

    def dot(self) -> 'Vector':
        pass

    def to_matrix(self, rows: int, cols: int) -> 'Matrix':
        """Turns the Vector into a new Matrix instance with dimensions ROWS x COLS
        
        Arguments:
        rows -- number of ROWS in our resulting matrix
        cols -- number of COLS in our resulting matrix
        """
        assert rows > 0, "Numbers of rows must be positive"
        assert cols > 0, "Numbers of columns must be positive"

        result = [[0] * self.cols for _ in range(self.rows)]
        for i in range(self._elements):
            r = i // cols
            c = i % cols
            result[r][c] = self._data[i]
        return result


# TODO: Figure out if this class is necessary (I think it's probably for the better (?))
class BinFile:
    def __init__(self, filename: str):
        self._filename = os.path.join(file_output_directory, filename)

    def to_matrix(self) -> Matrix:
        # TODO: assert that _filename exists and is a file
        with open(self._filename, 'rb') as f:
            num_rows: int = int.from_bytes(f.read(WORD_SIZE), 'little')
            num_cols: int = int.from_bytes(f.read(WORD_SIZE), 'little')
            array = [[0] * num_cols for _ in range(num_rows)]
            for r in range(num_rows):
                for c in range(num_cols):
                    array[r][c] = int.from_bytes(f.read(WORD_SIZE), "little")
        
        return Matrix(array)
    
"""
Test design:

- Use unittest library in Python
- Be able to read a Matrix from memory in raw bytes (hopefully with Python) and read it into a Matrix instance (which you can then call matmul on locally)
- For matmul, have a chain test which is for calling convention errors
"""


class AssemblyTest:
    def __init__(self, test: unittest.TestCase, check_calling_convention: bool = True, use_venus_utils: bool = True):
        self.name = _test_id_to_name(test)
        self._test = test
        self._has_executed = False
        self.data: List[str] = []
        self._checks: List[str] = []
        self._args = List[str] = []
        self._call: Optional[str] = None # Is this needed
        self._imports: List[str] = []
        self._arrays: dict = {}
        # TODO: fill with more after you understand
        self.check_calling_convention = check_calling_convention
    
    def include(self, name: str):
        """Include a .s file 
        """
        pass

    def call(self, fn: str):
        assert self._call is None, f"Can only call one function per test. Already called {self._call}"
        self._call = fn

    def input_scalar(self, register: str, value: int):
        """Sets REGISTER to VALUE

        Arguments:
        register -- a string corresponding to the argument a register
        value -- the integer passed into the register
        """
        pass

    def add_data_array(self, arr: List[int]):
        """Adds ARR to the data section in the program.
        """
        pass


# AssemblyTest should only be for inputting arguments and seeing whether the results match what you expect from the output - there should be a way to make this test work for vectors and for matrices (so for both Project 2A and Project 2B)

class AssemblyTest:
    """represents a single assembly test"""

    def __init__(
        self,
        test: unittest.TestCase,
        assembly: str,
        check_calling_convention: bool = True,
        no_utils: bool = False,
    ):
        self.name = _test_id_to_name(test)
        self._test = test
        self._has_executed = False
        self.data: List[str] = []
        self._checks: List[str] = []
        self._args: List[str] = []
        self._call: Optional[str] = None
        self._imports: List[str] = []
        self._array_count: int = 0
        self._msg_count: int = 0
        self._labels: Set[str] = set()
        self._output_regs: Set[int] = set()
        self._arrays: dict = {}
        self._assembly = assembly
        self._program_executed = False
        self._write_files: Set[str] = set()
        self._std_out: Optional[str] = None
        self.check_calling_convention = check_calling_convention

        if not no_utils:
            self.include("utils.s")
        self.include(assembly)

    def include(self, name: str):
        filename = (_source_dir / name).resolve()
        assert filename.is_file(), f"{filename} does not exist"
        self._imports.append(name)

    def call(self, function: str):
        """Specifies which function to call. Remember to provide any input with the `input` method."""
        assert (
            self._call is None
        ), f"Can only call one function per test! Already called {self._call}"
        self._call = function

    # This function puts the arguments into the unittest which is nice because then you can just
    # copy the test to venus, however we recommend students use the optional `args` argument to
    # the `execute` method instead.
    def _input_args(self, args: List[str]):
        """Provides command line arguments through the a0 (argc) and a1 (argv) registers."""
        assert (
            self._call is None
        ), f"You need to specify all inputs before calling `{self._call}`"
        assert isinstance(
            args, list
        ), f"{args} is a {type(args)}, expected a list of strings!"
        assert len(args) > 0, f"Expected a non-empty argument list!"
        assert all(
            isinstance(a, str) for a in args
        ), f"Expected a list of strings, not {[type(a) for a in args]}!"
        # all arguments could potentially be filenames that we write to, so let's just add them
        self._write_files |= set(args)
        # add dummy argument zero
        args = [""] + args
        # allocate args in memory
        arg_strings = [self._str(a, "arg") for a in args]
        # allocate a pointer array for argv
        self.data += [f"argv: .word " + " ".join("0" for _ in range(len(args)))]
        # load argc and argv
        self._args += ["", "# argument count in a0", f"li a0, {len(args)}"]
        self._args += [
            "",
            "# load pointers to argument strings into argv",
            f"la a0, argv",
        ]
        for ii, aa in enumerate(arg_strings):
            self._args += [f"la t1, {aa}", f"sw t1, {ii * 4}(a1)"]

    def input_scalar(self, register: str, value: int):
        """Provides a scalar input through an "a" register"""
        assert (
            self._call is None
        ), f"You need to specify all inputs before calling `{self._call}`"
        assert (
            register in a_regs
        ), f"Register {register} must be one of the a registers!"
        assert isinstance(value, int), f"{value} is a {type(value)}, expected an int!"
        self._args += ["", f"# load {value} into {register}", f"li {register} {value}"]

    def input_array(self, register: str, value: ArrayData):
        """Provides an array input through an "a" register"""
        assert (
            self._call is None
        ), f"You need to specify all inputs before calling `{self._call}`"
        assert (
            register in a_regs
        ), f"Register {register} must be one of the a registers!"
        assert isinstance(
            value, ArrayData
        ), f"{value} is a {type(value)}, expected an array (created with the array([..]) method!"
        name = self._lookup_array(value)
        self._args += [
            "",
            f"# load address to array {name} into {register}",
            f"la {register} {name}",
        ]

    def input_read_filename(self, register: str, filename: str):
        """Provides a filename string input through an "a" register"""
        full_path = (test_asm_dir / filename).resolve()
        if not full_path.is_file():
            print(f"WARN: Input file {full_path} does not exist.")
        self._input_filename(register, filename)

    def input_write_filename(self, register: str, filename: str):
        """Provides a filename string input through an "a" register"""
        dir_path = (test_asm_dir / filename).resolve().parent
        if not dir_path.is_dir():
            print(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        self._write_files.add(filename)
        self._input_filename(register, filename)

    def _input_filename(self, register: str, filename: str):
        assert (
            self._call is None
        ), f"You need to specify all inputs before calling `{self._call}`"
        assert (
            register in a_regs
        ), f"Register {register} must be one of the a registers!"
        path = self._str(filename)
        self._args += [
            "",
            f"# load filename {filename} into {register}",
            f"la {register} {path}",
        ]

    def check_scalar(self, register: str, value: int):
        """checks the the value of register"""
        assert (
            self._call is not None
        ), f"You must first call a function before checking its return values!"
        assert isinstance(value, int), f"{value} is a {type(value)}, expected an int!"
        """ Checks that when this function is called, we have not already assembled and run the test. """
        assert not self._has_executed, f"Test has already been assembled and run!"
        exit_code = 8
        saved_register = self._parse_register(register)
        lbl = self._make_lbl(f"{register}_eq_{value}")
        msg = f"msg{self._msg_count}"
        self._msg_count += 1
        self.data += [f'{msg}: .asciiz "Expected {register} to be {value} not: "']
        self._checks += [
            "",
            f"# check that {register} == {value}",
            f"li t0 {value}",
            f"beq {saved_register} t0 {lbl}",
            "# print error and exit",
            f"la a0, {msg}",
            "jal print_str",
            f"mv a0 {saved_register}",
            "jal print_int",
            "# Print newline",
            "li a0 '\\n'",
            "jal ra print_char",
            f"# exit with code {exit_code} to indicate failure",
            f"li a0 {exit_code}",
            "jal exit",
            f"{lbl}:",
            "",
        ]

    def check_array(self, array: ArrayData, value: List[int]):
        """checks the the value of an array in memory"""
        assert (
            self._call is not None
        ), f"You must first call a function before checking its return values!"
        """ Checks that when this function is called, we have not already assembled and run the test. """
        assert not self._has_executed, f"Test has already been assembled and run!"
        assert (
            len(value) > 0
        ), "Array to compare against has to contain at least one element."
        assert isinstance(
            array, ArrayData
        ), f"Input ({array}) was of the wrong type. Expected a t.array() return value"
        assert len(value) <= len(
            array
        ), "Array to compare against must contain a smaller or equal amount of elements."
        expected = self.array(value).name
        actual = "la a2, " + self._lookup_array(array)
        self._compare_int_array(array.name, actual, expected, value, exit_code=2)

    def check_array_pointer(self, register: str, value: List[int]):
        """check the memory region pointed to by the register content"""
        assert (
            self._call is not None
        ), f"You must first call a function before checking its return values!"
        """ Checks that when this function is called, we have not already assembled and run the test. """
        assert not self._has_executed, f"Test has already been assembled and run!"
        assert (
            len(value) > 0
        ), "Array to compare against has to contain at least one element."
        saved_register = self._parse_register(register)
        array_name = f"array pointed to by {register}"
        expected = self.array(value).name
        actual = f"mv a2 {saved_register}"
        self._compare_int_array(array_name, actual, expected, value, exit_code=2)

    def check_file_output(self, actual: str, expected: str):
        """compares the actual file to the expected file"""
        assert (
            self._program_executed
        ), f"You first need to `execute` the program before checking its outputs!"
        assert (
            actual in self._write_files
        ), f"Unknown output file {actual}. Did you forget to provide it to the program by calling input_write_filename?"
        full_expected = (test_asm_dir / expected).resolve()
        assert (
            full_expected.is_file()
        ), f"Reference file {full_expected} does not exist!"
        # check to make sure the output file exists
        full_actual = (test_asm_dir / actual).resolve()
        if not full_actual.is_file():
            self._test.fail(
                f"It seems like the program never created the output file {full_actual}",
            )
        # open and compare the files
        with open(full_actual, "rb") as a:
            actual_bin = a.read()
        with open(full_expected, "rb") as e:
            expected_bin = e.read()
        if actual_bin != expected_bin:
            self._test.fail(f"Bytes of {actual} and {expected} did not match!")

    def check_stdout(self, expected: str):
        """compares the output of the program"""
        assert (
            self._std_out is not None
        ), f"You first need to `execute` the program before checking stdout!"
        line = "-" * 35
        if self._std_out.strip() != expected.strip():
            assert_msg = f"\n{line}\nExpected stdout:\n{expected.strip()}\n{line}\nActual stdout:\n{self._std_out.strip()}\n{line}"
            self._test.fail(assert_msg)

    def _parse_register(self, register: str) -> str:
        assert register in a_regs, "Only a registers can be checked"
        register_index = int(register[1:])
        assert (
            register_index not in self._output_regs
        ), f"Register {register} was already checked!"
        self._output_regs.add(register_index)
        return f"s{register_index}"

    def _compare_int_array(
        self,
        array_name: str,
        actual: str,
        expected: str,
        value: List[int],
        exit_code: int,
    ):
        value_str = " ".join(str(v) for v in value)
        msg = self._str(
            f"Expected {array_name} to be:\\n{value_str}\\nInstead it is:\\n"
        )
        self._checks += [
            "",
            "##################################",
            f"# check that {array_name} == {value}",
            "##################################",
            "# a0: exit code",
            f"li a0, {exit_code}",
            "# a1: expected data",
            f"la a1, {expected}",
            "# a2: actual data",
            actual,
            "# a3: length",
            f"li a3, {len(value)}",
            "# a4: error message",
            f"la a4, {msg}",
            "jal compare_int_array",
        ]

    _can_fail = {"fopen", "fclose", "fread", "fwrite", "malloc", ""}

    def execute(
        self,
        code: int = 0,
        args: Optional[List[str]] = None,
        fail: str = "",
        verbose: bool = False,
        always_print_stdout: bool = False,
    ):
        if "-mcv" in _venus_default_args:
            always_print_stdout = True

        """Assembles the test and runs it through the venus simulator."""
        assert (
            fail in AssemblyTest._can_fail
        ), f"Invalid fail={fail}. Can only fail: {list(AssemblyTest._can_fail)}"

        """ As soon as this function is called, the AssemblyTest is considered "executed" for the duration of the life cycle of this test and should be treated as such. """
        self._has_executed = True

        # turn function to fail into a define
        if len(fail) == 0:
            defines = []
        else:
            ret = 0 if fail == "malloc" else -1
            defines = ["--def", f"#{fail.upper()}_RETURN_HOOK=li a0 {ret}"]

        # check arguments
        if args is not None:
            # TODO: check to see if any args clash with venus arguments
            assert len(args) > 0, "use None if you don't want to pass any arguments"
            for a in args:
                assert not a.startswith(
                    "-"
                ), f"argument '{a}' starting with '-' is not allowed"
            # all arguments could potentially be filenames that we write to, so let's just add them
            self._write_files |= set(args)
        else:
            # ensure that args is always a list
            args = []

        lines = []

        lines += [f".import ../src/{i}" for i in self._imports]
        lines += ["", ".data"] + self.data
        lines += [
            "",
            ".globl main_test",
            ".text",
            "# main_test function for testing",
            "main_test:",
        ]

        # prologue
        if len(self._output_regs) > 0:
            assert (
                len(self._output_regs) < 13
            ), f"Too many output registers: {len(self._output_regs)}!"
            p = [
                "# Prologue",
                f"addi sp, sp, -{4 * (len(self._output_regs) + 1)}",
                "sw ra, 0(sp)",
            ]
            p += [f"sw s{i}, {(i+1) * 4}(sp)" for i in range(len(self._output_regs))]
            lines += _indent(p + [""])

        lines += _indent(self._args)

        assert self._call is not None, "No function was called!"
        foo_call = ["", f"# call {self._call} function", f"jal ra {self._call}"]
        lines += _indent(foo_call)

        if len(self._output_regs) > 0:
            lines += _indent(["", "# save all return values in the save registers"])
            lines += _indent([f"mv s{i} a{i}" for i in self._output_regs] + [""])

        lines += _indent(self._checks)
        if code != 0:
            lines += _indent(
                [f"# we expect {self._call} to exit early with code {code}"]
            )

        lines += _indent(["", "# exit normally"])
        # epilogue
        if len(self._output_regs) > 0:
            p = ["# Epilogue", "lw ra, 0(sp)"]
            p += [f"lw s{i}, {(i + 1) * 4}(sp)" for i in range(len(self._output_regs))]
            p += [f"addi sp, sp, {4 * (len(self._output_regs) + 1)}"]
            lines += _indent(p + [""])
        # lines += _indent(["mv a0, zero", "ret"])
        lines += _indent([f"li a0 0", "jal exit"])
        lines += [""]

        if verbose:
            print()
        filename = save_assembly(self.name, "\n".join(lines), verbose=verbose)
        r, coverage = run_venus(
            filename, self.check_calling_convention, defines, args, verbose=verbose
        )
        _process_coverage(coverage, self._assembly)
        self._program_executed = True
        self._std_out = r.stdout.decode("UTF-8")
        venus_stderr_clean = (
            r.stderr.decode("UTF-8").replace("Found 0 warnings!", "").strip()
        )
        if r.returncode != code or venus_stderr_clean != "":
            self._print_failure(r, code)
        elif always_print_stdout:
            print(
                "stdout:\n"
                + r.stdout.decode("UTF-8")
                + "\n\nstderr:\n"
                + r.stderr.decode("UTF-8")
            )

    def _print_failure(self, r, expected_code):
        venus_out = (
            "stdout:\n"
            + r.stdout.decode("UTF-8")
            + "\n\nstderr:\n"
            + r.stderr.decode("UTF-8")
        )
        if expected_code != r.returncode:
            self._test.fail(
                f"Venus returned exit code {r.returncode} not {expected_code}.\n{venus_out}"
            )
        else:
            self._test.fail(
                f"Unexpected results from venus (exited with {r.returncode}).\n{venus_out}"
            )

    def _make_lbl(self, prefix: str) -> str:
        name = prefix
        ii = 0
        while name in self._labels:
            name = f"{prefix}_{ii}"
            ii += 1
        self._labels.add(name)
        return name

    def _lookup_array(self, a: ArrayData) -> str:
        assert (
            a.name in self._arrays
        ), f"Unknown array {a.name}. Did you declare it for this test?"
        assert (
            self._arrays[a.name] is a
        ), f"Array {a.name} was declared with a different test!"
        return a.name

    def array(self, data: List[int]) -> ArrayData:
        name = f"m{self._array_count}"
        self._array_count += 1
        self.data += [".align 4", f"{name}: .word " + " ".join((str(v) for v in data))]
        a = ArrayData(name, data)
        self._arrays[a.name] = a
        return a

    def _str(self, data: str, prefix: str = "msg") -> str:
        name = f"{prefix}{self._msg_count}"
        self._msg_count += 1
        self.data += [f'{name}: .asciiz "{data}"']
        return name

