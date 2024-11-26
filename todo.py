"""
Todo:
- [ ] Are the .bin files signed or unsigned?
- [ ] Make an interface for multiplying matrices (perhaps just convert array to numpy array and do calculations?)
- [ ] Figure out what to do for the docstrings for abs
- [ ] Make .bin files output to the output folder
- [ ] Remove numpy stuff
- [ ] Go from Python unit tests
"""
# Imports
import numpy as np
from classes import *

# Constants
WORD_SIZE = 4

def bin_to_matrix(filename: str) -> Matrix:
    """Takes a .bin file and writes it into a 2D matrix
    
    Arguments:
    filename -- a string comtaining the input .bin file FILENAME
    Returns: a MATRIX corresponding to the .bin file
    """
    
    with open(filename, "rb") as f:
        num_rows: int = int.from_bytes(f.read(WORD_SIZE), "little")
        num_cols: int = int.from_bytes(f.read(WORD_SIZE), "little")
        matrix = [[0] * num_cols for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                matrix[r][c] = int.from_bytes(f.read(WORD_SIZE), "little")

        return matrix
    