import numpy as np

from variables import *
from function import *
import os
from pathlib import Path

def test_error ():
    f = lambda x: 2*(x+2)
    tol = 0.01
    a = -5
    b = 10

# Call the bisection method
root = bisection (f, a, b, tol)

#Define Error, expected answer = -2
error = root + 2
if tol > error:
    print ("test passed")
else:
    print ("test failed")
