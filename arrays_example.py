"""
A simple example of using the MaxPlus algebra for 2D arrays (matrices).
"""
import numpy as np
from tropical_algebra import MaxPlus, MParray

if __name__ == '__main__':

    x = MParray([1,2], dtype=MaxPlus)
    M = MParray(np.array([[1,2],\
                          [3,4] ]), dtype=MaxPlus)
    print('There are a few ways to use a row/column vector:')
    print((M@x.T).A)
    x = MParray(x.A.T, dtype=MaxPlus)
    print((M@x).A)
    x = MParray(np.array([[1,2]]).T, dtype=MaxPlus)
    print((M@x).A)
