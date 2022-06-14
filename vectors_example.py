"""
A simple example of using the MaxPlus algebra for vectors.
"""
from tropical_algebra import MaxPlus, MParray

if __name__ == '__main__':

    x = MParray([-1,4], dtype=MaxPlus)
    y = MParray([2,-4], dtype=MaxPlus)
    print(x.dot(y))
    print(y.dot(x))
    print((x@y.T).A)
