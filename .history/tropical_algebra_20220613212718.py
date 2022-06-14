"""
Defining the basic elements of tropical algebra:
real numbers, tropical arrays (vectors and matrices)
"""

import numpy as np

class MaxPlus(float):
    """
    Define a MaxPlus object as a subclass of float.

    Parameters:
    -----------
    float: the (real) value of the MaxPlus object
    """
    def __add__(self, y):
        return MaxPlus(max(self, y))

    def __mul__(self, y):
        return MaxPlus(float(self) + float(y))


class MinPlus(float):
    """
    Define a MinPlus object as a subclass of float.

    Parameters:
    -----------
    float: the (real) value of the MinPlus object
    """
    def __add__(self, y: float):
        return MinPlus(min(self, y))

    def __mul__(self, y: float):
        return MinPlus(np.add(self, y))

def dims(array):
    """
    Gather the dimensions of an input array. Automatically
    yields a row vector shape when input is one dimensional.
    """
    array_length = len(array.shape)
    assert array_length <= 2, \
        f'Array must be one or two dimensional. Array has {array_length} dimensions: {array.shape}'
    if array_length == 2:
        dim0, dim1 = array.shape
    elif array_length == 1:
        dim0 = 1
        dim1 = len(array)
    return dim0, dim1

def dotprod(vector1, vector2, dtype=None):
    """
    Calculate the dot product of two vectors,
    in the max-algebra setting (depending on dtype).
    """
    if dtype == MaxPlus:
        return (vector1 + vector2).max()
    elif dtype == MinPlus:
        return (vector1 + vector2).min()
    elif dtype is None:
        return vector1.dot(vector2)
    else:
        raise TypeError(f'Unknown dtype: {dtype}')

class MParray:
    """
    General array (vector or matrix) for the max/min-plus algebra.
    Parameters:
    -----------

    """
    def __init__(self, array, dtype=None):
        if dtype in ('max', max, MaxPlus):
            self.dtype = MaxPlus
        elif dtype == 'min' or dtype == min or dtype == MinPlus:
            self.dtype = MinPlus

        else:
            # if all([xi.dtype == MaxPlus for xi in self.arr]):
            #     self.dtype = MaxPlus
            # elif all([xi.dtype == MinPlus for xi in self.arr]):
            #     self.dtype = MinPlus
            # else:
            raise TypeError('Cannot determine vector type.')

        assert not isinstance(arr[0], MParray), 'Cannot create MParray from MParray. Use NumPy arrays or list as input.'
        self.arr = np.asarray(arr)
        if len(self.arr.shape) == 1:
            self.arr = self.arr.reshape(1, len(self.arr))

    def __iter__(self):
        return iter(self.arr)

    def __next__(self):
        return next(self.arr)

    def __add__(self, y):
        if isinstance(y, float) or isinstance(y, int):
            return MParray([float(xi) + y for xi in self.arr], dtype=self.dtype)
        else:
            assert self.dtype == y.dtype, f'Cannot add vectors of different (semi-field) types.\nTypes: {self.dtype} and {y.dtype}'
            assert len(self.arr) == len(y.arr), f'Cannot add vectors of different lengths.\nLengths: {len(self.arr)} and {len(y.arr)}'
            if self.dtype == MaxPlus:
                oplus = np.maximum
            elif self.dtype == MinPlus:
                oplus = np.minimum

            return MParray(oplus(self.arr,y.arr), dtype=self.dtype)

    def __mul__(self, y):
        # Hadamard product (elementwise multiplication)
        if isinstance(y, float) or isinstance(y, int):
            return y*self.arr
            
        else:
            assert self.dtype == y.dtype, f'Cannot multiply vectors of different (semi-field) types.\nTypes: {self.dtype} and {y.dtype}'
            return MParray([xi + yi for xi, yi in zip(self.arr, y.arr)], dtype=self.dtype)
        
    
    __rmul__ = __mul__ # Multiplication is commutative

    def __sum__(self):
        res = self.dtype(0)
        for i in self.arr:
            res = i + self.dtype(res)
        return res

    sum = __sum__

    def dot(self, y):
        assert self.dtype == y.dtype, f'Cannot dot vectors of different (semi-field) types.\nTypes: {self.dtype} and {y.dtype}'

        return dotprod(self.arr, y.arr, dtype=self.dtype)

    @property
    def T(self):
        return MParray(self.arr.T, dtype=self.dtype)

    def __matmul__(self, y):
        # implements the @ operation, matrix multiplication
        assert self.dtype == y.dtype, 'Cannot multiply matrices of different types.'

        dims1 = dims(self.arr)
        dims2 = dims(y.arr)

       
        assert dims1[1] == dims2[0], f'Dimensions ({dims1[1]} and {dims2[0]}) are not aligned.'

        result = np.empty(shape=(dims1[0],dims2[1]))
        yT = y.T.arr
        dt=self.dtype
        for i, row_i in enumerate(self):
            for j, col_j in enumerate(yT):
                result[i][j] = dotprod(row_i, col_j, dtype=dt)

        return MParray(result, dtype=self.dtype)

if __name__ == '__main__':
    a = MaxPlus(4)
    b = MaxPlus(3)
    c = MaxPlus(7)
    print( a*(b+c) )
    print( a*b + a*c )
    A = MParray([1,2], dtype=MaxPlus)
    B = MParray([3,4], dtype=MaxPlus)
    print((A.dot(B)))
    print((A@B.T).arr)
    C = MParray(np.array([[1,2],\
                          [3,4] ]), dtype=MaxPlus)
    print((A.dot(A)))
    print((C@A.T).arr)