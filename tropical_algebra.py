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
        assert isinstance(y, MaxPlus), f'Cannot add MaxPlus with {type(y)}.'
        return MaxPlus(max(self, y))

    def __mul__(self, y):
        assert isinstance(y, MaxPlus), f'Cannot multiply MaxPlus with {type(y)}.'
        return MaxPlus(float(self) + float(y))

class MinPlus(float):
    """
    Define a MinPlus object as a subclass of float.

    Parameters:
    -----------
    float: the (real) value of the MinPlus object
    """
    def __add__(self, y):
        assert isinstance(y, MinPlus), f'Cannot multiply MinPlus with {type(y)}.'
        return MinPlus(min(self, y))

    def __mul__(self, y: float):
        assert isinstance(y, MinPlus), f'Cannot multiply MinPlus with {type(y)}.'
        return MinPlus(np.add(self, y))

def dims(array):
    """
    Gather the dimensions of an input array. Automatically
    yields a row vector shape when input is one dimensional.
    Parameters:
    -----------
    array: np.ndarray or MParray
    Returns:
    --------
    tuple: the dimensions of the input array
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

def dotprod(vector1, vector2, dtype=None) -> float:
    """
    Calculate the dot product of two vectors,
    in the max-algebra setting (depending on dtype).
    Parameters:
    -----------
    vector1: np.ndarray or MParray
    vector2: np.ndarray or MParray
    dtype: MaxPlus or MinPlus
    Returns:
    --------
    float: the dot product of vector1 and vector2
    """
    result = None
    if dtype == MaxPlus:
        result = (vector1 + vector2).max()
    elif dtype == MinPlus:
        result = (vector1 + vector2).min()
    elif dtype is None:
        result = vector1.dot(vector2)
    else:
        raise TypeError(f'Unknown dtype: {dtype}')
    return result

class MParray:
    """
    General array (vector or matrix) for the max/min-plus algebra.
    Parameters:
    -----------
    array: a NumPy array or list 
    dtype: the algebra type (min or max)
    """
    def __init__(self, array, dtype=None):
        if dtype in ('max', max, MaxPlus):
            self.dtype = MaxPlus
        elif dtype in ('min', min, MinPlus):
            self.dtype = MinPlus

        else:
            # if all([xi.dtype == MaxPlus for xi in self.arr]):
            #     self.dtype = MaxPlus
            # elif all([xi.dtype == MinPlus for xi in self.arr]):
            #     self.dtype = MinPlus
            # else:
            raise TypeError('Cannot determine vector type.')

        assert not isinstance(array[0], MParray), \
            'Cannot create MParray from MParray. Use NumPy arrays or list as input.'
        self.array = np.asarray(array)
        if len(self.array.shape) == 1:
            self.array = self.array.reshape(1, len(self.array))

    def __iter__(self):
        """
        Make the MParray iterable. This allows us to enumerate over the elements of the array.
        """
        return iter(self.A)

    @property
    def A(self):
        """ NumPy matrix-like array property. """
        return self.array

    def __add__(self, array):
        """
        Implements the + operation in the given algebra, elementwise addition.
        Parameters:
        -----------
        array: an MParray of compatible dtype (max/min) and size.
        """
        if isinstance(array, (float, int)):
            result = MParray([float(xi) + array for xi in self.A], dtype=self.dtype)
        else:
            assert self.dtype == array.dtype, \
                f'Cannot add vectors of different types: {self.dtype} and {array.dtype}'
            assert len(self.A) == len(array.A), \
                f'Cannot add vectors of different lengths: {len(self.A)} and {len(array.A)}'
            if self.dtype == MaxPlus:
                oplus = np.maximum
            elif self.dtype == MinPlus:
                oplus = np.minimum

            result = MParray(oplus(self.A, array.A), dtype=self.dtype)

        return result

    def __mul__(self, array):
        """
        Implements Hadamard product (elementwise multiplication) in the given algebra.
        Parameters:
        -----------
        array: an MParray of compatible dtype (max/min) and size.
        """
        if isinstance(array, (float, int)):
            return array * self.A
        else:
            assert self.dtype == array.dtype, \
                f'Cannot multiply vectors of different type: {self.dtype} and {array.dtype}.'
            return MParray([xi + yi for xi, yi in zip(self.A, array.A)], dtype=self.dtype)    
    __rmul__ = __mul__ # (Elementwise) multiplication is commutative

    def __sum__(self):
        res = self.dtype(0)
        for i in self.A:
            res = i + self.dtype(res)
        return res

    sum = __sum__

    def dot(self, array):
        """
        Calculate the dot product of two vectors,
        in the max-algebra setting (depending on dtype).
        Parameters:
        -----------
        array: a compatible MParray
        """
        assert self.dtype == array.dtype, \
            f'Cannot dot vectors of different (semi-field) types: {self.dtype} and {array.dtype}.'

        return dotprod(self.A, array.A, dtype=self.dtype)

    @property
    def transpose(self):
        """ NumPy-like transpose method. """
        return MParray(self.A.T, dtype=self.dtype)

    T = transpose

    def __matmul__(self, array):
        """
        Implements the @ operation, matrix multiplication.
        Parameters:
        -----------
        array: an MParray of compatible dtype (max/min) and size
        """
        assert self.dtype == array.dtype, 'Cannot multiply matrices of different types.'

        dims1 = dims(self.A)
        dims2 = dims(array.A)

        assert dims1[1] == dims2[0], f'Dimensions ({dims1[1]} and {dims2[0]}) are not aligned.'

        result = np.empty(shape=(dims1[0],dims2[1]))
        transposed_array = array.T.A
        dt=self.dtype
        for i, row_i in enumerate(self):
            for j, col_j in enumerate(transposed_array):
                result[i][j] = dotprod(row_i, col_j, dtype=dt)

        return MParray(result, dtype=self.dtype)
