import numpy as np

class MaxPlus(float):
    def __add__(self, y):
        return MaxPlus(max(self, y))

    def __mul__(self, y):
        return MaxPlus(float(self) + float(y))


class MinPlus(float):
    def __add__(self, y: float):
        return MinPlus(min(self, y))

    def __mul__(self, y: float):
        return MinPlus(np.add(self, y))

class MParray:
    def __init__(self, arr, dtype=None):
        if dtype == 'max' or dtype == max or dtype == MaxPlus:
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

        if isinstance(arr[0], MParray):
            arr = [xi.arr for xi in arr]

        self.arr = np.asarray(arr)

    def __iter__(self):
        return iter(self.arr)

    def __next__(self):
        return next(self.arr)

    def __add__(self, y):
        if isinstance(y, float) or isinstance(y, int):
            return MParray([float(xi) + y for xi in self.arr], dtype=self.dtype)
        else:
            assert self.dtype == y.dtype, 'Cannot add vectors of different types.'
            assert len(self.arr) == len(y.arr), 'Cannot add vectors of different lengths.'
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
            assert self.dtype == y.dtype, 'Cannot multiply vectors of different types.'
            return MParray([xi + yi for xi, yi in zip(self.arr, y.arr)], dtype=self.dtype)
        
    
    __rmul__ = __mul__ # Multiplication is commutative

    def __sum__(self):
        res = self.dtype(0)
        for i in self.arr:
            res = i + self.dtype(res)
        return res

    sum = __sum__

    def dot(self, y):
        assert self.dtype == y.dtype, 'Cannot dot vectors of different types.'

        if self.dtype == MaxPlus:
            return max(self.arr + y.arr)
        elif self.dtype == MinPlus:
            return min(self.arr + y.arr)

    @property
    def T(self):
        return MParray(self.arr.T, dtype=self.dtype)

    def __matmul__(self, y):
        # implements the @ operation, matrix multiplication
        assert self.dtype == y.dtype, 'Cannot multiply matrices of different types.'

        if len(self.arr.shape) == 2:
            dim1 = self.arr.shape[0]
        elif len(self.arr.shape) == 1:
            dim1 = self.arr.shape[0]
        if len(y.arr.shape) == 2:
            dim2 = y.arr.shape[1]
        elif len(y.arr.shape) == 1:
            dim2 = *y.arr.shape
        print(dim1)
        print(dim2)


        result = np.empty(shape=(dim1,dim2))
        yT = y.T.arr
        dt=self.dtype
        for i, row_i in enumerate(self):
            for j, col_j in enumerate(yT):
                result[i][j] = MParray(row_i, dtype=dt).dot(MParray(col_j, dtype=dt))

        return MParray(result, dtype=self.dtype)

if __name__ == '__main__':
    a = MaxPlus(4)
    b = MaxPlus(3)
    c = MaxPlus(7)
    print( a*(b+c) )
    print( a*b + a*c )
    A = MParray([1,2], dtype=MaxPlus)
    B = MParray([3,4], dtype=MaxPlus)
    # print((A.dot(B)))
    C = MParray(np.array([A,B]), dtype=MaxPlus)
    print((C@A.T).arr)