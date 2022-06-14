# tropycalgebra
Python implementation of tropical algebra objects.

# About
Tropical algebra is a mathematical theory of algebraic structures which differ from the algebra of real numbers, by changing the binary operators of $\otimes = +$ and $\oplus = \max$ (or $\oplus = \min$). The tropical algebraic structure is not a field (like the reals), it is a semi-ring. [wiki link](https://en.wikipedia.org/wiki/Tropical_semiring)

With this library, one can instantiate "tropical numbers" which add and multiply as they should in the tropical ring.

Furthermore, one can readily extend the notion of tropical numbers to vectors and matrices, allowing for a study of linear algebra in this framework.

## Examples
For a number, we can use the following syntax for numbers:

```python
a = MaxPlus(4)
b = MaxPlus(6)
a + b
>>> 6
a * b
>>> 10
```

Or for vectors:
```python
A = MParray([1,2], dtype=MaxPlus)
B = MParray([3,4], dtype=MaxPlus)
A.dot(B)
>>> 6
# Which is equivalent to: (we just have to call the array)
(A @ B.T).arr
>>> [[6.]]
```

And for matrices:
```python
C = MParray(np.array([[1,2],\
                      [3,4] ]), dtype=MaxPlus)
    
(C@A.T).arr
>>> [[4.]
     [6.]]
```



# TODO:
- [x] Add documentation for the tropical algebra objects
- [ ] Add identities/units
- [ ] Add more examples
- [ ] Eigenvalues/vectors
- [x] Change .arr to .A?
- [ ] Refactor file directory
