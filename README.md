# tropycalgebra
Python implementation of tropical algebra objects.

# About
Tropical algebra is the study of algebraic structures, whose base field differs from that of the real numbers. By changing the field's binary operators to $\otimes = +$ and $\oplus = \max$ (or $\oplus = \min$), the tropical algebraic structure is no longer a field (like the reals), it is a semi-field. Although this may be seen as a disadvantage, objects in tropical algebra enjoy many interesting properties and are useful in optimization applications. Furthermore, the geometric interpretation arising from such objects (in the sense of algebraic geometry) leads to some interesting results. [You can read a bit more about the field of tropical algebra at this wikipedia link](https://en.wikipedia.org/wiki/Tropical_semiring) or [this one.](https://en.wikipedia.org/wiki/Tropical_geometry)


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
(A @ B.T).A
>>> [[6.]]
```

And for matrices:
```python
C = MParray(np.array([[1,2],\
                      [3,4] ]), dtype=MaxPlus)
    
(C@A.T).A
>>> [[4.]
     [6.]]
```

A simple example for plotting a tropical polynomial (in one variable only!) can be found in the tropical_geometry.py file:
```python
l1 = TropicalPolynomial(0, 1., -2, dtype=MaxPlus)
# This represents the polynomial "max(0, 1 + x, -2 + 2x)"
l1.plot(-10,10)
```
which results in 

![an image](https://github.com/JacobHA/tropycalgebra/blob/main/lineplot.png)

# TODO:
- [x] Add documentation for the tropical algebra objects
- [ ] Add identities/units
- [ ] Add more examples
- [ ] Eigenvalues/vectors
- [x] Change .arr to .A?
- [ ] Refactor file directory
- [ ] Make tropical field elements subclass a common objects (a general semiring element)
- [ ] Add a print method for the tropical algebra objects
- [ ] Make multivariable polynomials (and plottable too!)
- [ ] More efficient graphing based on node locations
