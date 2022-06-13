# tropycalgebra
Python implementation of tropical algebra objects.

# About
Tropical algebra is a mathematical theory of algebraic structures which differ from the algebra of real numbers, by changing the binary operators of $\otimes = +$ and $\oplus = \max$ (or $\oplus = \min$). The tropical algebraic structure is not a field, it is a semi-field.

With this library, one can instantiate "tropical numbers" which add and multiply as they should in the tropical ring.

Furthermore, one can readily extend the notion of tropical numbers to vectors and matrices, allowing for a study of linear algebra in this framework.

## Examples
For a number, we can use the following syntax:

```python
a = MaxPlus(4)
b = MaxPlus(6)
a + b
>>> 6
a * b
>>> 10
```

# TODO:
- [ ] Add documentation for the tropical algebra objects
- [ ] Add identities/units
- [ ] Add more examples
- [ ] Eigenvalues/vectors
