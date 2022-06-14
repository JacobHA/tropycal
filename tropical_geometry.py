"""
Defining the basic elements of tropical geometry:
lines.
"""

from tropical_field import MaxPlus, MinPlus

class TropicalPolynomial:
    """
    A class for representing a tropical line.
    """
    def __init__(self, *coefficients, dtype=None) -> None:
        """
        Initialize a tropical polynomial.
        Parameters:
        -----------
        coefficients: list of floats
        dtype: MaxPlus or MinPlus
        """

        if dtype in (MaxPlus, MinPlus):
            self.dtype = dtype
        elif dtype is None:
            raise TypeError('dtype must be specified.')
        else:
            raise TypeError(f'Unknown dtype: {dtype}')

        self.coefficients = [dtype(coeff) for coeff in coefficients]

    def evaluate(self, x: float) -> float:
        """
        Evaluate the polynomial at a point.
        Parameters:
        -----------
        x: float
        Returns:
        --------
        float: the value of the polynomial at x
        """
        x = self.dtype(x) # Convert the input to a MaxPlus object.
        if self.dtype == MaxPlus:
            f = max
        elif self.dtype == MinPlus:
            f = min
        else:
            raise TypeError('Unknown dtype.')
        return f(coefficient * ( x ** i ) for i, coefficient in enumerate(self.coefficients))

    def __str__(self) -> str:
        """
        Return a string representation of the polynomial.
        Returns:
        --------
        str: the string representation of the polynomial
        """
        return 'max(' + ''.join(str(coefficient) + '+' + str(i) + 'x, ' for i, coefficient in enumerate(self.coefficients) )[:-2] + ')'

    def maximization_expr(self):
        """
        Use the coefficients to determine the expressions
        over which we must maximize (minimize).
        Returns:
        --------
        list: the expressions over which we must maximize (minimize)
        """

        from sympy import symbols, Symbol, lambdify
        x = Symbol('x')
        # lambda this:
        return [float(coefficient) + (i * x) for i, coefficient in enumerate(self.coefficients)]

    def calculate_nodes(self):
        """
        Calculate the nodes of the polynomial using SymPy
        Returns:
        --------
        list: the nodes of the polynomial
        """
        solutions = []
        list_of_expressions =self.maximization_expr()
        for i, expression_i in enumerate(list_of_expressions):
            for j, expression_j in enumerate(list_of_expressions):
                if i != j:
                    # Use Sympy to solve the equations.
                    from sympy import solve, Symbol, lambdify
                    x = Symbol('x')
                    solution = solve(expression_i - expression_j, x)
                    for xi in solution:
                        plotted_y = self.evaluate(xi)
                        # Evaluate expression_i at xi
                        expected_y = lambdify(x, expression_i)(xi)
                        if plotted_y == expected_y:
                            solutions.append([xi, plotted_y])

        return solutions

    def plot(self, x_min: float, x_max: float, n_points: int = 1000, show_nodes: bool = True) -> None:
        """
        Plot the polynomial.
        Parameters:
        -----------
        x_min: float
        x_max: float
        n_points: int
        """
        import matplotlib.pyplot as plt
        import numpy as np
        nodes = np.array(self.calculate_nodes()).T

        x_values = np.linspace(x_min, x_max, n_points)
        y_values = [self.evaluate(x) for x in x_values]

        plt.plot(x_values, y_values)
        if show_nodes:
            plt.plot(*nodes, 'ro')
        plt.show()

if __name__ == '__main__':
    l1 = TropicalPolynomial(0, 1., -2, dtype=MaxPlus)
    print(l1.evaluate(3))
    print(l1)
    l1.plot(-10,10)
    print(l1.calculate_nodes())