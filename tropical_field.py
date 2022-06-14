"""
Defining the basic elements of tropical semi-field.
"""
class SemiRing(float):
    """
    A class for representing a semi-ring element.
    """
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError

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

    def __pow__(self, n):
        return MaxPlus( n * float(self) )

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
        return MinPlus(float(self), float(y))

    def __pow__(self, n):
        return MinPlus( n * float(self) )
        
