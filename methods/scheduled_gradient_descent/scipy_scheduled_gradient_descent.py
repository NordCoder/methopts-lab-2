from utils.types import ScalarFunction, InitialPoint
from methods.abstractions.scipy_abstract_minimizer import SciPyAbstractOptimizer


class SciPySteepestGradientDescent(SciPyAbstractOptimizer):
    def __init__(self, function: ScalarFunction, x0: InitialPoint, **kwargs):
        super().__init__("CG", function, x0, **kwargs)