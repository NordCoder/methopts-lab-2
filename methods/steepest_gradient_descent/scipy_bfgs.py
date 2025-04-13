from utils.types import ScalarFunction, InitialPoint, GradientFunction
from methods.abstractions.scipy_abstract_minimizer import SciPyAbstractOptimizer


class SciPyBFGS(SciPyAbstractOptimizer):
    def __init__(self, function: ScalarFunction, x0: InitialPoint, grad: GradientFunction = None, **kwargs):
        super().__init__("BFGS", function, x0, grad=grad, **kwargs)
