from methods.abstractions.scipy_abstract_minimizer import SciPyAbstractOptimizer
from utils.types import ScalarFunction, InitialPoint, HessianFunction, GradientFunction


class SciPyNelderMead(SciPyAbstractOptimizer):
    def __init__(self, function: ScalarFunction, x0: InitialPoint, grad: GradientFunction, hess: HessianFunction, **kwargs):
        super().__init__("Nelder-Mead", function, x0, grad, hess, **kwargs)