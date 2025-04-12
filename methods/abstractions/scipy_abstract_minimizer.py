from typing import Any, Tuple

import numpy as np
from scipy.optimize import minimize

from methods.abstractions.abstract_optimizator import AbstractOptimizer
from utils.types import ScalarFunction, HistoryDict, InitialPoint, GradientFunction, HessianFunction


class SciPyAbstractOptimizer(AbstractOptimizer):
    """
    Абстрактный оптимизатор.

    Атрибуты:
        name: имя оптимизатора.
        tol: Порог для нормы градиента.
        max_iter: Максимальное число итераций.
        verbose: Вывод подробной информации.
        params: дополнительные параметры, переданные через kwargs.
    """

    def __init__(self, method_name: str, fun: ScalarFunction, x0: InitialPoint, grad: GradientFunction = None,
                 hess: HessianFunction = None, **kwargs: Any) -> None:
        super().__init__("scipy" + method_name, fun, x0, grad, hess, **kwargs)
        self.method_name = method_name

    def optimize(self) -> Tuple[np.ndarray, HistoryDict]:
        history: HistoryDict = {'x': [self.x0.copy()], 'f': [self.fun(self.x0)]}

        def callback(xk):
            history['x'].append(xk.copy())
            history['f'].append(self.fun(xk))

            return callback

        result = minimize(self.counted_function,
                          self.x0,
                          jac=self.counted_gradient,
                          hess=self.counted_hessian,
                          method=self.method_name,
                          callback=callback,
                          options={"maxiter": self.max_iter, "gtol": self.tol, "disp": self.verbose})

        return result.x, history
