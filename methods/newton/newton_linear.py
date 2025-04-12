import numpy as np
from typing import Tuple

from methods.abstractions.abstract_optimizator import AbstractOptimizer
from methods.linear_search import golden_section_line_search
from utils.types import HistoryDict, ScalarFunction, GradientFunction, HessianFunction


class NewtonLineSearch(AbstractOptimizer):
    def __init__(self, f: ScalarFunction, x0: np.ndarray, grad: GradientFunction = None, hess: HessianFunction = None, **kwargs) -> None:
        super().__init__("NewtonLineSearch", f, x0, grad, hess, **kwargs)
        self.f = f
        self.x0 = x0.copy()
        self.params = kwargs

    def line_search(self, x: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> float:
        direction = -np.linalg.solve(hess, grad)
        alpha = golden_section_line_search(self.counted_function, x, direction, **self.params)
        return alpha

    def optimize(self) -> Tuple[np.ndarray, HistoryDict]:
        x = self.x0.copy()
        history = {'x': [x.copy()], 'f': [self.f(x)]}

        for i in range(self.max_iter):
            grad = self.counted_gradient(x)
            norm_g = np.linalg.norm(grad)
            if norm_g < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}, ||grad||={norm_g:.2e}")
                break

            hess = self.counted_hessian(x)

            alpha = self.line_search(x, grad, hess)
            x = x - alpha * np.linalg.solve(hess, grad)
            fx = self.f(x)
            history['x'].append(x.copy())
            history['f'].append(fx)
            if self.verbose:
                print(f"Iter={i}, alpha={alpha:.2e}, f(x)={fx:.6e}, ||grad||={norm_g:.2e}")

        return x, history
