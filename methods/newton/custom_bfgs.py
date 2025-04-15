import numpy as np
from typing import Tuple

from methods.abstractions.abstract_optimizator import AbstractOptimizer
from utils.types import ScalarFunction, HistoryDict, GradientFunction, InitialPoint


def backtracking_line_search(
        f: ScalarFunction,
        x: np.ndarray,
        p: np.ndarray,
        grad: np.ndarray,
        **kwargs
) -> float:
    c = kwargs.get("c", 1e-4)
    tau = kwargs.get("tau", 0.5)
    alpha = kwargs.get("alpha_init", 1.0)

    f_x = f(x)
    while f(x + alpha * p) > f_x + c * alpha * np.dot(grad, p):
        alpha *= tau
        if alpha < 1e-8:
            break
    return alpha


class CustomBfgs(AbstractOptimizer):
    def __init__(self, f: ScalarFunction, x0: InitialPoint, grad: GradientFunction, **kwargs) -> None:
        super().__init__("CustomBFGS", f, x0, grad, **kwargs)
        self.params = kwargs

    def optimize(self) -> Tuple[np.ndarray, HistoryDict]:
        n = self.x0.size
        x = self.x0.copy()
        H = np.eye(n)
        history: HistoryDict = {'x': [x.copy()], 'f': [self.fun(x)]}

        for k in range(self.max_iter):
            grad = self.counted_gradient(x)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {k}, ||grad|| = {grad_norm:.2e}")
                break

            p = -H.dot(grad)
            alpha = backtracking_line_search(self.counted_function, x, p, grad, **self.params)
            s = alpha * p
            x_new = x + s
            f_new = self.fun(x_new)
            history['x'].append(x_new.copy())
            history['f'].append(f_new)

            grad_new = self.counted_gradient(x_new)
            y = grad_new - grad

            if np.dot(y, s) <= 0:
                H = np.eye(n)
            else:
                rho = 1.0 / np.dot(y, s)
                I = np.eye(n)
                H = (I - rho * np.outer(s, y)).dot(H).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)

            x = x_new
            if self.verbose:
                print(f"Iter={k:03d}, α={alpha:.2e}, f(x)={f_new:.6e}, ||grad||={grad_norm:.2e}")

        return x, history
