from typing import Callable

import numpy as np

from utils.types import ScalarFunction


class FunctionCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

    def get_count(self):
        return self.count


def sanitize_filename(filename: str) -> str:
    return filename.replace('{', '_').replace('}', '_').replace(',', '_').replace(' ', '_').replace("'", "_").replace(":", "_")

def numerical_gradient(f: ScalarFunction,
                       x: np.ndarray,
                       eps: float = 1e-9) -> np.ndarray:
    """
    Вычисляет численный градиент функции f в точке x с помощью центральных разностей.

    Для автоматизированного вычисления производных можно использовать библиотеки
    автоматического дифференцирования (например, JAX или autograd).

    Args:
        f: Функция f: ℝⁿ → ℝ.
        x: Точка, в которой считается градиент.
        eps: Малое приращение для разностной схемы (по умолчанию 1e-8).

    Returns:
        Вектор численного градиента (np.ndarray).
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_forward = np.copy(x)
        x_backward = np.copy(x)
        x_forward[i] += eps
        x_backward[i] -= eps
        grad[i] = (f(x_forward) - f(x_backward)) / (2 * eps)
    return grad

def create_numerical_gradient(f: ScalarFunction) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: numerical_gradient(f, x)


def numerical_hessian(f: ScalarFunction, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Численно вычисляет гессиан функции f в точке x с помощью центральных разностей.

    Args:
        f: Скалярная функция f: ℝⁿ → ℝ.
        x: Точка, в которой вычисляется гессиан (np.ndarray размерности n).
        eps: Малое приращение (по умолчанию 1e-5).

    Returns:
        Гессиан f в точке x (матрица n×n).
    """
    n = x.size
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_ijp = x.copy()
            x_ijm = x.copy()
            x_imp = x.copy()
            x_imm = x.copy()

            x_ijp[i] += eps
            x_ijp[j] += eps

            x_ijm[i] += eps
            x_ijm[j] -= eps

            x_imp[i] -= eps
            x_imp[j] += eps

            x_imm[i] -= eps
            x_imm[j] -= eps

            hessian[i, j] = (
                f(x_ijp) - f(x_ijm) - f(x_imp) + f(x_imm)
            ) / (4 * eps ** 2)

    return (hessian + hessian.T) / 2

def create_numerical_hessian(f: ScalarFunction) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: numerical_hessian(f, x)