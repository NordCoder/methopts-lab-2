import numpy as np

from utils.utils import FunctionCounter

def ternary_search_line(f: FunctionCounter, x: np.ndarray, d: np.ndarray, **params):
    """
    Одномерный поиск методом тернарного разбиения для минимизации функции
    φ(α) = f(x + α*d).

    Если границы не заданы, определяется интервал с помощью bracket_minimum.

    Args:
        f: Функция f: ℝⁿ → ℝ.
        x: Текущая точка (np.ndarray).
        d: Направление поиска (например, -градиент).

    Returns:
        float: Оптимальное значение шага α.
    """

    left = params.get('linear_left', 0)
    right = params.get('linear_right', 5)
    tol = params.get('tol', 1e-9)

    phi = lambda alpha: f(x + alpha * d)
    while right - left > tol:
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        if phi(m1) < phi(m2):
            right = m2
        else:
            left = m1
    alpha_opt = (left + right) / 2
    return alpha_opt


def golden_section_line_search(f: FunctionCounter, x: np.ndarray, d: np.ndarray, **params) -> float:
    """
    Одномерный поиск методом золотого сечения для минимизации функции
    φ(α) = f(x + α*d).

    Если границы не заданы, определяется интервал с помощью bracket_minimum.

    Args:
        f: Функция f: ℝⁿ → ℝ.
        x: Текущая точка (np.ndarray).
        d: Направление поиска.

    Returns:
        float: Оптимальное значение шага α.
    """

    left = params.get('linear_left', 0)
    right = params.get('linear_right', 5)
    tol = params.get('tol', 1e-9)

    phi = lambda alpha: f(x + alpha * d)
    invphi = (np.sqrt(5) - 1) / 2
    m1 = left + (1 - invphi) * (right - left)
    m2 = left + invphi * (right - left)
    f1, f2 = phi(m1), phi(m2)
    while right - left > tol:
        if f1 < f2:
            right = m2
            m2 = m1
            f2 = f1
            m1 = left + (1 - invphi) * (right - left)
            f1 = phi(m1)
        else:
            left = m1
            m1 = m2
            f1 = f2
            m2 = left + invphi * (right - left)
            f2 = phi(m2)
    alpha_opt = (left + right) / 2
    return alpha_opt
