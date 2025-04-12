import numpy as np
from typing import Any, Callable, Tuple

from methods.abstractions.abstract_optimizator import AbstractOptimizer
from utils.types import ScalarFunction, HistoryDict, InitialPoint, GradientFunction
from methods.linear_search import ternary_search_line


class CustomGradientDescentOptimizer(AbstractOptimizer):
    def __init__(self, fun: ScalarFunction, x0: InitialPoint, grad: GradientFunction = None, line_search_method: Callable[..., float] = ternary_search_line, **kwargs: Any) -> None:
        """
        Args:
            line_search_method: Метод одномерного поиска для выбора длины шага.
            **kwargs: Дополнительные параметры (включая линейный поиск).
        """
        super().__init__(name="CustomSteepestGradientDescent", fun=fun, x0=x0, gradient=grad, **kwargs)
        self.line_search_method = line_search_method
        self.params = kwargs

    def optimize(self) -> Tuple[np.ndarray, HistoryDict]:
        """
        Реализует метод наискорейшего градиентного спуска с адаптивным выбором длины шага.

        На каждой итерации вычисляется градиент (точный или численный), после чего
        находится шаг (alpha) с использованием заданного метода одномерного поиска,
        и выполняется обновление x ← x - α * grad(f(x)).

        Возвращает:
            Tuple[np.ndarray, HistoryDict]:
                - np.ndarray: Найденная точка минимума (x*).
                - HistoryDict: История оптимизации с ключами:
                    - 'x': список пройденных точек (np.ndarray),
                    - 'f': список значений функции в этих точках.
        """

        history: HistoryDict = {'x': [self.x0.copy()], 'f': [self.fun(self.x0)]}
        x = self.x0.copy()

        for i in range(self.max_iter):
            g = self.counted_gradient(x)
            if np.linalg.norm(g) < self.tol:
                if self.verbose:
                    print(f"Сходимость достигнута на итерации {i}")
                break
            d = -g
            alpha = self.line_search_method(self.counted_function, x, d, **self.params)
            x = x + alpha * d
            history['x'].append(x.copy())
            history['f'].append(self.fun(x))
            if self.verbose:
                print(f"Итерация {i}: f(x) = {self.fun(x):.6f}, α = {alpha:.6f}, ||g|| = {np.linalg.norm(g):.6f}")

        return x, history
