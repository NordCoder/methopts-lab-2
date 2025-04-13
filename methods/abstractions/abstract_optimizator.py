from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np

from utils.types import ScalarFunction, InitialPoint, HistoryDict, GradientFunction, HessianFunction
from utils.utils import create_numerical_gradient, FunctionCounter


@dataclass
class OptimizationResult:
    x: np.ndarray
    iterations: int
    history: Dict[str, List]
    function_call_count: int
    gradient_call_count: int
    hessian_call_count: int


class AbstractOptimizer(ABC):
    """
    Абстрактный оптимизатор.

    Атрибуты:
        name: имя оптимизатора.
        tol: Порог для нормы градиента.
        max_iter: Максимальное число итераций.
        verbose: Вывод подробной информации.
        params: дополнительные параметры, переданные через kwargs.
    """

    def __init__(self, name: str, fun: ScalarFunction, x0: InitialPoint, gradient: GradientFunction = None, hess: HessianFunction = None,  **kwargs: Any) -> None:
        self.name = name
        self.tol = kwargs.get("tol", 1e-6)
        self.max_iter = kwargs.get("max_iter", 1500)
        self.verbose = kwargs.get("verbose", False)
        self.fun = fun
        self.x0 = x0

        self.gradient = gradient if gradient else create_numerical_gradient(self.fun)
        self.hessian = hess if hess else create_numerical_gradient(self.fun)

        self.counted_function = FunctionCounter(self.fun)
        self.counted_gradient = FunctionCounter(self.gradient)
        self.counted_hessian = FunctionCounter(self.hessian)

    def run(self) -> OptimizationResult:
        x, history = self.optimize()
        return OptimizationResult(
            x=x,
            iterations=len(history['x']),
            history=history,
            function_call_count=self.counted_function.get_count(),
            gradient_call_count=self.counted_gradient.get_count(),
            hessian_call_count=self.counted_hessian.get_count())

    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, HistoryDict]:
        """
        Выполняет оптимизацию функции fun, начиная с точки x0.

        Returns:
            Результаты оптимизации в виде словаря, например:
                ('x': найденное решение (np.ndarray),
                 'history': история итераций)
        """
        pass
