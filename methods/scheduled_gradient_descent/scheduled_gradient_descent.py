from typing import Tuple, Any
import numpy as np

from methods.abstractions.abstract_optimizator import AbstractOptimizer
from utils.types import HistoryDict, ScalarFunction, InitialPoint, GradientFunction


def update_learning_rate(strategy: str, k: int, **kwargs) -> float:
    """
    Returns the learning rate based on the scheduling strategy.

    :param strategy: One of ('constant', 'piecewise', 'exp_decay', 'poly_decay').
    :param k: Current iteration.
    :param kwargs: Hyperparameters: initial_lr, step_size, alpha, beta, lambda_exp.
    :return: The computed learning rate.
    """
    initial_lr = kwargs.get('initial_lr', 0.01)
    if strategy == 'constant':
        return initial_lr
    elif strategy == 'piecewise':
        step_size = kwargs.get('step_size', 100)
        return initial_lr * (0.5 ** (k // step_size))
    elif strategy == 'exp_decay':
        lambda_exp = kwargs.get('lambda_exp', 0.01)
        return initial_lr * np.exp(-lambda_exp * k)
    elif strategy == 'poly_decay':
        alpha = kwargs.get('alpha', 0.5)
        beta = kwargs.get('beta', 1.0)
        return initial_lr / ((beta * k + 1) ** alpha)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


class CustomScheduledGradientDescent(AbstractOptimizer):
    def __init__(self, fun: ScalarFunction, x0: InitialPoint, strategy: str, grad: GradientFunction = None, **kwargs: Any) -> None:
        """
        Args:
            fun: Objective function f(x).
            x0: Initial approximation (np.ndarray).
            strategy: Learning rate strategy ('constant', 'piecewise', 'exp_decay', 'poly_decay').
            **kwargs: Hyperparameters for the learning rate (initial_lr, step_size, alpha, beta, lambda_exp).
        """
        super().__init__(name="CustomScheduledGradientDescent", fun=fun, x0=x0, gradient=grad, **kwargs)
        self.strategy = strategy
        self.params = kwargs

    def optimize(self) -> Tuple[np.ndarray, HistoryDict]:
        """
        Gradient descent with learning rate scheduling.

        Returns:
            A tuple of (x, iterations, history), where:
                - x: The computed minimizer (np.ndarray).
                - history: A dictionary with history {'x': [...], 'f': [...]}.
        """
        x = self.x0.copy()
        history: HistoryDict = {'x': [x.copy()], 'f': [self.fun(x)]}
        for k in range(self.max_iter):
            g = self.counted_gradient(x)

            grad_norm = np.linalg.norm(g)
            if grad_norm < self.tol:
                if self.verbose:
                    print(f"[STOP] Iteration {k}: Convergence reached (||grad|| = {grad_norm:.2e})")
                break
            lr = update_learning_rate(strategy=self.strategy, k=k, **self.params)
            x = x - lr * g
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                if self.verbose:
                    print(f"[STOP] Iteration {k}: NaN or Inf encountered in x")
                break
            history['x'].append(x.copy())
            history['f'].append(self.fun(x))
            if self.verbose:
                print(f"[{k:03d}] f(x) = {self.fun(x):.6f}, ||grad|| = {grad_norm:.2e}, lr = {lr:.4e}")
        return x, history




