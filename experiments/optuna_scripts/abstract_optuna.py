import os
import optuna
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, List, Tuple

from functions.funcs import *

optuna.logging.set_verbosity(optuna.logging.ERROR)

optuna_functions = [
    (quadratic_cond_100, grad_quadratic_cond_100, hess_quadratic_cond_100),
    (three_hump_camel_function, three_hump_camel_grad, three_hump_camel_hessian),
    (sincos_landscape, grad_sincos_landscape, sincos_hessian)
]


def _objective(
    trial: optuna.Trial,
    func: Callable,
    grad_func: Callable,
    hess_func: Callable,
    optimizer_class: Callable,
    x0: np.ndarray,
    search_space: Dict[str, Callable[[optuna.Trial], Any]],
    fixed_kwargs: Dict[str, Any],
) -> float:
    """
    Внутренняя функция-цель для Optuna: собирает гиперпараметры произвольного вида.
    """
    trial_params = {name: suggest_fn(trial) for name, suggest_fn in search_space.items()}

    optimizer_kwargs = {**fixed_kwargs, **trial_params}

    optimizer = optimizer_class(
        func,
        x0,
        grad=grad_func,
        hess=hess_func,
        **optimizer_kwargs,
    )
    _, history = optimizer.optimize()

    return history["f"][-1] 


def optimize_for_all_functions(
    functions: List[Tuple[Callable, Callable, Callable]],
    optimizer_class: Callable,
    search_space: Dict[str, Callable[[optuna.Trial], Any]],
    *,
    x0: np.ndarray | None = None,
    fixed_kwargs: Dict[str, Any] | None = None,
    n_trials: int = 50,
    report_dir: str = "optuna_report",
) -> pd.DataFrame:
    """
    Универсальный перебор гиперпараметров для набора тестовых функций.

    Parameters
    ----------
    functions       : список кортежей (f, grad_f, hess_f)
    optimizer_class : класс оптимизатора, совместимый по сигнатуре
    search_space    : словарь {имя_параметра: lambda trial: trial.suggest_*()}
    x0              : начальная точка (по умолчанию np.array([1, 1]))
    fixed_kwargs    : фиксированные параметры, которые не тюним
    n_trials        : количество экспериментов Optuna на каждую функцию
    report_dir      : куда сохранить итоговый .csv-отчёт

    Returns
    -------
    results_df      : DataFrame с лучшими гиперпараметрами для каждой функции
    """
    if x0 is None:
        x0 = np.array([1.0, 1.0])
    fixed_kwargs = fixed_kwargs or {}

    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(
        report_dir, f"{optimizer_class.__name__}_optimization_results.csv"
    )

    results: List[Dict[str, Any]] = []

    for func, grad_func, hess_func in functions:
        print(f"Optimizing {func.__name__} with {optimizer_class.__name__}")

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda tr: _objective(
                tr,
                func,
                grad_func,
                hess_func,
                optimizer_class,
                x0,
                search_space,
                fixed_kwargs,
            ),
            n_trials=n_trials,
        )

        best_params = study.best_params
        print(f"Best params for {func.__name__}: {best_params}\n{'-'*50}")

        results.append({"Function": func.__name__, **best_params})

    results_df = pd.DataFrame(results)
    results_df.to_csv(report_file, index=False)
    print(f"Full report saved to: {report_file}")

    return results_df
