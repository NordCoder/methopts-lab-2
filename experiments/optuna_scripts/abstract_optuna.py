import optuna
import pandas as pd
import os
from typing import Callable

from methods.linear_search import golden_section_line_search

from functions.funcs import *

optuna.logging.set_verbosity(optuna.logging.ERROR)

optuna_functions = [
    (quadratic_cond_100, grad_quadratic_cond_100, hess_quadratic_cond_100),
    (three_hump_camel_function, three_hump_camel_grad, three_hump_camel_hessian),
    (sincos_landscape, grad_sincos_landscape, sincos_hessian)
]

def objective(trial, func: Callable, grad_func: Callable, hess_func: Callable, optimizer_class: Callable) -> float:

    tol = trial.suggest_float('tol', 1e-8, 1e-4, log=True)
    max_iter = trial.suggest_int('max_iter', 500, 2000)
    step_selector = trial.suggest_categorical('step_selector', [golden_section_line_search])

    x0 = np.array([1, 1])

    optimizer = optimizer_class(
        func,
        x0,
        grad=grad_func,
        hess=hess_func,
        tol=tol,
        max_iter=max_iter,
        step_selector=step_selector
    )

    x_opt, history = optimizer.optimize()

    f_opt = history['f'][-1]

    return f_opt


def optimize_for_all_functions(functions: list, optimizer_class: Callable):

    report_dir = 'optuna_report'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    report_file = os.path.join(report_dir, f"{optimizer_class.__name__}_optimization_results.csv")

    results = []

    for func, grad_func, hess_func in functions:
        print(f"Optimizing function: {func.__name__} using {optimizer_class.__name__}")

        study = optuna.create_study(direction='minimize')

        study.optimize(lambda trial: objective(trial, func, grad_func, hess_func, optimizer_class), n_trials=50)

        best_params = study.best_params
        print(f"Best hyperparameters for {func.__name__}: {best_params}")
        print("-" * 50)

        results.append({
            'Function': func.__name__,
            'tol': best_params['tol'],
            'max_iter': best_params['max_iter'],
            'step_selector': best_params['step_selector']
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(report_file, index=False)

