from experiments.optuna_scripts.abstract_optuna import optimize_for_all_functions, optuna_functions
from functions.funcs import *
from methods.linear_search import golden_section_line_search
from methods.scheduled_gradient_descent.scheduled_gradient_descent import CustomScheduledGradientDescent

search_space = {
    "tol": lambda tr: tr.suggest_float("tol", 1e-8, 1e-4, log=True),
    "max_iter": lambda tr: tr.suggest_int("max_iter", 500, 2000),
    "strategy": lambda tr: tr.suggest_categorical(
        "strategy", ["constant", "piecewise", "exp_decay", "poly_decay"]
    ),
    "initial_lr": lambda tr: tr.suggest_float("initial_lr", 1e-5, 1.0, log=True),
    "step_size": lambda tr: tr.suggest_int("step_size", 50, 500),
    "lambda_exp": lambda tr: tr.suggest_float("lambda_exp", 1e-4, 1e-1, log=True),
    "alpha": lambda tr: tr.suggest_float("alpha", 0.1, 1.0),
    "beta": lambda tr: tr.suggest_float("beta", 0.1, 10.0),
}

fixed_kwargs = {
    "verbose": False,
}

optimize_for_all_functions(
    functions=optuna_functions,
    optimizer_class=CustomScheduledGradientDescent,
    search_space=search_space,
    fixed_kwargs=fixed_kwargs,
    n_trials=100,
)