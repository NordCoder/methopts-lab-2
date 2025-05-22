from methods.linear_search import golden_section_line_search
from methods.newton.custom_bfgs import CustomBfgs

from experiments.optuna_scripts.abstract_optuna import optimize_for_all_functions, optuna_functions

search_space = {
    "tol": lambda tr: tr.suggest_float("tol", 1e-8, 1e-4, log=True),
    "max_iter": lambda tr: tr.suggest_int("max_iter", 500, 2000),
    "c": lambda tr: tr.suggest_float("c", 1e-6, 1e-2, log=True),
    "tau": lambda tr: tr.suggest_float("tau", 0.1, 0.9),
    "alpha_init": lambda tr: tr.suggest_float("alpha_init", 1e-2, 10.0, log=True),
}


fixed_kwargs = {
    "verbose": False,
}

optimize_for_all_functions(
    functions=optuna_functions,
    optimizer_class=CustomBfgs,
    search_space=search_space,
    fixed_kwargs=fixed_kwargs,
    n_trials=100,
)
