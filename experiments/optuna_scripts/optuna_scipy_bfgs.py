from experiments.optuna_scripts.abstract_optuna import optimize_for_all_functions, optuna_functions

from methods.steepest_gradient_descent.scipy_bfgs import SciPyBFGS

search_space = {
    "tol":      lambda tr: tr.suggest_float("tol",      1e-8,   1e-4,  log=True),
    "max_iter": lambda tr: tr.suggest_int(  "max_iter", 500,    2000),
    "eps":      lambda tr: tr.suggest_float("eps",      1e-8,   1e-1,  log=True),
}

fixed_kwargs = {
    "verbose": False,
}

optimize_for_all_functions(
    functions=optuna_functions,
    optimizer_class=SciPyBFGS,
    search_space=search_space,
    fixed_kwargs=fixed_kwargs,
    n_trials=100,
)