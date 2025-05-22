from experiments.optuna_scripts.abstract_optuna import optimize_for_all_functions, optuna_functions

from methods.newton.scipy_nelder_mead import SciPyNelderMead

search_space = {
    "tol":      lambda tr: tr.suggest_float("tol",      1e-8, 1e-4, log=True),
    "max_iter": lambda tr: tr.suggest_int(  "max_iter", 500,  2000),
    "xatol":    lambda tr: tr.suggest_float("xatol",    1e-8, 1e-2, log=True),
    "fatol":    lambda tr: tr.suggest_float("fatol",    1e-8, 1e-2, log=True),
    "max_fun":  lambda tr: tr.suggest_int(  "max_fun", 1000, 5000),
}

fixed_kwargs = {
    "verbose": False,
}

optimize_for_all_functions(
    functions=optuna_functions,
    optimizer_class=SciPyNelderMead,
    search_space=search_space,
    fixed_kwargs=fixed_kwargs,
    n_trials=100,
)