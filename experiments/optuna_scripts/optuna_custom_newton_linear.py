from experiments.optuna_scripts.abstract_optuna import optimize_for_all_functions, optuna_functions
from methods.linear_search import golden_section_line_search
from methods.newton.newton_linear import NewtonLineSearch

search_space = {
    "tol": lambda tr: tr.suggest_float("tol", 1e-8, 1e-4, log=True),
    "max_iter": lambda tr: tr.suggest_int("max_iter", 300, 1500),
    "step_selector": lambda tr: tr.suggest_categorical(
        "step_selector", [golden_section_line_search]
    ),
    "initial_lr": lambda tr: tr.suggest_float("initial_lr", 1e-4, 1e-1, log=True),
    "alpha": lambda tr: tr.suggest_float("alpha", 0.1, 1.0),
}

fixed_kwargs = {
    "verbose": False,
}

optimize_for_all_functions(
    functions=optuna_functions,
    optimizer_class=NewtonLineSearch,
    search_space=search_space,
    fixed_kwargs=fixed_kwargs,
    n_trials=100,
)
