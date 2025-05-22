from methods.linear_search import ternary_search_line, golden_section_line_search
from methods.steepest_gradient_descent.steepest_gradient_descent import CustomGradientDescentOptimizer

from experiments.optuna_scripts.abstract_optuna import optimize_for_all_functions, optuna_functions

search_space = {
    "tol": lambda tr: tr.suggest_float("tol", 1e-8,   1e-4, log=True),
    "max_iter": lambda tr: tr.suggest_int("max_iter", 500,   2000),
    "line_search_method":lambda tr: tr.suggest_categorical(
                             "line_search_method",
                             [ternary_search_line, golden_section_line_search]
                         ),
    "a": lambda tr: tr.suggest_float("a",   0.0, 1.0),
    "b": lambda tr: tr.suggest_float("b",   1.0, 10.0),
    "eps": lambda tr: tr.suggest_float("eps", 1e-6, 1e-2, log=True),
}


fixed_kwargs = {
    "verbose": False,
}

optimize_for_all_functions(
    functions=optuna_functions,
    optimizer_class=CustomGradientDescentOptimizer,
    search_space=search_space,
    fixed_kwargs=fixed_kwargs,
    n_trials=100,
)