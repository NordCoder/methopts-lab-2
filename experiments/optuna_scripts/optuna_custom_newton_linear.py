from methods.newton.newton_linear import NewtonLineSearch

from experiments.optuna_scripts.abstract_optuna import optimize_for_all_functions
from functions.funcs import quadratic_cond_100, grad_quadratic_cond_100, \
    three_hump_camel_function, three_hump_camel_grad, \
    sincos_landscape, grad_sincos_landscape, hess_quadratic_cond_100, three_hump_camel_hess, sincos_landscape_hess

functions = [
    (quadratic_cond_100, grad_quadratic_cond_100, hess_quadratic_cond_100),
    (three_hump_camel_function, three_hump_camel_grad, three_hump_camel_hess),
    (sincos_landscape, grad_sincos_landscape, sincos_landscape_hess)
]

optimize_for_all_functions(functions, NewtonLineSearch)
