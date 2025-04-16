from methods.newton.custom_bfgs import CustomBfgs

from experiments.optuna_scripts.abstract_optuna import optimize_for_all_functions
from functions.funcs import quadratic_cond_100, grad_quadratic_cond_100, \
    three_hump_camel_function, three_hump_camel_grad, \
    sincos_landscape, grad_sincos_landscape

functions = [
    (quadratic_cond_100, grad_quadratic_cond_100, None),
    (three_hump_camel_function, three_hump_camel_grad, None),
    (sincos_landscape, grad_sincos_landscape, None)
]

optimize_for_all_functions(functions, CustomBfgs)
