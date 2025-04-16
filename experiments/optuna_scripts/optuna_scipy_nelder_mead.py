from methods.abstractions.scipy_abstract_minimizer import SciPyAbstractOptimizer
from experiments.optuna_scripts.abstract_optuna import optimize_for_all_functions
from functions.funcs import quadratic_cond_100, grad_quadratic_cond_100, \
    three_hump_camel_function, three_hump_camel_grad, \
    sincos_landscape, grad_sincos_landscape, hess_quadratic_cond_100, three_hump_camel_hess, sincos_landscape_hess

functions = [
    (quadratic_cond_100, grad_quadratic_cond_100, hess_quadratic_cond_100),
    (three_hump_camel_function, three_hump_camel_grad, three_hump_camel_hess),
    (sincos_landscape, grad_sincos_landscape, sincos_landscape_hess)
]


from methods.newton.scipy_nelder_mead import SciPyNelderMead

optimize_for_all_functions(functions, SciPyNelderMead)
