from experiments.base.single_test import single_test
from functions.funcs import *

from methods.abstractions.abstract_optimizator import AbstractOptimizer
from methods.newton.scipy_newton_cg import SciPyNewtonCG


def test_with_diff_hyperparams(functions, gradients, hessians):
    hyperparams = [{'maxiter': 1500, 'tol': 1e-6}]
    x_0s = [np.array([-5, 10])]

    for i in range(len(functions)):
        for hyperparam in hyperparams:
            for x0 in x_0s:
                optimizer: AbstractOptimizer = SciPyNewtonCG(functions[i],
                                                             x0,
                                                             gradients[i],
                                                             hessians[i],
                                                             maxiter=hyperparam['maxiter'],
                                                             tol=hyperparam['tol'])

                single_test(optimizer, f"{functions[i].__name__} {x0}",
                            [functions[i].__name__, str(x0), str(i)])


test_with_diff_hyperparams([quadratic_cond_100], [grad_quadratic_cond_100], [hess_quadratic_cond_100])
