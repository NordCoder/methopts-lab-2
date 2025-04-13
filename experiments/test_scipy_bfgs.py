import numpy as np

from experiments.base.single_test import single_test
from functions.funcs import grad_quadratic_cond_100, quadratic_cond_100
from methods.abstractions.abstract_optimizator import AbstractOptimizer
from methods.steepest_gradient_descent.scipy_bfgs import SciPyBFGS


def test_with_diff_hyperparams(functions, gradients):
    hyperparams = [{'maxiter': 1500, 'tol': 1e-6}]
    x_0s = [np.array([-5, 10])]

    for i in range(len(functions)):
        for hyperparam in hyperparams:
            for x0 in x_0s:
                optimizer: AbstractOptimizer = SciPyBFGS(functions[i],
                                                                x0,
                                                                gradients[i],
                                                                maxiter=hyperparam['maxiter'],
                                                                tol=hyperparam['tol'])

                single_test(optimizer, f"{functions[i].__name__} {x0}",
                            [functions[i].__name__, str(x0), str(i)])


test_with_diff_hyperparams([quadratic_cond_100], [grad_quadratic_cond_100])
