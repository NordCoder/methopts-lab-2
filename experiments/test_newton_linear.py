import numpy as np

from experiments.base.single_test import single_test
from functions.funcs import hess_quadratic_cond_100, grad_quadratic_cond_100, quadratic_cond_100
from methods.abstractions.abstract_optimizator import AbstractOptimizer
from methods.newton.newton_linear import NewtonLineSearch


def test_with_diff_hyperparams(functions, gradients, hessians):
    hyperparams = [{'left': 0, 'right': 5, 'tol': 1e-6}]  # left right tol
    x_0s = [np.array([-5, 10])]

    for i in range(len(functions)):
        for hyperparam in hyperparams:
            for x0 in x_0s:
                optimizer: AbstractOptimizer = NewtonLineSearch(functions[i],
                                                                x0,
                                                                gradients[i],
                                                                hessians[i],
                                                                linear_left=hyperparam['left'],
                                                                linear_right=hyperparam['right'],
                                                                tol=hyperparam['tol'])

                single_test(optimizer, f"{functions[i].__name__} {x0}",
                            [functions[i].__name__, str(x0), str(i)])


test_with_diff_hyperparams([quadratic_cond_100], [grad_quadratic_cond_100], [hess_quadratic_cond_100])
