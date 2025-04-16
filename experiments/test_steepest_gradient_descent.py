from experiments.base.single_test import single_test
from functions.funcs import *

from methods.abstractions.abstract_optimizator import AbstractOptimizer
from methods.linear_search import ternary_search_line
from methods.steepest_gradient_descent.steepest_gradient_descent import CustomGradientDescentOptimizer


def test_with_diff_hyperparams(functions, gradients):
    strategies = [ternary_search_line]
    hyperparams = [{'left': 0, 'right': 5, 'tol': 1e-6, 'maxiter': 1500}]
    x_0s = [np.array([-5, 10])]

    for i in range(len(functions)):
        for strategy in strategies:
            for hyperparam in hyperparams:
                for x0 in x_0s:
                    optimizer: AbstractOptimizer = CustomGradientDescentOptimizer(functions[i],
                                                                                  x0,
                                                                                  gradients[i],
                                                                                  strategy,
                                                                                  linear_left=hyperparam['left'],
                                                                                  linear_right=hyperparam['right'],
                                                                                  tol=hyperparam['tol'],
                                                                                  maxiter=hyperparam['maxiter'])

                    single_test(optimizer, f"{functions[i].__name__} {x0} {strategy.__name__}",
                                [functions[i].__name__, str(x0), strategy.__name__, str(i)])


test_with_diff_hyperparams([quadratic_cond_100], [grad_quadratic_cond_100])
