from experiments.base.single_test import single_test
from functions.funcs import *

from methods.abstractions.abstract_optimizator import AbstractOptimizer
from methods.linear_search import golden_section_line_search
from methods.newton.newton_linear import NewtonLineSearch


def constant_step(counted_f, x, direction, **params):
    return 1.0


def exp_decay_step(counted_f, x, direction, **params):
    return params.get("initial_lr", 1.0) * np.exp(-params.get("lambda_exp", 0.01) * 0)


def test_with_diff_hyperparams(functions, gradients, hessians):
    hyperparams = [{'left': 0, 'right': 5, 'tol': 1e-6, 'maxiter': 1500, 'initial_lr': 1.0, 'lambda_exp': 0.01}, ]
    x_0s = [np.array([-5, 10])]
    strategies = [constant_step, exp_decay_step, golden_section_line_search]

    for i in range(len(functions)):
        for hyperparam in hyperparams:
            for x0 in x_0s:
                for strategy in strategies:
                    optimizer: AbstractOptimizer = NewtonLineSearch(functions[i],
                                                                    x0,
                                                                    gradients[i],
                                                                    hessians[i],
                                                                    step_selector=strategy,
                                                                    maxiter=hyperparam['maxiter'],
                                                                    linear_left=hyperparam['left'],
                                                                    linear_right=hyperparam['right'],
                                                                    initial_lr=hyperparam['initial_lr'],
                                                                    lambda_exp=hyperparam['lambda_exp'],
                                                                    tol=hyperparam['tol'])

                    single_test(optimizer, f"{functions[i].__name__} {x0} {strategy.__name__}",
                                [functions[i].__name__, str(x0), str(i), strategy.__name__])


test_with_diff_hyperparams([quadratic_cond_100], [grad_quadratic_cond_100], [hess_quadratic_cond_100])
