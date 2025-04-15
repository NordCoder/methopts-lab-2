import numpy as np

from experiments.base.single_test import single_test
from functions.funcs import quadratic_cond_10, grad_quadratic_cond_10
from methods.abstractions.abstract_optimizator import AbstractOptimizer
from methods.newton.custom_bfgs import CustomBfgs


def test_with_diff_hyperparams(functions, grads):
    hyperparams = [{'alpha_init': 1, 'tau': 0.5, 'c': 1e-4, 'tol': 1e-6, 'maxiter': 1500}]

    x_0s = [np.array([1, 1])]

    for i in range(len(functions)):
        for hyperparam in hyperparams:
            for x0 in x_0s:
                print(f"running bfgs {x0} {hyperparam} {functions[i].__name__}")
                optimizer: AbstractOptimizer = CustomBfgs(
                    functions[i],
                    x0,
                    grads[i],
                    tol=hyperparam['tol'],
                    maxiter=hyperparam['maxiter'],
                    alpha_init=hyperparam['alpha_init'],
                    tau=hyperparam['tau'],
                    c=hyperparam['c'],
                )

                single_test(optimizer, f"{functions[i].__name__} {x0} {hyperparam}",
                            [functions[i].__name__, str(x0), str(hyperparam), str(i)])


test_with_diff_hyperparams([quadratic_cond_10], [grad_quadratic_cond_10])
