from experiments.base.single_test import single_test
from functions.funcs import *

from methods.abstractions.abstract_optimizator import AbstractOptimizer
from methods.scheduled_gradient_descent.scheduled_gradient_descent import CustomScheduledGradientDescent


def test_with_diff_hyperparams(functions, grads):
    strategies = ['constant', 'exp_decay', 'piecewise', 'poly_decay']
    hyperparams = [
        {'initial_lr': 0.1, 'step_size': 40, 'alpha': 0.5, 'beta': 1, 'lambda_exp': 0.01, 'tol': 1e-6, 'maxiter': 1500}]

    x_0s = [np.array([1, 1])]

    for i in range(len(functions)):
        for strategy in strategies:
            for hyperparam in hyperparams:
                for x0 in x_0s:
                    print(f"running scheduled {strategy} {x0} {hyperparam} {functions[i].__name__}")
                    optimizer: AbstractOptimizer = CustomScheduledGradientDescent(
                        functions[i],
                        x0,
                        strategy,
                        grads[i],
                        tol=hyperparam['tol'],
                        maxiter=hyperparam['maxiter'],
                        initial_lr=hyperparam['initial_lr'],
                        step_size=hyperparam['step_size'],
                        alpha=hyperparam['alpha'],
                        beta=hyperparam['beta'],
                        lambda_exp=hyperparam['lambda_exp']
                    )

                    single_test(optimizer, f"{functions[i].__name__} {x0} {strategy}",
                                [functions[i].__name__, str(x0), strategy, str(i)])


test_with_diff_hyperparams([quadratic_cond_1], [grad_quadratic_cond_1])
