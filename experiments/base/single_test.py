from dataclasses import asdict
from typing import List

from methods.abstractions.abstract_optimizator import AbstractOptimizer, OptimizationResult
from utils.paths import get_report_path
from visualizations.data_table import render_test_table
from visualizations.plotting import plot_full_history, plot_surface_and_trajectory, \
    plot_surface_and_trajectory_with_path


def single_test(optimizer: AbstractOptimizer, plot_title: str, filename_unique_params: List[str]) -> None:
    result: OptimizationResult = optimizer.run()

    draw(optimizer, result, plot_title, filename_unique_params)

    table(optimizer, result, filename_unique_params)

def draw(optimizer: AbstractOptimizer, result: OptimizationResult, plot_title: str, filename_unique_params: List[str]):
    filename = get_report_path(
        optimizer.name,
        "plot",
        "result",
        filename_unique_params,
        ".png"
    )

    plot_full_history(
        history=result.history,
        f=optimizer.fun,
        title=plot_title,
        save_path=filename
    )

    plot_surface_and_trajectory(
        f=optimizer.fun,
        history=result.history,
        title=plot_title,
        save_path=filename.replace(".png", "_dual.png"),
        lim=4  # или 3, или auto
    )

    plot_surface_and_trajectory_with_path(
        f=optimizer.fun,
        history=result.history,
        title=plot_title + " — 3D Trajectory",
        save_path=filename.replace(".png", "_3dtraj.png"),
        lim=4  # можно поставить 3 или 10 — в зависимости от функции
    )

def table(optimizer: AbstractOptimizer, result: OptimizationResult, filename_unique_params: List[str]):
    filename = get_report_path(
        optimizer.name,
        "table",
        "result",
        filename_unique_params,
        ".csv"
    )

    result.history = None

    render_test_table(optimizer.name, filename_unique_params, [asdict(result)], filename)
