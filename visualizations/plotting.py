import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Any, Callable

# Типовые алиасы для удобства
HistoryDict = Dict[str, List[Any]]
ScalarFunction = Callable[[np.ndarray], float]


# ==============================
# Вспомогательные функции
# ==============================
def compute_meshgrid_data(f: ScalarFunction,
                          x_min: float, x_max: float,
                          y_min: float, y_max: float,
                          num_points: int = 300) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Вычисляет сетку (X, Y) и соответствующие значения Z = f([x, y]) на этой сетке.

    Args:
        f: Целевая функция от np.ndarray.
        x_min: Минимальное значение по оси X.
        x_max: Максимальное значение по оси X.
        y_min: Минимальное значение по оси Y.
        y_max: Максимальное значение по оси Y.
        num_points: Число точек по каждой оси (по умолчанию 300).

    Returns:
        Кортеж (X, Y, Z), где X и Y — массивы сетки, а Z — двумерный массив значений f.
    """
    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([[f(np.array([x, y])) for x in x_vals] for y in y_vals])
    return X, Y, Z


def save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """
    Если указан путь сохранения, сохраняет фигуру, иначе отображает её.

    Args:
        fig: Объект Figure.
        save_path: Путь для сохранения, или None.
    """
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


# ==============================
# Функции построения графиков
# ==============================
def plot_convergence_curve(history: HistoryDict,
                           title: str = "Convergence Curve",
                           save_path: Optional[str] = None) -> None:
    """
    Строит график сходимости, показывая изменение f(x) по итерациям.

    Args:
        history: История оптимизации с ключом 'f' – список значений f(x).
        title: Заголовок графика.
        save_path: Если указан, сохраняет график в файл, иначе отображает.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history['f'], marker='o')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("f(x)")
    ax.set_title(title)
    ax.grid(True)
    save_or_show(fig, save_path)


def plot_contour_and_trajectory(history: HistoryDict,
                                f: ScalarFunction,
                                title: str = "Trajectory on Contour",
                                save_path: Optional[str] = None) -> None:
    """
    Строит контурный график функции f и накладывает траекторию оптимизации.

    Args:
        history: История оптимизации с ключами 'x' (список точек) и 'f'.
        f: Целевая функция для построения контуров.
        title: Заголовок графика.
        save_path: Путь для сохранения графика, или None.
    """
    x_hist = np.array(history['x'])
    if x_hist.shape[1] != 2:
        print("⚠️ Trajectory visualization is supported only for 2D functions.")
        return

    # Определяем границы сетки с запасом
    x_min, x_max = x_hist[:, 0].min() - 1, x_hist[:, 0].max() + 1
    y_min, y_max = x_hist[:, 1].min() - 1, x_hist[:, 1].max() + 1
    X, Y, Z = compute_meshgrid_data(f, x_min, x_max, y_min, y_max)

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contour(X, Y, Z, levels=50, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.plot(x_hist[:, 0], x_hist[:, 1], 'r.-', label="Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    save_or_show(fig, save_path)


def plot_full_history(history: HistoryDict,
                      f: Optional[ScalarFunction] = None,
                      title: str = "Optimization History",
                      save_path: Optional[str] = None) -> None:
    """
    Строит комбинированный график: слева — кривую сходимости, справа — контур с траекторией.

    Args:
        history: История оптимизации с ключами 'x' и 'f'.
        f: Целевая функция (если задана, используется для построения контура; для 2D-функций).
        title: Заголовок графика.
        save_path: Путь для сохранения графика, или None.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Сходимость
    axes[0].plot(history['f'], marker='o')
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("f(x)")
    axes[0].set_title(title + " — Convergence")
    axes[0].grid(True)

    # Контур с траекторией
    x_hist = np.array(history['x'])
    if f is not None and x_hist.shape[1] == 2:
        x_min, x_max = x_hist[:, 0].min() - 1, x_hist[:, 0].max() + 1
        y_min, y_max = x_hist[:, 1].min() - 1, x_hist[:, 1].max() + 1
        X, Y, Z = compute_meshgrid_data(f, x_min, x_max, y_min, y_max)
        contour = axes[1].contour(X, Y, Z, levels=50, cmap='viridis')
        axes[1].clabel(contour, inline=True, fontsize=8)
        axes[1].plot(x_hist[:, 0], x_hist[:, 1], 'r.-', label="Trajectory")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title(title + " — Trajectory")
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].text(0.5, 0.5, "2D trajectory not available",
                     horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    save_or_show(fig, save_path)


def plot_all_lr_strategies(actual_iters: int,
                           initial_lr: float = 0.1,
                           step_size: float = 50,
                           alpha: float = 0.5,
                           beta: float = 1.0,
                           lambda_exp: float = 0.05,
                           save_path: Optional[str] = None) -> None:
    """
    Строит график, показывающий изменение learning rate для разных стратегий за заданное число итераций.

    Args:
        actual_iters: Число итераций.
        initial_lr: Начальное значение learning rate.
        step_size: Параметр для piecewise стратегии.
        alpha: Параметр для полиномиальной стратегии.
        beta: Параметр для полиномиальной стратегии.
        lambda_exp: Параметр для экспоненциального затухания.
        save_path: Путь для сохранения графика, или None.
    """

    def update_lr(strategy: str, k: int) -> float:
        if strategy == 'constant':
            return initial_lr
        elif strategy == 'piecewise':
            return initial_lr * (0.5 ** (k // step_size))
        elif strategy == 'exp_decay':
            return initial_lr * np.exp(-lambda_exp * k)
        elif strategy == 'poly_decay':
            return initial_lr / ((beta * k + 1) ** alpha)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    strategies = ['constant', 'piecewise', 'exp_decay', 'poly_decay']
    fig, ax = plt.subplots(figsize=(10, 6))
    for strat in strategies:
        lrs = [update_lr(strat, k) for k in range(actual_iters)]
        ax.plot(range(actual_iters), lrs, label=strat)

    ax.set_title("Learning Rate Decay for Different Strategies")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning Rate")
    ax.legend()
    ax.grid(True)
    save_or_show(fig, save_path)


def plot_surface_and_trajectory_with_path(f: ScalarFunction,
                                          history: HistoryDict,
                                          title: str = "3D Surface with Trajectory",
                                          save_path: Optional[str] = None,
                                          lim: float = 5) -> None:
    """
    Строит 3D-поверхность функции f и накладывает траекторию оптимизации.

    Args:
        f: Целевая функция f(x).
        history: История оптимизации с ключами 'x' и 'f'.
        title: Заголовок графика.
        save_path: Путь для сохранения графика, или None.
        lim: Ограничение для осей X и Y.
    """
    x_hist_full = np.array(history['x'])
    z_hist_full = np.array([f(x) for x in x_hist_full])
    actual_len = len(history['f'])
    x_hist = x_hist_full[:actual_len]
    z_hist = z_hist_full[:actual_len]

    X_vals = np.linspace(-lim, lim, 200)
    Y_vals = np.linspace(-lim, lim, 200)
    X, Y = np.meshgrid(X_vals, Y_vals)
    Z = np.array([[f(np.array([x, y])) for x in X_vals] for y in Y_vals])

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Траектория (выше поверхности, чтобы её видно было)
    ax.plot(x_hist[:, 0], x_hist[:, 1], z_hist, 'r-', linewidth=2, marker='o', markersize=4, label="Trajectory",
            zorder=10)
    ax.scatter(x_hist[-1, 0], x_hist[-1, 1], z_hist[-1], color='black', s=60, label="x*", zorder=11)

    # 3D-поверхность с полупрозрачным наложением
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, antialiased=True, zorder=1)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(Z.min(), min(Z.max(), 100))
    ax.view_init(elev=35, azim=135)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.legend()

    save_or_show(fig, save_path)


def plot_surface_and_trajectory(f: ScalarFunction,
                                history: HistoryDict,
                                title: str = "3D Surface and Trajectory",
                                save_path: Optional[str] = None,
                                lim: float = 100) -> None:
    """
    Строит комбинированный график: слева — 3D-поверхность функции,
    справа — контурный график с наложенной траекторией оптимизации.

    Args:
        f: Целевая функция f(x).
        history: История оптимизации с ключами 'x' и 'f'.
        title: Заголовок комбинированного графика.
        save_path: Путь для сохранения графика, или None.
        lim: Ограничение для осей X и Y в контурном графике.
    """
    x_hist = np.array(history['x'])
    X_vals = np.linspace(-lim, lim, 200)
    Y_vals = np.linspace(-lim, lim, 200)
    X, Y = np.meshgrid(X_vals, Y_vals)
    Z = np.array([[f(np.array([x, y])) for x in X_vals] for y in Y_vals])

    fig = plt.figure(figsize=(14, 6))

    # 3D-поверхность
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title("3D View")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("f(X,Y)")
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)

    # Контурный график с траекторией
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax2)
    ax2.set_title("Contour Plot with Trajectory")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    if x_hist.shape[1] == 2:
        ax2.plot(x_hist[:, 0], x_hist[:, 1], 'r.-', label="Trajectory")
        ax2.plot(x_hist[-1, 0], x_hist[-1, 1], 'ko', label="x*")
        ax2.legend()

    save_or_show(fig, save_path)

# ==============================
# Конец модуля визуализации
# ==============================
