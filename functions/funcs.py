import numpy as np


# ==== КВАДРАТИЧНЫЕ ФОРМЫ С РАЗНЫМ ЧИСЛОМ ОБУСЛОВЛЕННОСТЕЙ ====

def quadratic_cond_1(x: np.ndarray) -> float:
    """Квадратичная функция с числом обусловленности 1: f(x, y) = x^2 + y^2"""
    return x[0] ** 2 + x[1] ** 2


def quadratic_cond_10(x: np.ndarray) -> float:
    """Квадратичная функция с числом обусловленности 10: f(x, y) = 10x^2 + y^2"""
    return 10 * x[0] ** 2 + x[1] ** 2


def quadratic_cond_100(x: np.ndarray) -> float:
    """Квадратичная функция с числом обусловленности 100: f(x, y) = 100x^2 + y^2"""
    return 100 * x[0] ** 2 + x[1] ** 2


def quadratic_cond_1000(x: np.ndarray) -> float:
    """Квадратичная функция с числом обусловленности 1000: f(x, y) = 1000x^2 + y^2"""
    return 1000 * x[0] ** 2 + x[1] ** 2


QUADRATIC_LIST = [quadratic_cond_1, quadratic_cond_10, quadratic_cond_100, quadratic_cond_1000]


def grad_quadratic_cond_1(x: np.ndarray) -> np.ndarray:
    """Градиент функции f(x, y) = x^2 + y^2"""
    return np.array([2 * x[0], 2 * x[1]])

def grad_quadratic_cond_10(x: np.ndarray) -> np.ndarray:
    """Градиент функции f(x, y) = 10x^2 + y^2"""
    return np.array([20 * x[0], 2 * x[1]])

def grad_quadratic_cond_100(x: np.ndarray) -> np.ndarray:
    """Градиент функции f(x, y) = 100x^2 + y^2"""
    return np.array([200 * x[0], 2 * x[1]])

def grad_quadratic_cond_1000(x: np.ndarray) -> np.ndarray:
    """Градиент функции f(x, y) = 1000x^2 + y^2"""
    return np.array([2000 * x[0], 2 * x[1]])


GRAD_QUADRATIC_LIST = [grad_quadratic_cond_1, grad_quadratic_cond_10, grad_quadratic_cond_100, grad_quadratic_cond_1000]

def hess_quadratic_cond_1(x: np.ndarray) -> np.ndarray:
    """Гессиан функции f(x, y) = x^2 + y^2"""
    return np.array([[2, 0],
                     [0, 2]])

def hess_quadratic_cond_10(x: np.ndarray) -> np.ndarray:
    """Гессиан функции f(x, y) = 10x^2 + y^2"""
    return np.array([[20, 0],
                     [0, 2]])

def hess_quadratic_cond_100(x: np.ndarray) -> np.ndarray:
    """Гессиан функции f(x, y) = 100x^2 + y^2"""
    return np.array([[200, 0],
                     [0, 2]])

def hess_quadratic_cond_1000(x: np.ndarray) -> np.ndarray:
    """Гессиан функции f(x, y) = 1000x^2 + y^2"""
    return np.array([[2000, 0],
                     [0, 2]])

HESS_QUADRATIC_LIST = [
    hess_quadratic_cond_1,
    hess_quadratic_cond_10,
    hess_quadratic_cond_100,
    hess_quadratic_cond_1000,
]


# ==== БАЗОВЫЕ ФУНКЦИИ ====

def quadratic_function(x: np.ndarray) -> float:
    """Классическая квадратичная функция: f(x, y) = (x - 3)^2 + (y - 2)^2"""
    return (x[0] - 3) ** 2 + (x[1] - 2) ** 2


def quadratic_grad(x: np.ndarray) -> np.ndarray:
    """Градиент квадратичной функции"""
    return np.array([2 * (x[0] - 3), 2 * (x[1] - 2)])


# ==== СРЕДНИЕ ПО СЛОЖНОСТИ ФУНКЦИИ ====

def rosenbrock_function(x: np.ndarray) -> float:
    """Функция Розенброка: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2"""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Градиент функции Розенброка"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dy = 200 * (x[1] - x[0] ** 2)
    return np.array([dx, dy])


# ==== ПРОДВИНУТЫЕ ФУНКЦИИ ====

def himmelblau_function(x: np.ndarray) -> float:
    """Функция Химмельблау: f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2"""
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def himmelblau_grad(x: np.ndarray) -> np.ndarray:
    """Градиент функции Химмельблау"""
    dx = 4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7)
    dy = 2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7)
    return np.array([dx, dy])


def three_hump_camel_function(x: np.ndarray) -> float:
    """Функция «трёхгорбый верблюд»: f(x, y) = 2x^2 - 1.05x^4 + x^6/6 + xy + y^2"""
    return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + (x[0] ** 6) / 6 + x[0] * x[1] + x[1] ** 2


def three_hump_camel_grad(x: np.ndarray) -> np.ndarray:
    """Градиент функции трёхгорбого верблюда"""
    dx = 4 * x[0] - 4.2 * x[0] ** 3 + x[0] ** 5 + x[1]
    dy = x[0] + 2 * x[1]
    return np.array([dx, dy])

def noisy_quadratic_function(x: np.ndarray, sigma: float = 0.1) -> float:
    """
    Классическая квадратичная функция с аддитивным гауссовым шумом.
    """
    true_value = (x[0] - 3)**2 + (x[1] - 2)**2
    noise = np.random.normal(0, sigma)
    return true_value + noise

def noisy_quadratic_grad(x: np.ndarray) -> np.ndarray:
    """
    Градиент без шума — считаем, что градиент точный.
    """
    return np.array([2 * (x[0] - 3), 2 * (x[1] - 2)])

def sincos_landscape(x: np.ndarray) -> float:
    return np.sin(x[0]) * np.cos(x[1]) + 0.1 * (x[0]**2 + x[1]**2)

def grad_sincos_landscape(x: np.ndarray) -> np.ndarray:
    """
    Градиент функции f(x, y) = sin(x) * cos(y) + 0.1 * (x² + y²)
    """
    df_dx = np.cos(x[0]) * np.cos(x[1]) + 0.2 * x[0]
    df_dy = -np.sin(x[0]) * np.sin(x[1]) + 0.2 * x[1]
    return np.array([df_dx, df_dy])