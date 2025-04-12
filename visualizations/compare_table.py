import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Union

def compare_results_to_table(
    func_name: str,
    method_name: str,
    my_result: Tuple[np.ndarray, dict[str, list], int, int],
    scipy_result: Tuple[np.ndarray, dict[str, list], int, int],
    strategy: str,
    x_0: Union[np.ndarray, list],
    hyperparams: Union[list, tuple],
    save_dir: str = "../report/tables",
    test_iteration: int = 0
) -> Tuple[pd.DataFrame, str]:
    """
    Сравнивает результаты двух реализаций (твоя и scipy) и сохраняет CSV-таблицу
    с результатами и параметрами эксперимента.

    :param test_iteration: test iteration number for generating filename
    :param func_name: Название исследуемой функции
    :param method_name: Название метода
    :param my_result: (x*, history: {'x': [...], 'f': [...]}, итерации)
    :param scipy_result: то же самое для scipy
    :param strategy: стратегия learning rate
    :param x_0: начальная точка
    :param hyperparams: гиперпараметры метода
    :param save_dir: папка, куда сохранить CSV
    :return: (таблица, путь к файлу)
    """
    os.makedirs(save_dir, exist_ok=True)

    def summarize(label: str, final_x, history: dict, f_calls: int, grad_calls) -> Dict:
        x_star = np.round(final_x, 5)
        return {
            "Источник": label,
            "Вызовы f(x)": f_calls,
            "Вызовы grad f(x)": grad_calls,
            "x*": f"[{x_star[0]}, {x_star[1]}]",
            "||x*||": round(np.linalg.norm(final_x), 5),
        }

    my_data = summarize("Моя реализация", *my_result)
    scipy_data = summarize("Scipy", *scipy_result)
    df_main = pd.DataFrame([my_data, scipy_data])
    # Параметры эксперимента (метаинформация)
    df_params = pd.DataFrame({
        "Параметр": ["Функция", "Метод", "Стратегия", "Начальная точка", "Гиперпараметры"],
        "Значение": [
            func_name,
            method_name,
            strategy,
            str(np.round(np.array(hyperparams), 4).tolist()),
            str(np.round(np.array([x_0['left'], x_0['right'], x_0['tol']]), 4).tolist())
        ]
    })

    # Собираем итоговую таблицу
    empty_row = pd.DataFrame([["", ""]], columns=df_params.columns)
    df_full = pd.concat([df_params, empty_row, df_main], ignore_index=True)

    # Генерируем имя файла
    filename = f"{func_name}_{method_name}_{test_iteration}_comparison.csv".lower().replace(" ", "_")
    filepath = os.path.join(save_dir, filename)
    df_full.to_csv(filepath, index=False)

    return df_full, filepath
