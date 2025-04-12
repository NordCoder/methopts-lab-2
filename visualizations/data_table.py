from typing import List, Dict, Any, Optional

import numpy as np
from tabulate import tabulate

def convert_ndarrays(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Преобразует все np.ndarray в списки для корректной работы tabulate.
    """
    return {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in obj.items()
    }


def render_test_table(
    method_name: str,
    params: List[str],
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    tablefmt: str = "fancy_grid"
) -> None:


    """
    Рисует таблицу с результатами одного метода оптимизации.

    Args:
        method_name: Название метода (например, "Steepest Descent").
        params: Параметры эксперимента (learning rate, стратегия и т.п.)
        results: Список словарей с результатами ({x*, f(x), iters...}).
        save_path: Путь до файла, если нужно сохранить.
        tablefmt: Формат таблицы (fancy_grid, github, pipe, latex и т.д.).
    """

    title_lines = [f"Method: {method_name}"]
    for v in params:
        title_lines.append(v)
    full_title = "\n".join(title_lines)

    processed_results = [convert_ndarrays(r) for r in results]
    table = tabulate(processed_results, headers="keys", tablefmt=tablefmt)

    output = f"{full_title}\n\n{table}"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(output)
