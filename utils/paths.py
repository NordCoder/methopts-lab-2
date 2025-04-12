# utils/paths.py
from pathlib import Path
from typing import List
import re

def get_project_root() -> Path:
    """Возвращает корневую директорию проекта (где находится report/)."""
    return Path(__file__).resolve().parents[1]

def sanitize_filename(s: str) -> str:
    """Удаляет недопустимые символы для имени файла."""
    return re.sub(r'[^\w\-_.]', '_', s)

def serialize_params(params: List[str]) -> str:
    """Преобразует параметры в строку, подходящую для имени файла."""
    return "_".join(str(p) for p in params)

def get_report_path(
    method_name: str,
    file_type: str = "plot",
    filename_base: str = "result",
    params: List[str] = None,
    extension: str = ".png"
) -> str:
    """
    Генерирует путь к файлу внутри report/<method>/<file_type> в корне проекта.

    Args:
        method_name: имя метода оптимизации.
        file_type: подкаталог ('plot', 'table' и т.д.).
        filename_base: базовая часть имени файла.
        params: список параметров (включаются в имя).
        extension: расширение (например, .png, .txt).

    Returns:
        Полный путь до файла, создающий папки при необходимости.
    """
    root = get_project_root() / "report" / sanitize_filename(method_name) / sanitize_filename(file_type)
    root.mkdir(parents=True, exist_ok=True)

    if params is not None:
        param_str = sanitize_filename(serialize_params(params))
        filename = f"{filename_base}_{param_str}{extension}"
    else:
        filename = f"{filename_base}{extension}"

    return str(root / filename)
