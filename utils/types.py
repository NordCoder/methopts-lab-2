from typing import Dict, List, Callable

import numpy as np

HistoryDict = Dict[str, List]

ScalarFunction = Callable[[np.ndarray], float]
GradientFunction = Callable[[np.ndarray], np.ndarray]
HessianFunction = Callable[[np.ndarray], np.ndarray]

InitialPoint = np.ndarray