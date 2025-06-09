import numpy as np
from properscoring import crps_ensemble

def calculate_picp(y_true: np.array, y_lower: np.array, y_upper: np.array) -> float:
    if len(y_true) != len(y_lower) or len(y_true) != len(y_upper):
        raise ValueError(f'all values (y_true, y_upper and y_lower) must be the same length')
    y_lower = _reduce_dimension(y_lower)
    y_upper = _reduce_dimension(y_upper)
    within_bounds = (y_true >= y_lower) & (y_true <= y_upper)
    picp_score = np.mean(within_bounds)
    return float(picp_score)


def calculate_crps(y_true: np.array, y_pred: np.array) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(f'all values (y_true, y_pred) must be the same length')
    crps_values = crps_ensemble(y_true, y_pred)
    crps_mean = np.mean(crps_values)
    return float(crps_mean)


def calculate_rmse(y_true: np.array, y_pred: np.array) -> float:
    # code for calculation
    y_pred = _reduce_dimension(y_pred)
    rmse_score = 0  ## code for calculation
    return float(rmse_score)


def calculate_mape(y_true: np.array, y_pred: np.array) -> float:
    ## code for calculation
    y_pred = _reduce_dimension(y_pred)
    mape_score = 0  ## code for calculation
    return float(mape_score)

def _reduce_dimension(array: np.array) -> np.array:
    return array.mean(axis=1) if len(array.shape) > 1 else array