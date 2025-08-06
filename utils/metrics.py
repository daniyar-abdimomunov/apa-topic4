import numpy as np
from properscoring import crps_ensemble
from sklearn.metrics import mean_absolute_percentage_error

def calculate_picp(y_true: np.array, y_lower: np.array, y_upper: np.array) -> float:
    if y_lower is None or y_upper is None:
        return np.nan
    if y_true.shape != y_lower.shape or y_true.shape != y_upper.shape:
        raise ValueError(f'all arrays (y_true, y_upper and y_lower) must be the same shape')
    y_true = y_true.flatten()
    y_lower = y_lower.flatten()
    y_upper = y_upper.flatten()
    within_bounds = (y_true >= y_lower) & (y_true <= y_upper)
    picp_score = np.mean(within_bounds)
    return float(picp_score)


def calculate_crps(y_true: np.array, y_pred: np.array) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError(f'all arrays (y_true, y_pred) must be the same shape')
    crps_values = crps_ensemble(y_true, y_pred)
    crps_mean = np.mean(crps_values)
    return float(crps_mean)


def calculate_rmse(y_true: np.array, y_pred: np.array) -> float:
    rmse_score = np.sqrt(np.mean((y_true-y_pred)**2))
    return float(rmse_score)


def calculate_mape(y_true: np.array, y_pred: np.array) -> float:
    nonzero_mask=y_true !=0 
    if not np.any(nonzero_mask):
        return float('inf')
    mape_score=np.mean(np.abs((y_true[nonzero_mask]-y_pred[nonzero_mask])/y_true[nonzero_mask]))
    return float(mape_score)

def _reduce_dimension(array: np.array) -> np.array:
    return array.mean(axis=1) if len(array.shape) > 1 else array