import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import sys


def robustness(y: list | np.ndarray, x: list | np.ndarray = None, dx: float = 0.1) -> np.float32:
    """
    Compute the absolute and effective robustness of a model given accuracy values under various level of 
    data imperfection, which is the area under the curve of the accuracy-imperfection curve.

    Args:
        y: Accuracy values under various level of data imperfection. Must be a 1-D array.
        x: Data imperfection levels, range from 0 to 1. If None, the sample points are assumed to be evenly 
            spaced along the x-axis. Default is None.
        dx: The spacing between sample points if x is None. Default is 0.1.
    
    Return:
        A float value of absolute robustness.
    """
    abs_robustness = np.trapz(y, x=x, dx=dx)
    eff_robustness = y[0] - abs_robustness
    return abs_robustness, eff_robustness


def cal_conv_metrics(
    y_pred, y_true,
    metrics=["has0", "non0", "acc5", "acc7", "mae", "corr"],
    fixed=4
):
    metrics_func = [getattr(sys.modules[__name__], m) for m in metrics]
    y_pred = y_pred.view(-1).cpu().detach().numpy()
    y_true = y_true.view(-1).cpu().detach().numpy()
    res = {}
    for m, f in zip(metrics, metrics_func):
        if m in ["has0", "non0"]:
            res[f"{m}_acc2"], res[f"{m}_f1"] = f(y_pred, y_true)
            res[f"{m}_acc2"] = round(res[f"{m}_acc2"], fixed)
            res[f"{m}_f1"] = round(res[f"{m}_f1"], fixed)
        else:
            res[m] = round(f(y_pred, y_true), fixed)
    return res


def has0(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    binary_truth = (y_true >= 0)
    binary_preds = (y_pred >= 0)
    acc2 = accuracy_score(binary_preds, binary_truth)
    f1 = f1_score(binary_truth, binary_preds, average='weighted')
    return acc2, f1


def non0(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    non_zero = np.array([i for i, e in enumerate(y_true) if e != 0])
    non_zero_binary_truth = (y_true[non_zero] > 0)
    non_zero_binary_preds = (y_pred[non_zero] > 0)
    non_zero_acc2 = accuracy_score(non_zero_binary_preds, non_zero_binary_truth)
    non_zero_f1 = f1_score(non_zero_binary_truth, non_zero_binary_preds, average='weighted')
    return non_zero_acc2, non_zero_f1


def acc3(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    test_preds_a3 = np.clip(y_pred, a_min=-1., a_max=1.)
    test_truth_a3 = np.clip(y_true, a_min=-1., a_max=1.)
    mult_acc3 = np.sum(np.round(test_preds_a3) == np.round(test_truth_a3)) / float(len(test_truth_a3))
    return mult_acc3

def acc5(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    test_preds_a5 = np.clip(y_pred, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(y_true, a_min=-2., a_max=2.)
    mult_acc5 = np.sum(np.round(test_preds_a5) == np.round(test_truth_a5)) / float(len(test_truth_a5))
    return mult_acc5


def acc7(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    test_preds_a7 = np.clip(y_pred, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_true, a_min=-3., a_max=3.)
    mult_acc7 = np.sum(np.round(test_preds_a7) == np.round(test_truth_a7)) / float(len(test_truth_a7))
    return mult_acc7


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_pred - y_true)).astype(np.float64)


def corr(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.corrcoef(y_pred, y_true)[0][1]

