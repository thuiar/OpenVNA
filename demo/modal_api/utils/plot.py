from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_missing_rate_non0_acc2(
    result_dir: str,
    model_name: list[str],
    dataset_name: str = "MOSI",
    save_dir: str = None,
    save_name: str = None,
    show: bool = False,
):
    results = []
    for model in model_name:
        result_path = Path(result_dir) / f"{model}-{dataset_name}.csv"
        result = pd.read_csv(result_path)
        result = result[result["missing_modality"] == "M"]
        result = result.drop(columns=["missing_modality", "seed"])
        result = result.groupby(["missing_rate"]).mean()
        results.append(result)