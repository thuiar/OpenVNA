import json
import logging
from pathlib import Path
from pprint import pformat
from typing import Any, Literal

import torch
import numpy as np
from typeguard import typechecked

ALL_MODELS_LITERAL = Literal[
    "TPFN", "T2FN", "TFR_Net"
]

ALL_DATASETS_LITERAL = Literal["MOSI"]
DATASET_ROOT_DIR = Path("/home/sharing/disk3/Datasets/MMSA-Standard")

class BaseConfig(object):
    @typechecked
    def __init__(
        self,
        model: ALL_MODELS_LITERAL = "TPFN",                 # model name
        dataset_name: ALL_DATASETS_LITERAL = "MOSI",        # name of the dataset
        num_workers: int = 0,                               # number of data loading threads
        feature_M: Path | str | None = None,                # feature file for all modalities
        feature_T: Path | str | None = None,                # override feature for Text modality
        feature_A: Path | str | None = None,                # override feature for Audio modality
        feature_V: Path | str | None = None,                # override feature for Video modality
        test_model_path: list[Path | str] | None = None,    # list of paths to the model weights of different seeds for robust test
        test_missing_rates: list[float] | np.ndarray = np.arange(0, 1.1, 0.1),  # missing rate list for robust test missing feature construction
        test_missing_mode: Literal["random_drop_aligned", "random_drop_unaligned", "temporal_drop"] = "random_drop_unaligned",   # choose how features are missing
        test_missing_seed: int = 0,                         # seed for robust test missing feature construction
        seeds: list[int] = [0, 1, 2, 3, 4],                 # seed list for training
        device: int = 0,                                    # gpu id. -1 for cpu.
        verbose: bool = False,                              # print more information
        model_save_dir: Path | str = 'checkpoints',         # directory to save model parameters
        res_save_dir: Path | str = 'results',               # directory to save csv result files
        log_dir: Path | str = 'logs',                       # logs saved dir
        **unknown_args
    ) -> dict:
        self.args = {}
        self.logger = logging.getLogger("Robust-MSA")
        self.model = model
        self.dataset_name = dataset_name
        self.num_workers = num_workers
        self.feature_M = feature_M
        self.feature_T = feature_T
        self.feature_A = feature_A
        self.feature_V = feature_V
        self.test_model_path = test_model_path
        self.test_missing_rates = test_missing_rates
        self.test_missing_mode = test_missing_mode
        self.test_missing_seed = test_missing_seed
        self.seeds = seeds
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu")
        self.verbose = verbose
        self.model_save_dir = model_save_dir
        self.res_save_dir = res_save_dir
        self.log_dir = log_dir
        return unknown_args

    def __repr__(self):
        return pformat(self.args, indent=4, sort_dicts=False) # print with insertion order

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setattr__(__name, __value)
        if __name not in ['args', 'logger']:
            self.args[__name] = __value
    
    @typechecked
    def from_json(self, json_file: Path | str) -> None:
        """
        Load configs from a json file.
        """
        args = json.load(open(json_file, 'r'))
        self.update(args)

    @typechecked
    def to_json(self, json_file: Path | str) -> None:
        """
        Save configs to a json file.
        """
        json.dump(self.args, open(json_file, 'w'), indent=4)
    
    @typechecked
    def update(self, args: dict) -> None:
        for k, v in args.items():
            setattr(self, k, v)
    
    def set_default_config(self) -> None:
        """
        Set default model specific configs in subclasses.
        """
        raise NotImplementedError()
    