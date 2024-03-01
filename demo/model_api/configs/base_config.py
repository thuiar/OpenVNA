import json
import logging
from pathlib import Path
from pprint import pformat
from typing import Any, Literal, List

from typeguard import typechecked
from model_api.util.functions import assign_gpu
ALL_TASKS_LITERAL = Literal["MSA", "MIR"]
ALL_MODELS_LITERAL = Literal["T2FN", "TPFN", "CTFN", "MMIN", "TFRNet", "GCNET", "NIAT", "EMT_DLFR"]
ALL_AUGMENTATIONS_LITERAL = Literal["none", "feat_random_drop", "feat_structural_drop","rawv_impulse_value","rawv_gblur","rawa_color_w","rawa_bg_park"]
ALL_DATASETS_LITERAL = Literal["MOSI", "MOSEI", "SIMSv2","MIntRec"]
DATASET_ROOT_DIR = Path("/home/sharing/disk3/Datasets/MMSA-Standard")
NOISY_DATASET_ROOT_DIR = Path("/home/sharing/disk3/Datasets/MMSA-Noise")

DATASET_TASK_MAP = {
    'MOSI': 'MSA',
}

DATASET_ORI_FEATURE_MAP = {
    'MOSI': 'MOSI/Processed/unaligned_v171_a25_l50.pkl'
}

class BaseConfig(object):
    @typechecked
    def __init__(
        self,
        task: ALL_TASKS_LITERAL = "MSA",                    # task name
        model: ALL_MODELS_LITERAL = "TPFN",                 # model name
        augmentation: List[ALL_AUGMENTATIONS_LITERAL] | None = None,   # augmentation type
        dataset: ALL_DATASETS_LITERAL = "MOSI",             # name of the dataset
        num_workers: int = 4,                               # number of data loading threads
        feature_origin= None,           # feature file for original modality sequences.
        mode= "train",                  # train mode or test mode. test mode is the standardized evaluation process
        test_model_path = None,         # list of paths to the model weights of different seeds for test mode
        eval_noise_type = "feat_random_drop",   # choose how features are missing
        test_missing_seed: List[int] = [0],                 # seed for test mode missing feature construction
        seeds = [0, 1, 2, 3, 4],                # seed list for training
        device: list = [0],                                 # gpu id list. -1 for cpu.
        verbose_level: int = 1,                             # print more information
        model_save_dir = 'results/saved_models',            # directory to save model parameters
        res_save_dir = 'results/results',                   # directory to save csv result files
        log_dir = 'results/logs',                           # logs saved dir
        **unknown_args
    ) -> dict:
        self.args = {}
        self.logger = logging.getLogger("OpenVNA")
        self.task = task
        self.model = model
        self.augmentation = augmentation
        self.dataset = dataset
        self.num_workers = num_workers
        self.feature_origin = feature_origin
        self.mode = mode
        self.test_model_path = test_model_path
        self.eval_noise_type = eval_noise_type
        self.test_missing_seed = test_missing_seed
        self.seq_lens = [50, 1432, 143]
        self.feature_dims = [768, 25, 171]
        self.seeds = seeds
        self.device = assign_gpu(device)
        self.verbose_level = verbose_level
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
    def from_json(self, json_file) -> None:
        """
        Load configs from a json file.
        """
        args = json.load(open(json_file, 'r'))
        self.update(args)

    @typechecked
    def to_json(self, json_file) -> None:
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
    