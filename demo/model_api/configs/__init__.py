from importlib import import_module

from typeguard import typechecked

from model_api.configs.base_config import *


@typechecked
def get_config(model: ALL_MODELS_LITERAL, dataset: ALL_DATASETS_LITERAL, **kwargs) -> BaseConfig:
    module = import_module(f'model_api.configs.{model}')
    config = getattr(module, f"{model}_Config")(model=model, dataset=dataset, **kwargs)
    return config