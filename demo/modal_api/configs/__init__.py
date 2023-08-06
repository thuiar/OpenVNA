from importlib import import_module

from typeguard import typechecked

from modal_api.configs.base_config import *


@typechecked
def get_config(model: ALL_MODELS_LITERAL, dataset_name: ALL_DATASETS_LITERAL, **kwargs) -> BaseConfig:
    module = import_module(f'modal_api.configs.{model}')
    config = getattr(module, f"{model}_Config")(model=model, dataset_name=dataset_name, **kwargs)
    return config