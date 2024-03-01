from importlib import import_module

from typeguard import typechecked

from model_api.configs.base_config import *

@typechecked
def get_model(config: BaseConfig, **kwargs):
    module = import_module(f'model_api.models.{config.model}')
    model = getattr(module, f"{config.model}")(config=config, **kwargs)
    return model