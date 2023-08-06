from importlib import import_module

from typeguard import typechecked

from modal_api.configs import BaseConfig
from modal_api.trainers.base_trainer import BaseTrainer


@typechecked
def get_trainer(config: BaseConfig) -> BaseTrainer:
    module = import_module(f'modal_api.trainers.{config.model}')
    trainer = getattr(module, f"{config.model}_Trainer")(config)
    return trainer
