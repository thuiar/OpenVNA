from modal_api.trainers.base_trainer import BaseTrainer

class TFR_Net_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config