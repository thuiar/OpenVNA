from modal_api.configs import BaseConfig


class T2FN_Config(BaseConfig):
    def __init__(self, **kwargs) -> None:
        remaining_args = super().__init__(**kwargs)
        self.set_default_config()
        self.update(remaining_args)
    
    def set_default_config(self):
        # common configs
        self.early_stop = 8
        self.pretrained_bert_model = "bert-base-uncased"
        self.finetune_bert = False
        self.feature_aligned = True     # The model uses aligned features
        # dataset specific configs
        if self.dataset_name == "MOSI":
            self.feature_dims = [768,25,171]
            self.seq_lens = [50,1432,143]
            self.hidden_dims = [64, 8, 4]
            self.text_out = 128
            self.post_fusion_dim = 128
            self.dropouts = [0.2, 0.2, 0.2, 0.2]
            self.batch_size = 32
            self.learning_rate = 0.00075
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-05
            self.weight_decay_bert = 0.0001
            self.learning_rate_other = 0.0005
            self.weight_decay_other = 0.0005
        elif self.dataset_name == "MOSEI":
            self.hidden_dims = [128, 16, 128]
            self.text_out = 64
            self.post_fusion_dim = 32
            self.dropouts = [0.3, 0.3, 0.3, 0.5]
            self.batch_size = 64
            self.learning_rate = 1e-3
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-06
            self.weight_decay_bert = 0.0
            self.learning_rate_other = 0.002
            self.weight_decay_other = 0.0005
        else:
            self.logger.warning(f"No default config for dataset {self.dataset_name}")