from modal_api.configs import BaseConfig


class TPFN_Config(BaseConfig):
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
            
            self.hidden_dims = [128, 16, 32]
            self.dropouts = [0.1, 0.2, 0.2]
            self.batch_size = 32
            self.output_dim = 1
            self.time_window = 4
            self.stride = 1
            self.rank = 24
            self.is_reg = True
            self.norm_decay = 0.002
            self.learning_rate = 0.002
            self.weight_decay = 0.0
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-05
            self.weight_decay_bert = 0.0001
            self.learning_rate_other = 0.0002
            self.weight_decay_other = 0.0005
        elif self.dataset_name == "MOSEI":
            self.hidden_dims = [256, 32, 16]
            self.dropouts = [0.1, 0.2, 0.2]
            self.batch_size = 32
            self.output_dim = 1
            self.time_window = 4
            self.stride = 1
            self.rank = 24
            self.is_reg = True
            self.norm_decay = 0.002
            self.learning_rate = 0.0005
            self.weight_decay = 0.0
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-06
            self.weight_decay_bert = 0.0
            self.learning_rate_other = 0.002
            self.weight_decay_other = 0.0005
        elif self.dataset_name == "SIMSv2": #aligned
            self.hidden_dims = [256, 32, 16]
            self.dropouts = [0.1, 0.2, 0.2]
            self.batch_size = 32
            self.output_dim = 1
            self.time_window = 4
            self.stride = 1
            self.rank = 24
            self.is_reg = True
            self.norm_decay = 0.002
            self.learning_rate = 0.0005
            self.weight_decay = 0.0
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-06
            self.weight_decay_bert = 0.0
            self.learning_rate_other = 0.002
            self.weight_decay_other = 0.0005
        else:
            self.logger.warning(f"No default config for dataset {self.dataset_name}")