from model_api.configs import BaseConfig


class T2FN_Config(BaseConfig):
    def __init__(self, **kwargs) -> None:
        remaining_args = super().__init__(**kwargs)
        self.set_default_config()
        self.update(remaining_args)
    
    def set_default_config(self):
        # common configs
        self.early_stop = 6
        
        self.finetune_bert = False
        self.coupled_instance = False   # Whether paired (clean, noisy) instances are provided in the training.
        # dataset specific configs
        if self.dataset == "MOSI":
            self.pretrained_bert_model = "/home/zhangbaozheng/paper_code/Robust_Framework/pretrained_model/bert_en"
            self.hidden_dims = [63, 15, 7]
            self.text_out = 128
            self.post_fusion_dim = 128
            self.dropouts = [0.1, 0.1, 0.1, 0.1]
            self.batch_size = 32
            self.learning_rate = 0.0007
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-05
            self.weight_decay_bert = 0.0001
            self.learning_rate_other = 0.0005
            self.weight_decay_other = 0.0005
            # Weight for Regularization Loss
            self.reg_loss_weight = 0.001
            self.output_dim = 1
        elif self.dataset == "MOSEI":
            self.pretrained_bert_model = "pretrained_model/bert_en"
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
            # Weight for Regularization Loss
            self.reg_loss_weight = 0.001
            self.output_dim = 1
        elif self.dataset == "SIMSv2":
            # self.pretrained_bert_model = 'bert-base-chinese'
            self.pretrained_bert_model = 'pretrained_model/bert_cn'
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
            # Weight for Regularization Loss
            self.reg_loss_weight = 0.001
            self.output_dim = 1
        else:
            self.logger.warning(f"No default config for dataset {self.dataset}")