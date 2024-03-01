from model_api.configs import BaseConfig

class TPFN_Config(BaseConfig):
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
            self.hidden_dims = [128, 16, 32]
            self.dropouts = [0.1, 0.2, 0.2]
            self.batch_size = 32
            self.output_dim = 1
            self.time_window = 4
            self.stride = 1
            self.rank = 24
            self.reg_loss_weight = 0.002 # weight of regularization loss.
            self.learning_rate = 0.001
            self.weight_decay = 0.0
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-05
            self.weight_decay_bert = 0.0001
            self.learning_rate_other = 0.0002
            self.weight_decay_other = 0.0005
        elif self.dataset == "MOSEI":
            self.pretrained_bert_model = "/home/zhangbaozheng/paper_code/Robust_Framework/pretrained_model/bert_en"
            self.hidden_dims = [256, 32, 16]
            self.dropouts = [0.1, 0.2, 0.2]
            self.batch_size = 32
            self.output_dim = 1
            self.time_window = 4
            self.stride = 1
            self.rank = 24
            self.reg_loss_weight = 0.002 # weight of regularization loss.
            self.learning_rate = 0.0005
            self.weight_decay = 0.0
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-06
            self.weight_decay_bert = 0.0
            self.learning_rate_other = 0.002
            self.weight_decay_other = 0.0005
        elif self.dataset == "SIMSv2":
            # self.pretrained_bert_model = 'bert-base-chinese'
            self.pretrained_bert_model = '/home/zhangbaozheng/paper_code/Robust_Framework/pretrained_model/bert_cn'
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
            self.reg_loss_weight = 0.002 # weight of regularization loss.
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-06
            self.weight_decay_bert = 0.0
            self.learning_rate_other = 0.002
            self.weight_decay_other = 0.0005
        elif self.dataset == "MIntRec":
            self.pretrained_bert_model = '/home/zhangbaozheng/paper_code/Robust_Framework/pretrained_model/bert_en'
            self.hidden_dims = [256, 32, 16]
            self.dropouts = [0.1, 0.2, 0.2]
            self.batch_size = 32
            self.output_dim = 20
            self.time_window = 4
            self.stride = 1
            self.rank = 24
            self.is_reg = True
            self.norm_decay = 0.002
            self.learning_rate = 0.0005
            self.weight_decay = 0.0
            self.reg_loss_weight = 0.002 # weight of regularization loss.
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-06
            self.weight_decay_bert = 0.0
            self.learning_rate_other = 0.002
            self.weight_decay_other = 0.0005
        else:
            self.logger.warning(f"No default config for dataset {self.dataset}")