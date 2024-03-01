from model_api.configs import BaseConfig


class GCNET_Config(BaseConfig):
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
            self.batch_size = 64
            self.dropout = 0.5
            self.hidden = 200
            self.l2 = 0.00001
            self.loss_recon = True
            self.lower_bound = False
            self.learning_rate = 0.0005
            self.output_dim = 1
            self.n_speakers = 1
            self.time_attn = False
            self.windowp = 2
            self.windowf = 2
            self.a_rec_weight = 1e-5
            self.t_rec_weight = 10
            self.v_rec_weight = 1e-4
        elif self.dataset == "MOSEI":
            self.pretrained_bert_model = "pretrained_model/bert_en"
            self.batch_size = 32
            self.dropout = 0.5
            self.hidden = 200
            self.l2 = 0.00001
            self.loss_recon = True
            self.lower_bound = False
            self.learning_rate = 0.001
            self.output_dim = 1
            self.n_speakers = 1
            self.time_attn = False
            self.windowp = 2
            self.windowf = 2
            self.a_rec_weight = 1e-5
            self.t_rec_weight = 10
            self.v_rec_weight = 1e-4
        elif self.dataset == "SIMSv2":
            # self.pretrained_bert_model = 'bert-base-chinese'
            self.pretrained_bert_model = 'pretrained_model/bert_cn'
            self.batch_size = 32
            self.dropout = 0.5
            self.hidden = 200
            self.l2 = 0.00001
            self.loss_recon = True
            self.lower_bound = False
            self.learning_rate = 0.001
            self.output_dim = 1
            self.n_speakers = 1
            self.time_attn = False
            self.windowp = 2
            self.windowf = 2
            self.a_rec_weight = 1e-5
            self.t_rec_weight = 10
            self.v_rec_weight = 1e-4

        elif self.dataset == "MIntRec":
            self.pretrained_bert_model = "pretrained_model/bert_en"
            self.batch_size = 32
            self.dropout = 0.5
            self.hidden = 200
            self.l2 = 0.00001
            self.loss_recon = True
            self.lower_bound = False
            self.learning_rate = 0.001
            self.output_dim = 20
            self.n_speakers = 1
            self.time_attn = False
            self.windowp = 2
            self.windowf = 2
            self.a_rec_weight = 1e-5
            self.t_rec_weight = 10
            self.v_rec_weight = 1e-4
        else:
            self.logger.warning(f"No default config for dataset {self.dataset}")