from model_api.configs import BaseConfig

class CTFN_Config(BaseConfig):
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
            self.trans_dropout = 0.5
            self.gru_units = 512
            self.dense_units = 128
            self.a_d_model = 600
            self.t_d_model = 600
            self.v_d_model = 600
            self.a_heads = 4
            self.t_heads = 4
            self.v_heads = 4
            self.a_num_layer = 3
            self.t_num_layer = 3
            self.v_num_layer = 3
            self.a_dim_forward = 2048
            self.t_dim_forward = 2048
            self.v_dim_forward = 2048
            self.lr = 0.0007
            self.factor = 0.5
            self.trans_lr = 0.003
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.batch_size = 64
            self.output_dim = 1
            
        elif self.dataset == "MOSEI":
            self.pretrained_bert_model = "pretrained_model/bert_en"
            self.trans_dropout = 0.5
            self.gru_units = 512
            self.dense_units = 128
            self.a_d_model = 600
            self.t_d_model = 600
            self.v_d_model = 600
            self.a_heads = 4
            self.t_heads = 4
            self.v_heads = 4
            self.a_num_layer = 3
            self.t_num_layer = 3
            self.v_num_layer = 3
            self.a_dim_forward = 2048
            self.t_dim_forward = 2048
            self.v_dim_forward = 2048
            self.lr = 0.0003
            self.factor = 0.5
            self.trans_lr = 0.003
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.batch_size = 64
            self.output_dim = 1
        elif self.dataset == 'SIMSv2':
            # self.pretrained_bert_model = 'bert-base-chinese'
            self.pretrained_bert_model = 'pretrained_model/bert_cn'
            self.trans_dropout = 0.5
            self.gru_units = 512
            self.dense_units = 128
            self.a_d_model = 600
            self.t_d_model = 600
            self.v_d_model = 600
            self.a_heads = 4
            self.t_heads = 4
            self.v_heads = 4
            self.a_num_layer = 3
            self.t_num_layer = 3
            self.v_num_layer = 3
            self.a_dim_forward = 2048
            self.t_dim_forward = 2048
            self.v_dim_forward = 2048
            self.lr = 0.0003
            self.factor = 0.5
            self.trans_lr = 0.003
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.batch_size = 64
            self.output_dim = 1
        elif self.dataset == "MIntRec":
            self.pretrained_bert_model = "bert-base-uncased"
            self.trans_dropout = 0.5
            self.gru_units = 512
            self.dense_units = 128
            self.a_d_model = 600
            self.t_d_model = 600
            self.v_d_model = 600
            self.a_heads = 4
            self.t_heads = 4
            self.v_heads = 4
            self.a_num_layer = 3
            self.t_num_layer = 3
            self.v_num_layer = 3
            self.a_dim_forward = 2048
            self.t_dim_forward = 2048
            self.v_dim_forward = 2048
            self.lr = 0.0003
            self.factor = 0.5
            self.trans_lr = 0.003
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.batch_size = 64
            self.output_dim = 20
        else:
            self.logger.warning(f"No default config for dataset {self.dataset_name}")