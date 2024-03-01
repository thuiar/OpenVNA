from model_api.configs import BaseConfig


class NIAT_Config(BaseConfig):
    def __init__(self, **kwargs) -> None:
        remaining_args = super().__init__(**kwargs)
        self.set_default_config()
        self.update(remaining_args)

    def set_default_config(self):
        # common configs
        self.early_stop = 6
        self.finetune_bert = True
        self.coupled_instance = True   # Whether paired (clean, noisy) instances are provided in the training.
        # dataset specific configs
        if self.dataset == "MOSI":
            self.pretrained_bert_model = "/home/zhangbaozheng/paper_code/Robust_Framework/pretrained_model/bert_en"
            self.fus_d_l, self.fus_d_a, self.fus_d_v = 96, 24, 40
            self.fus_conv1d_kernel_l = 3
            self.fus_conv1d_kernel_a = 3
            self.fus_conv1d_kernel_v = 9
            self.fus_nheads = 8
            self.fus_layers = 3
            self.fus_attn_mask = True
            self.fus_position_embedding = False
            self.fus_relu_dropout = 0.0
            self.fus_embed_dropout = 0.5
            self.fus_res_dropout = 0.4
            self.fus_attn_dropout = 0.5
            self.rec_hidden_dim1 = 80
            self.rec_dropout = 0.4
            self.rec_hidden_dim2 = 96
            self.disc_hidden_dim1 = 128
            self.disc_hidden_dim2 = 64
            self.clf_dropout = 0.3
            self.clf_hidden_dim = 80
            self.alpha = 0.6
            self.batch_size = 32
            self.beta = 1.0
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-05
            self.weight_decay_bert = 0.0005
            self.learning_rate_other = 0.0001
            self.weight_decay_other = 0.0005
            self.grad_clip = 0.6

        elif self.dataset == "MOSEI":
            self.pretrained_bert_model = "pretrained_model/bert_en"
            self.fus_d_l, self.fus_d_a, self.fus_d_v = 96, 16, 32
            self.fus_conv1d_kernel_l = 3
            self.fus_conv1d_kernel_a = 5
            self.fus_conv1d_kernel_v = 3
            self.fus_nheads = 4
            self.fus_layers = 3
            self.fus_attn_mask = True
            self.fus_position_embedding = False
            self.fus_relu_dropout = 0.5
            self.fus_embed_dropout = 0.0
            self.fus_res_dropout = 0.5
            self.fus_attn_dropout = 0.1
            self.rec_hidden_dim1 = 128
            self.rec_dropout = 0.2
            self.rec_hidden_dim2 = 64
            self.disc_hidden_dim1 = 80
            self.disc_hidden_dim2 = 32
            self.clf_dropout = 0.2
            self.clf_hidden_dim = 256
            self.alpha = 0.6
            self.batch_size = 32
            self.beta = 1.0
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-06
            self.weight_decay_bert = 0.002
            self.learning_rate_other = 0.0
            self.weight_decay_other = 0.0005
            self.grad_clip = 0.6
        elif self.dataset == 'SIMSv2': 
            # NOTE  just a copy and paste of MOSEI Hyperparameters w.o. tuning.
            # self.pretrained_bert_model = 'bert-base-chinese'
            self.pretrained_bert_model = 'pretrained_model/bert_cn'
            self.fus_d_l, self.fus_d_a, self.fus_d_v = 96, 16, 32
            self.fus_conv1d_kernel_l = 3
            self.fus_conv1d_kernel_a = 5
            self.fus_conv1d_kernel_v = 3
            self.fus_nheads = 4
            self.fus_layers = 3
            self.fus_attn_mask = True
            self.fus_position_embedding = False
            self.fus_relu_dropout = 0.5
            self.fus_embed_dropout = 0.0
            self.fus_res_dropout = 0.5
            self.fus_attn_dropout = 0.1
            self.rec_hidden_dim1 = 128
            self.rec_dropout = 0.2
            self.rec_hidden_dim2 = 64
            self.disc_hidden_dim1 = 80
            self.disc_hidden_dim2 = 32
            self.clf_dropout = 0.2
            self.clf_hidden_dim = 256
            self.alpha = 0.6
            self.batch_size = 32
            self.beta = 1.0
            # below is used when finetune_bert is True
            self.learning_rate_bert = 2e-06
            self.weight_decay_bert = 0.002
            self.learning_rate_other = 0.0
            self.weight_decay_other = 0.0005
            self.grad_clip = 0.6
        else:
            self.logger.warning(f"No default config for dataset {self.dataset}")