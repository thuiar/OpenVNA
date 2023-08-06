from modal_api.configs import BaseConfig


class TFR_Net_Config(BaseConfig):
    def __init__(self, **kwargs) -> None:
        remaining_args = super().__init__(**kwargs)
        self.set_default_config()
        self.update(remaining_args)
    
    def set_default_config(self):
        # common configs
        self.early_stop = 8
        self.finetune_bert = False
        self.feature_aligned = False     # The model uses aligned features
        self.attn_mask = True
        self.update_epochs = 4
        self.alignmentModule = "crossmodal_attn"
        self.generatorModule = "linear"
        self.fusionModule = "c_gate"
        self.recloss_type = "combine"
        self.without_generator = False
        self.missing_rate = [0.2, 0.2, 0.2]
        self.missing_seed = [1111, 1111, 1111]
        # dataset specific configs
        if self.dataset_name == "MOSI":
            self.feature_dims = [768,25,171]
            self.seq_lens = [50,1432,143]
            self.pretrained_bert_model = "bert-base-uncased"
            self.text_dropout = 0.2
            self.conv1d_kernel_size_l = 1
            self.conv1d_kernel_size_a = 5
            self.conv1d_kernel_size_v = 3
            self.attn_dropout = 0.2
            self.attn_dropout_a = 0.1
            self.attn_dropout_v = 0.0
            self.relu_dropout = 0.2
            self.embed_dropout = 0.2
            self.res_dropout = 0.2
            self.dst_feature_dim_nheads = [30, 6]
            self.nlevels = 3
            self.fusion_t_in = 90
            self.fusion_a_in = 90
            self.fusion_v_in = 90
            self.fusion_t_hid = 36
            self.fusion_a_hid = 20
            self.fusion_v_hid = 48
            self.fusion_gru_layers = 3
            self.use_linear = True
            self.fusion_drop = 0.2
            self.cls_hidden_dim = 128
            self.cls_dropout = 0.0
            self.grad_clip = 0.8
            self.batch_size = 24
            self.learning_rate_bert = 2e-6
            self.learning_rate_other = 0.001
            self.patience = 5
            self.weight_decay_bert = 0.0001
            self.weight_decay_other = 0.001
            self.weight_gen_loss = [5, 2, 20]
            self.weight_sim_loss = 5
            self.num_temporal_head = 1
        elif self.dataset_name == "MOSEI":
            self.pretrained_bert_model = "bert-base-uncased"
            self.text_dropout = 0.2
            self.conv1d_kernel_size_l = 1
            self.conv1d_kernel_size_a = 5
            self.conv1d_kernel_size_v = 3
            self.attn_dropout = 0.2
            self.attn_dropout_a = 0.1
            self.attn_dropout_v = 0.0
            self.relu_dropout = 0.2
            self.embed_dropout = 0.2
            self.res_dropout = 0.2
            self.dst_feature_dim_nheads = [30, 6]
            self.nlevels = 3
            self.fusion_t_in = 90
            self.fusion_a_in = 90
            self.fusion_v_in = 90
            self.fusion_t_hid = 36
            self.fusion_a_hid = 20
            self.fusion_v_hid = 48
            self.fusion_gru_layers = 3
            self.use_linear = True
            self.fusion_drop = 0.2
            self.cls_hidden_dim = 128
            self.cls_dropout = 0.0
            self.grad_clip = 0.8
            self.batch_size = 24
            self.learning_rate_bert = 1e-5
            self.learning_rate_other = 0.002
            self.patience = 5
            self.weight_decay_bert = 0.0001
            self.weight_decay_other = 0.001
            self.weight_gen_loss = [5e-6, 2e-6, 2e-5]
            self.weight_sim_loss = 5
            self.num_temporal_head = 1
        elif self.dataset_name == "SIMS":
            self.pretrained = "bert-base-chinese"
            self.text_dropout = 0.2
            self.conv1d_kernel_size_l = 1
            self.conv1d_kernel_size_a = 5
            self.conv1d_kernel_size_v = 3
            self.attn_dropout = 0.2
            self.attn_dropout_a = 0.1
            self.attn_dropout_v = 0.0
            self.relu_dropout = 0.0
            self.embed_dropout = 0.1
            self.res_dropout = 0.2
            self.dst_feature_dim_nheads = [30, 6]
            self.nlevels = 2
            self.trans_hid_t = 40
            self.trans_hid_t_drop = 0.0
            self.trans_hid_a = 80
            self.trans_hid_a_drop = 0.1
            self.trans_hid_v = 48
            self.trans_hid_v_drop = 0.3
            self.fusion_t_in = 90
            self.fusion_a_in = 90
            self.fusion_v_in = 90
            self.fusion_t_hid = 36
            self.fusion_a_hid = 20
            self.fusion_v_hid = 48
            self.fusion_gru_layers = 3
            self.use_linear = True
            self.fusion_drop = 0.2
            self.cls_hidden_dim = 128
            self.cls_dropout = 0.1
            self.grad_clip = 0.8
            self.batch_size = 16
            self.learning_rate_bert = 1e-5
            self.learning_rate_other = 0.002
            self.patience = 10
            self.weight_decay_bert = 0.0001
            self.weight_decay_other = 0.001
            self.weight_gen_loss = [1, 0.01, 0.0001]
            self.weight_sim_loss = 5
            self.num_temporal_head = 25
        else:
            self.logger.warning(f"No default config for dataset {self.dataset_name}")