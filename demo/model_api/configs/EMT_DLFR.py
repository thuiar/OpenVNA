from model_api.configs import BaseConfig


class EMT_DLFR_Config(BaseConfig):
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
            self.batch_size = 32
            self.learning_rate_bert = 5e-5
            self.learning_rate_audio = 1e-3
            self.learning_rate_video = 1e-4
            self.learning_rate_other = 1e-3
            self.weight_decay_bert = 0.001
            self.weight_decay_audio = 0.01
            self.weight_decay_video = 0.001
            self.weight_decay_other = 0.001

            self.a_lstm_hidden_size = 32
            self.v_lstm_hidden_size = 64
            self.a_lstm_layers = 1
            self.v_lstm_layers = 1
            self.text_out = 768
            self.audio_out = 16
            self.video_out = 32
            self.a_lstm_dropout = 0.0
            self.v_lstm_dropout = 0.0
            self.t_bert_dropout = 0.1

            self.d_model = 128
            self.fusion_layers = 3
            self.heads = 4
            self.learnable_pos_emb = False
            self.emb_dropout = 0.0
            self.attn_dropout = 0.3
            self.ff_dropout = 0.1
            self.ff_expansion = 4
            self.mpu_share = True
            self.modality_share = True
            self.layer_share = True
            self.attn_act_fn = 'tanh'

            self.gmc_tokens_pred_dim = 128
            self.text_pred_dim = 256
            self.audio_pred_dim = 8
            self.video_pred_dim = 16

            self.recon_loss =  'SmoothL1Loss'
            self.recon_loss_wa = 0.001
            self.recon_loss_wv = 0.002
            self.loss_attra_weight = 1 
            self.loss_recon_weight = 1
            self.post_fusion_dim = 256
            self.post_fusion_dropout = 0.3

            self.H = 3.0
            self.output_dim = 1

        elif self.dataset == "MOSEI":
            self.pretrained_bert_model = "pretrained_model/bert_en"
            self.batch_size = 16
            self.learning_rate_bert = 2e-5
            self.learning_rate_audio = 1e-4
            self.learning_rate_video = 1e-4
            self.learning_rate_other = 1e-4
            self.weight_decay_bert = 0.001
            self.weight_decay_audio = 0.001
            self.weight_decay_video = 0.001
            self.weight_decay_other = 0.001
            
            self.a_lstm_hidden_size = 16
            self.v_lstm_hidden_size = 32
            self.a_lstm_layers = 1
            self.v_lstm_layers = 1
            self.text_out = 768
            self.audio_out = 16
            self.video_out = 32
            self.a_lstm_dropout = 0.0
            self.v_lstm_dropout = 0.0
            self.t_bert_dropout = 0.1

            self.d_model = 128
            self.fusion_layers = 2
            self.heads = 4
            self.learnable_pos_emb = False
            self.emb_dropout = 0.0
            self.attn_dropout = 0.0
            self.ff_dropout = 0.0
            self.ff_expansion = 4
            self.mpu_share = True
            self.modality_share = True
            self.layer_share = True
            self.attn_act_fn = 'tanh'

            self.gmc_tokens_pred_dim = 128
            self.text_pred_dim = 256
            self.audio_pred_dim = 8
            self.video_pred_dim = 16

            self.recon_loss =  'SmoothL1Loss'
            self.recon_loss_wa = 0.001
            self.recon_loss_wv = 0.002
            self.loss_attra_weight = 1 
            self.loss_recon_weight = 1
            self.post_fusion_dim = 128
            self.post_fusion_dropout = 0.0

            self.H = 3.0
            self.output_dim = 1

        elif self.dataset == 'SIMSv2':
            # self.pretrained_bert_model = 'bert-base-chinese'
            self.pretrained_bert_model = 'pretrained_model/bert_cn'
            self.batch_size = 32
            self.learning_rate_bert = 2e-5
            self.learning_rate_audio = 1e-3
            self.learning_rate_video = 1e-3
            self.learning_rate_other = 1e-3
            self.weight_decay_bert = 0.001
            self.weight_decay_audio = 0.0
            self.weight_decay_video = 0.0
            self.weight_decay_other = 0.0
            
            self.a_lstm_hidden_size = 16
            self.v_lstm_hidden_size = 32
            self.a_lstm_layers = 1
            self.v_lstm_layers = 1
            self.text_out = 768
            self.audio_out = 16
            self.video_out = 32
            self.a_lstm_dropout = 0.0
            self.v_lstm_dropout = 0.0
            self.t_bert_dropout = 0.1

            self.d_model = 32
            self.fusion_layers = 4
            self.heads = 4
            self.learnable_pos_emb = False
            self.emb_dropout = 0.0
            self.attn_dropout = 0.0
            self.ff_dropout = 0.0
            self.ff_expansion = 4
            self.mpu_share = True
            self.modality_share = True
            self.layer_share = True
            self.attn_act_fn = 'tanh'

            self.gmc_tokens_pred_dim = 128
            self.text_pred_dim = 256
            self.audio_pred_dim = 8
            self.video_pred_dim = 16

            self.recon_loss =  'SmoothL1Loss'
            self.recon_loss_wa = 0.01
            self.recon_loss_wv = 0.02
            self.loss_attra_weight = 0.5
            self.loss_recon_weight = 0.5
            self.post_fusion_dim = 256
            self.post_fusion_dropout = 0.0

            self.H = 1.0
            self.output_dim = 1
        elif self.dataset == "MIntRec":
            self.pretrained_bert_model = "pretrained_model/bert_en"
            self.batch_size = 16
            self.learning_rate_bert = 2e-5
            self.learning_rate_audio = 1e-4
            self.learning_rate_video = 1e-4
            self.learning_rate_other = 1e-4
            self.weight_decay_bert = 0.001
            self.weight_decay_audio = 0.001
            self.weight_decay_video = 0.001
            self.weight_decay_other = 0.001
            
            self.a_lstm_hidden_size = 16
            self.v_lstm_hidden_size = 32
            self.a_lstm_layers = 1
            self.v_lstm_layers = 1
            self.text_out = 768
            self.audio_out = 16
            self.video_out = 32
            self.a_lstm_dropout = 0.0
            self.v_lstm_dropout = 0.0
            self.t_bert_dropout = 0.1

            self.d_model = 128
            self.fusion_layers = 2
            self.heads = 4
            self.learnable_pos_emb = False
            self.emb_dropout = 0.0
            self.attn_dropout = 0.0
            self.ff_dropout = 0.0
            self.ff_expansion = 4
            self.mpu_share = True
            self.modality_share = True
            self.layer_share = True
            self.attn_act_fn = 'tanh'

            self.gmc_tokens_pred_dim = 128
            self.text_pred_dim = 256
            self.audio_pred_dim = 8
            self.video_pred_dim = 16

            self.recon_loss =  'SmoothL1Loss'
            self.recon_loss_wa = 0.001
            self.recon_loss_wv = 0.002
            self.loss_attra_weight = 1 
            self.loss_recon_weight = 1
            self.post_fusion_dim = 128
            self.post_fusion_dropout = 0.0

            self.H = 3.0
            self.output_dim = 20
        else:
            self.logger.warning(f"No default config for dataset {self.dataset}")