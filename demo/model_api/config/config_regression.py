from model_api.utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'tpfn': self.__TPFN,
            't2fn': self.__T2FN,
            'tfr_net': self.__TFR_Net,
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.model_name)
        dataset_name = str.lower(args.dataset_name)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]

        dataArgs = dataArgs['unaligned']

        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                                 ))

    def __datasetCommonParams(self):
        tmp = {
            'mosi': {
                'unaligned': {
                    'seq_lens': (50, 1432, 143),
                    # (text, audio, video)
                    'feature_dims': (768, 25, 171),
                    'missing_rate': [0.2, 0.2, 0.2],
                    'train_samples': 1284,
                    "pretrained": 'bert-base-uncased',
                    'KeyEval': 'Loss',
                    "missing_seed": [1111, 1111, 1111]
                }
            }
        }
        return tmp

    def __TPFN(self):
        tmp = {
            'commonParas': {
                'use_bert_finetune': False,
                'need_data_aligned': False,  # 使用对齐数据
                'need_normalized': False,
                'need_model_aligned': False,
                'early_stop': 8,
            },
            # dataset
            'datasetParas': {
                'mosi': {
                    'hidden_dims': (128, 16, 32),
                    'dropouts': (0.1, 0.2, 0.2),
                    'batch_size': 32,
                    'output_dim': 1,
                    'time_window': 4,
                    'stride': 1,
                    'rank': 24,
                    "is_reg": True,
                    'norm_decay': 0.002,
                    'learning_rate': 0.0001,
                    'weight_decay': 0.0,
                    # 当 fine_tune 为 True 时使用。
                    'learning_rate_bert': 2e-05,
                    'learning_rate_other': 0.0005,
                    'weight_decay_bert': 0.0001,
                    'weight_decay_other': 0.0005,
                }
            },
        }
        return tmp

    def __T2FN(self):
        tmp = {
            'commonParas': {
                # use finetune for bert
                'use_bert_finetune': False,
                'need_data_aligned': False,  # 使用对齐数据
                'need_normalized': False,
                'need_model_aligned': False,
                'early_stop': 8
            },
            # dataset
            'datasetParas': {
                'mosi': {
                    'hidden_dims': (64, 8, 4),
                    'text_out': 128,
                    'post_fusion_dim': 128,
                    'dropouts': (0.2, 0.2, 0.2, 0.2),
                    'batch_size': 64,
                    'learning_rate': 0.00075,
                    # 当 fine_tune 为 True 时使用。
                    'learning_rate_bert': 2e-05,
                    'learning_rate_other': 0.0005,
                    'weight_decay_bert': 0.0001,
                    'weight_decay_other': 0.0005,
                }
            },
        }
        return tmp

    def __TFR_Net(self):
        tmp = {
            'commonParas': {
                "data_missing": True,
                "deal_missing": True,
                "need_data_aligned": False,
                "alignmentModule": "crossmodal_attn",
                "generatorModule": "linear",
                "fusionModule": "c_gate",
                "recloss_type": "combine",
                "without_generator": False,
                "early_stop": 6,
                "use_bert": True,
                "use_bert_finetune": True,
                "attn_mask": True,
                "update_epochs": 4
            },
            # dataset
            'datasetParas': {
                'mosi': {
                    "text_dropout": 0.2,
                    "conv1d_kernel_size_l": 1,
                    "conv1d_kernel_size_a": 5,
                    "conv1d_kernel_size_v": 3,
                    "attn_dropout": 0.2,
                    "attn_dropout_a": 0.1,
                    "attn_dropout_v": 0.0,
                    "relu_dropout": 0.2,
                    "embed_dropout": 0.2,
                    "res_dropout": 0.2,
                    "dst_feature_dim_nheads": [30, 6],
                    "nlevels": 3,
                    "fusion_t_in": 90,
                    "fusion_a_in": 90,
                    "fusion_v_in": 90,
                    "fusion_t_hid": 36,
                    "fusion_a_hid": 20,
                    "fusion_v_hid": 48,
                    "fusion_gru_layers": 3,
                    "use_linear": True,
                    "fusion_drop": 0.2,
                    "cls_hidden_dim": 128,
                    "cls_dropout": 0.0,
                    "grad_clip": 0.8,
                    "batch_size": 24,
                    "learning_rate_bert": 1e-05,
                    "learning_rate_other": 0.0005,
                    "patience": 5,
                    "weight_decay_bert": 0.0001,
                    "weight_decay_other": 0.001,
                    "weight_gen_loss": [5e-6, 2e-6, 2e-5],
                    "weight_sim_loss": 5,
                    "num_temporal_head": 1
                },
                
            },
        }
        return tmp
    
    def get_config(self):
        return self.args
