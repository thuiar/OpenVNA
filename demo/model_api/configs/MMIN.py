from model_api.configs import BaseConfig


class MMIN_Config(BaseConfig):
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
        if self.dataset == 'MOSI':
            self.pretrained_bert_model = '/home/zhangbaozheng/paper_code/Robust_Framework/pretrained_model/bert_en'
            
            self.embd_size_a = 64
            self.embd_size_t = 64
            self.embd_size_v = 64
            self.batch_size = 64
            self.embd_method_a = 'maxpool'
            self.embd_method_v = 'maxpool'
            self.output_dim = 1
            # Phase one Settings.
            self.module_names_p1 = ['A', 'V', 'L', 'C1']
            self.cls_layers_p1 = '64,64'
            self.dropout_rate_p1 = 0.3
            self.niter, self.niter_decay = 8, 8
            
            # Phase two Settings.
            self.module_names_p2 = ['A', 'V', 'L', 'C2', 'AE', 'AE_cycle']
            self.ae_layers = '128,64,32'
            self.n_blocks = 3
            self.cls_layers_p2 = '64,64'
            self.dropout_rate_p2 = 0.1
            self.ce_weight = 2.0
            self.mse_weight = 1.0
            self.cycle_weight = 1.0

            self.batch_size = 128
            self.lr = 0.0001
            self.beta1 = 0.5

        elif self.dataset == 'MOSEI':
            self.pretrained_bert_model = 'pretrained_model/bert_en'
            
            self.embd_size_a = 64
            self.embd_size_t = 64
            self.embd_size_v = 64
            self.batch_size = 64
            self.embd_method_a = 'maxpool'
            self.embd_method_v = 'maxpool'
            self.output_dim = 1
            # Phase one Settings.
            self.module_names_p1 = ['A', 'V', 'L', 'C1']
            self.cls_layers_p1 = '64,64'
            self.dropout_rate_p1 = 0.3
            self.niter, self.niter_decay = 10, 10
            
            # Phase two Settings.
            self.module_names_p2 = ['A', 'V', 'L', 'C2', 'AE', 'AE_cycle']
            self.ae_layers = '128,64,32'
            self.n_blocks = 3
            self.cls_layers_p2 = '64,64'
            self.dropout_rate_p2 = 0.1
            self.ce_weight = 1.0
            self.mse_weight = 1.0
            self.cycle_weight = 1.0

            self.batch_size = 64
            self.lr = 0.002
            self.beta1 = 0.5
        elif self.dataset == 'SIMSv2':
            # self.pretrained_bert_model = 'bert-base-chinese'
            self.pretrained_bert_model = 'pretrained_model/bert_cn'
            
            self.embd_size_a = 64
            self.embd_size_t = 64
            self.embd_size_v = 64
            self.batch_size = 64
            self.embd_method_a = 'maxpool'
            self.embd_method_v = 'maxpool'
            self.output_dim = 1
            # Phase one Settings.
            self.module_names_p1 = ['A', 'V', 'L', 'C1']
            self.cls_layers_p1 = '64,64'
            self.dropout_rate_p1 = 0.3
            self.niter, self.niter_decay = 10, 10
            
            # Phase two Settings.
            self.module_names_p2 = ['A', 'V', 'L', 'C2', 'AE', 'AE_cycle']
            self.ae_layers = '128,64,32'
            self.n_blocks = 3
            self.cls_layers_p2 = '64,64'
            self.dropout_rate_p2 = 0.1
            self.ce_weight = 1.0
            self.mse_weight = 1.0
            self.cycle_weight = 1.0

            self.batch_size = 64
            self.lr = 0.002
            self.beta1 = 0.5

        elif self.dataset == 'MIntRec':
            self.pretrained_bert_model = 'pretrained_model/bert_en'
            
            self.embd_size_a = 64
            self.embd_size_t = 64
            self.embd_size_v = 64
            self.batch_size = 64
            self.embd_method_a = 'maxpool'
            self.embd_method_v = 'maxpool'
            self.output_dim = 20
            # Phase one Settings.
            self.module_names_p1 = ['A', 'V', 'L', 'C1']
            self.cls_layers_p1 = '64,64'
            self.dropout_rate_p1 = 0.3
            self.niter, self.niter_decay = 10, 10
            
            # Phase two Settings.
            self.module_names_p2 = ['A', 'V', 'L', 'C2', 'AE', 'AE_cycle']
            self.ae_layers = '128,64,32'
            self.n_blocks = 3
            self.cls_layers_p2 = '64,64'
            self.dropout_rate_p2 = 0.1
            self.ce_weight = 1.0
            self.mse_weight = 1.0
            self.cycle_weight = 1.0

            self.batch_size = 64
            self.lr = 0.002
            self.beta1 = 0.5
        else:
            self.logger.warning(f'No default config for dataset {self.dataset}')