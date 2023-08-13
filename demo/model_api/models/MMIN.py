import torch
from model_api.models.base_model import BaseModel
from model_api.models.networks.lstm import LSTMEncoder
from model_api.models.networks.classifier import FcClassifier
from model_api.models.networks.autoencoder import ResidualAE,TextCNN
from model_api.models.subNets.BertTextEncoder import BertTextEncoder

class MMINModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=25, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=768, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=171, help='visual input dim')
        parser.add_argument('--embd_size_a', default=64, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=64, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=64, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--AE_layers', type=str, default='128,64,32', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cls_layers', type=str, default='64,64', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.1, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--output_dim', type=int, default=1, help='output classification. linear classification')
        parser.add_argument('--share_weight', action='store_true', help='share weight of forward and backward autoencoders')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['L1', 'mse', 'cycle']
        self.model_names = ['A', 'V', 'L', 'C', 'AE', 'AE_cycle']
        
        self.text_model = BertTextEncoder(pretrained='bert-base-uncased', finetune=False)
        self.text_model = self.text_model.to(self.device)
        # acoustic model
        self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.netA.to(self.device)
        # lexical model
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
        self.netL.to(self.device)

        # visual model
        self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
        self.netV.to(self.device)

        # AE model
        AE_layers = list(map(lambda x: int(x), opt.AE_layers.split(',')))
        AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        self.netAE = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        self.netAE.to(self.device)
        if opt.share_weight:
            self.netAE_cycle = self.netAE
            self.model_names.pop(-1)
        else:
            self.netAE_cycle = ResidualAE(AE_layers, opt.n_blocks, AE_input_dim, dropout=0, use_bn=False)
        self.netAE_cycle = self.netAE_cycle.to(self.device)
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = AE_layers[-1] * opt.n_blocks
        
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netC.to(self.device)

    def set_input(self, batch_data):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """

        text = torch.from_numpy(batch_data['text_bert']).to(self.device)
        audio = torch.from_numpy(batch_data['audio']).float().to(self.device)
        vision = torch.from_numpy(batch_data['vision']).float().to(self.device)

        self.A_miss = audio
        self.V_miss = vision
        self.L_miss = text

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # get utt level representattion
        self.feat_A_miss = self.netA(self.A_miss)
        self.L_miss = self.text_model(self.L_miss)

        self.feat_L_miss = self.netL(self.L_miss)
        self.feat_V_miss = self.netV(self.V_miss)
        # fusion miss
        self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)
        # calc reconstruction of teacher's output
        self.recon_fusion, self.latent = self.netAE(self.feat_fusion_miss)
        self.recon_cycle, self.latent_cycle = self.netAE_cycle(self.recon_fusion)
        # get fusion outputs for missing modality
        self.logits, _ = self.netC(self.latent)
        self.pred = self.logits
        
    def backward(self):
        pass

    def optimize_parameters(self):
        pass
