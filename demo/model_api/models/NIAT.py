import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model_api.models.subnets.AlignSubNet import AlignSubNet
from model_api.models.subnets.BertTextEncoder import BertTextEncoder
from model_api.models.subnets.transformers_encoder.transformer import TransformerEncoder


class NIAT(nn.Module):
   
    def __init__(self, config):
        super(NIAT, self).__init__()
        self.config = config
        self.alignment_network = AlignSubNet(config, mode='avg_pool')
        self.fusion = transformer_based(config)
        self.reconstruction = decoder_v1(config)
        self.discriminator = disc_two_class(config)
        self.classifier = classifier(config)
        
    def forward(self, text_x, audio_x, video_x):
        pass


class transformer_based(nn.Module):
    def __init__(self, config):
        super(transformer_based, self).__init__()
        self.config = config
        # BERT SUBNET FOR TEXT
        self.text_model = BertTextEncoder(pretrained=config.pretrained_bert_model, finetune=config.finetune_bert)
        config.fusion_dim = config.fus_d_l+config.fus_d_a+config.fus_d_v
        orig_d_l, orig_d_a, orig_d_v = config.feature_dims
        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(orig_d_l, config.fus_d_l, kernel_size=config.fus_conv1d_kernel_l, padding=(config.fus_conv1d_kernel_l-1)//2, bias=False)
        self.proj_a = nn.Conv1d(orig_d_a, config.fus_d_a, kernel_size=config.fus_conv1d_kernel_a, padding=(config.fus_conv1d_kernel_a-1)//2, bias=False)
        self.proj_v = nn.Conv1d(orig_d_v, config.fus_d_v, kernel_size=config.fus_conv1d_kernel_v, padding=(config.fus_conv1d_kernel_v-1)//2, bias=False)

        self.fusion_trans = TransformerEncoder(embed_dim=config.fus_d_l+config.fus_d_a+config.fus_d_v, num_heads=config.fus_nheads, layers=config.fus_layers, 
                                                attn_dropout=config.fus_attn_dropout, relu_dropout=config.fus_relu_dropout, res_dropout=config.fus_res_dropout,
                                                embed_dropout=config.fus_embed_dropout, attn_mask=config.fus_attn_mask)

    def forward(self, text_x, audio_x, video_x):
        x_l = self.text_model(text_x).transpose(1, 2)
        x_a = audio_x.transpose(1, 2) # batch_size, da, seq_len
        x_v = video_x.transpose(1, 2)
        
        proj_x_l = self.proj_l(x_l).permute(2, 0, 1) # seq_len, batch_size, dl
        proj_x_a = self.proj_a(x_a).permute(2, 0, 1)
        proj_x_v = self.proj_v(x_v).permute(2, 0, 1)

        trans_seq = self.fusion_trans(torch.cat((proj_x_l, proj_x_a, proj_x_v), axis=2))
        if type(trans_seq) == tuple:
            trans_seq = trans_seq[0]

        return trans_seq[0] # Utilize the [CLS] of text for full sequences representation.    

class decoder_v1(nn.Module):
    """效仿ARGF模型"""
    def __init__(self, config):
        super(decoder_v1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(config.fusion_dim, config.rec_hidden_dim1),
            nn.Dropout(config.rec_dropout),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(config.rec_hidden_dim1, config.rec_hidden_dim2),
            nn.Dropout(config.rec_dropout),
            nn.BatchNorm1d(config.rec_hidden_dim2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(config.rec_hidden_dim2, config.fusion_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class disc_two_class(nn.Module):
    """效仿ARGF模型"""
    def __init__(self, config):
        """ Basic Binary Discriminator. 
        """
        super(disc_two_class, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm1d(config.fusion_dim),
            nn.Linear(config.fusion_dim, config.disc_hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(config.disc_hidden_dim1, config.disc_hidden_dim2),
            nn.Tanh(),
            nn.Linear(config.disc_hidden_dim2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)


class classifier(nn.Module):

    def __init__(self, config):

        super(classifier, self).__init__()
        self.norm = nn.BatchNorm1d(config.fusion_dim)
        self.drop = nn.Dropout(config.clf_dropout)
        self.linear_1 = nn.Linear(config.fusion_dim, config.clf_hidden_dim)
        self.linear_2 = nn.Linear(config.clf_hidden_dim, 1)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, fusion_feature):
        '''
        config:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(fusion_feature)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        # y_2 = F.sigmoid(self.linear_2(y_1))
        y_2 = torch.sigmoid(self.linear_2(y_1))
        # 强制将输出结果转化为 [-3,3] 之间
        output = y_2 * self.output_range + self.output_shift

        return output

