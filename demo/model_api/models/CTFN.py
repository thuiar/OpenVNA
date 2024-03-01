import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Dropout, Module
from torch.nn.modules import ModuleList
import warnings
from torch.nn import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_

from model_api.configs import BaseConfig
from model_api.models.subnets.AlignSubNet import AlignSubNet
from model_api.models.subnets.BertTextEncoder import BertTextEncoder

def my_relu(x):
    return torch.maximum(x, torch.zeros_like(x))

class CTFN(nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        text_dim, audio_dim, vision_dim = config.feature_dims
        self.align_subnet = AlignSubNet(config, mode='avg_pool')
        self.text_model = BertTextEncoder(pretrained=config.pretrained_bert_model, finetune=config.finetune_bert)

        self.a2t_model = nn.ModuleList([
            TransEncoder(d_dual=(audio_dim, text_dim), 
                         d_model=config.t_d_model, nhead=config.t_heads, 
                         num_encoder_layers=config.t_num_layer,
                         dim_feedforward=config.t_dim_forward,
                         dropout=config.trans_dropout),

            TransEncoder(d_dual=( text_dim, audio_dim), 
                         d_model=config.a_d_model, 
                         nhead=config.a_heads,
                         num_encoder_layers=config.a_num_layer,
                         dim_feedforward=config.a_dim_forward, 
                         dropout=config.trans_dropout)
        ])

        self.a2v_model = nn.ModuleList([
            TransEncoder(d_dual=(audio_dim, vision_dim), 
                         d_model=config.v_d_model, nhead=config.v_heads, 
                         num_encoder_layers=config.v_num_layer,
                         dim_feedforward=config.v_dim_forward,
                         dropout=config.trans_dropout),

            TransEncoder(d_dual=(vision_dim, audio_dim), 
                         d_model=config.a_d_model, 
                         nhead=config.a_heads,
                         num_encoder_layers=config.a_num_layer,
                         dim_feedforward=config.a_dim_forward, 
                         dropout=config.trans_dropout)
        ])

        self.v2t_model = nn.ModuleList([
            TransEncoder(d_dual=(vision_dim, text_dim), 
                         d_model=config.t_d_model, nhead=config.t_heads, 
                         num_encoder_layers=config.t_num_layer,
                         dim_feedforward=config.t_dim_forward,
                         dropout=config.trans_dropout),

            TransEncoder(d_dual=(text_dim, vision_dim), 
                         d_model=config.v_d_model, 
                         nhead=config.v_heads,
                         num_encoder_layers=config.v_num_layer,
                         dim_feedforward=config.v_dim_forward, 
                         dropout=config.trans_dropout)
        ])
    
        self.sa_model = EmotionClassifier(config)

    def forward(self, *config, **kwconfig) -> None:
        pass


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


class TransEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm(src)
        src2 = self.linear2(self.dropout(my_relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm(src)
        return src


class TransEncoder(Module):
    def __init__(self, d_dual, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_encoder_layers
        self.linear1 = Linear(d_dual[0], d_model)
        self.linear2 = Linear(d_model, d_dual[1])
        self.dropout = Dropout(dropout)

        encoder_layer = TransEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)])

        self.norm = LayerNorm(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        res = list()
        output = self.dropout(my_relu(self.linear1(src)))
        res.append(output)
        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            res.append(output)
        if self.norm:
            output = self.norm(output)
            res.append(output)
        return self.linear2(output), res


class CNNFusionBlock(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, out_channels=128, kernel_heights=[3,4,5], dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        embd_size = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights)*out_channels, embd_size),
            nn.ReLU(inplace=True),
        )

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = my_relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, frame_x):
        batch_size, seq_len, feat_dim = frame_x.size()
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        max_out3 = self.conv_block(frame_x, self.conv3)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)
        return embd


class EmotionClassifier(nn.Module):
    def __init__(self, config):
        super(EmotionClassifier, self).__init__()
        self.output_dim = config.output_dim
        self.rnn_dropout = nn.Dropout(p=0.3, inplace=True)

        self.rnn_text = nn.LSTM(input_size=config.feature_dims[0], hidden_size=config.gru_units,
                                num_layers=1, bidirectional=False, dropout=0.0)
        self.rnn_audio = nn.LSTM(input_size=config.feature_dims[1], hidden_size=config.gru_units,
                                 num_layers=1, bidirectional=False, dropout=0.0)
        self.rnn_video = nn.LSTM(input_size=config.feature_dims[2], hidden_size=config.gru_units,
                                 num_layers=1, bidirectional=False, dropout=0.0)

        self.dense_text = nn.Linear(in_features=config.gru_units * 1, out_features=config.dense_units)
        self.dense_audio = nn.Linear(in_features=config.gru_units * 1, out_features=config.dense_units)
        self.dense_video = nn.Linear(in_features=config.gru_units * 1, out_features=config.dense_units)

        self.dense_dropout = nn.Dropout(p=0.3, inplace=True)

        cat_dims = (config.a_d_model + config.t_d_model + config.v_d_model)*2 + config.dense_units * 3
        self.fusionBlock = CNNFusionBlock(input_dim = cat_dims, out_channels=config.dense_units)
        self.out_layer_1 = nn.Linear(in_features=config.dense_units, out_features=config.dense_units)
        self.out_layer_2 = nn.Linear(in_features=config.dense_units, out_features=config.output_dim)
        self.out_dropout = nn.Dropout(p=0.3, inplace=True)
    
    def forward(self, audio, text, video, audio_a2t, text_a2t, video_v2t, text_v2t, audio_a2v, video_a2v):
        rnn_t, _ = self.rnn_text(text)
        encoded_text = my_relu(self.dense_dropout(self.dense_text(my_relu(rnn_t))))
        rnn_a, _ = self.rnn_audio(audio)
        encoded_audio = my_relu(self.dense_dropout(self.dense_audio(my_relu(rnn_a))))
        rnn_v, _ = self.rnn_video(video)
        encoded_video = my_relu(self.dense_dropout(self.dense_video(my_relu(rnn_v))))

        encoded_feature = torch.cat((encoded_text, encoded_audio, encoded_video,audio_a2t, text_a2t, video_v2t, text_v2t, audio_a2v, video_a2v), dim=-1)
        
        out = self.fusionBlock(encoded_feature)
        
        out = self.out_dropout(my_relu(self.out_layer_1(out)))

        return self.out_layer_2(out) 
