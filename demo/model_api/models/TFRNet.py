""" TFR-Net Trainer is partially adapted from https://github.com/thuiar/TFR-Net.
    NOTE: Modification is made for fair comparison under the situation where the missing 
    position is unknown in the noisy instances during the training periods.
"""

import torch
import torch.nn as nn

from model_api.configs import BaseConfig
import torch.nn.functional as F
from model_api.models.subnets.BertTextEncoder import BertTextEncoder
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_api.models.subnets.transformers_encoder.transformer import TransformerEncoder


# The fusion network and the classification network
class TFRNet(nn.Module):
    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        self.text_model = BertTextEncoder(pretrained=config.pretrained_bert_model, finetune=config.finetune_bert)
        self.align_subnet = CM_ATTN(config)

        if config.mode == 'train':
            self.generator_t = LinearTrans(config, modality='text')
            self.generator_a = LinearTrans(config, modality='audio')
            self.generator_v = LinearTrans(config, modality='vision')
            self.gen_loss = RECLoss(config)
   
        config.fusion_t_in = config.fusion_a_in = config.fusion_v_in = config.dst_feature_dim_nheads[0] * 3
        self.fusion_subnet = GATE_F(config)
    
    def forward(self, text, audio, vision, audio_lengths, vision_lengths, mode='train'):
        text, text_m = text
        audio, audio_m = audio 
        vision, vision_m= vision
        # length to mask.
        text_mask = text[:,1,:]
        text_lengths = torch.sum(text_mask, dim=1)
        if mode == "train":
            audio_mask = torch.arange(audio.shape[1]).expand(audio.shape[0], audio.shape[1]).to(audio.device) < audio_lengths.unsqueeze(1)
            vision_mask = torch.arange(vision.shape[1]).expand(vision.shape[0], vision.shape[1]).to(vision.device) < vision_lengths.unsqueeze(1)

            text_m = self.text_model(text_m)
            text = self.text_model(text)

            text_h, audio_h, vision_h, text_h_g, audio_h_g, vision_h_g = self.align_subnet(text_m, audio_m, vision_m)
    
            text_ = self.generator_t(text_h_g)
            audio_ = self.generator_a(audio_h_g)
            vision_ = self.generator_v(vision_h_g)

            text_gen_loss = self.gen_loss(text_, text, text_mask)
            audio_gen_loss = self.gen_loss(audio_, audio, audio_mask)
            vision_gen_loss = self.gen_loss(vision_, vision, vision_mask)

            prediction = self.fusion_subnet((text_h, text_lengths), (audio_h, audio_lengths), (vision_h, vision_lengths))

            return prediction, self.config.weight_gen_loss[0] * text_gen_loss + self.config.weight_gen_loss[1] * audio_gen_loss + self.config.weight_gen_loss[2] * vision_gen_loss
            
        else:
            text_mask = text[:,1,:]
            text = self.text_model(text)
            text_h, audio_h, vision_h, text_h_g, audio_h_g, vision_h_g = self.align_subnet(text, audio, vision)
            prediction = self.fusion_subnet((text_h, text_lengths), (audio_h, audio_lengths), (vision_h, vision_lengths))
            return prediction


class RECLoss(nn.Module):
    def __init__(self, args):
        super(RECLoss, self).__init__()

        self.eps = torch.FloatTensor([1e-4]).to(args.device)
        self.args = args

        if args.recloss_type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif args.recloss_type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        elif args.recloss_type == 'cmd':
            self.loss = CMD()
        elif args.recloss_type == 'combine':
            self.loss = nn.SmoothL1Loss(reduction='sum')
            self.loss_cmd = CMD()

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2])

        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + self.eps)

        if self.args.recloss_type == 'combine' and self.args.weight_sim_loss!=0:
            loss += (self.args.weight_sim_loss * self.loss_cmd(pred*mask, target*mask) / (torch.sum(mask) + self.eps))
        return loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=3):
        x1 = x1.view(-1, x1.shape[-1])
        x2 = x2.view(-1, x2.shape[-1])
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        b = torch.max(x2, dim=0)[0]
        a = torch.min(x2, dim=0)[0]
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = (summed+1e-12)**(0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class CM_ATTN(nn.Module):
    def __init__(self, args):
        super(CM_ATTN, self).__init__()
        self.args = args
        self.seq_lens = args.seq_lens
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        args.generator_in = (dst_feature_dims*2, dst_feature_dims*2, dst_feature_dims*2)

        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout

        
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=(args.conv1d_kernel_size_l-1)//2, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=(args.conv1d_kernel_size_a-1)//2, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=(args.conv1d_kernel_size_v-1)//2, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
    
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
    
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Intramodal Attentions
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_final = self.get_network(self_type='l_final', layers=3)
        self.trans_a_final = self.get_network(self_type='a_final', layers=3)
        self.trans_v_final = self.get_network(self_type='v_final', layers=3)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_l, self.attn_dropout, self.num_heads, True
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_a, self.attn_dropout_a, self.num_heads, True
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_v, self.attn_dropout_v, self.num_heads, True
        elif self_type == 'l_mem':
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_l, self.attn_dropout, self.num_heads, True
        elif self_type == 'a_mem':
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_a, self.attn_dropout, self.num_heads, True
        elif self_type == 'v_mem':
            embed_dim, attn_dropout, num_heads, position_embedding = self.d_v, self.attn_dropout, self.num_heads, True
        elif self_type == 'l_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.seq_lens[0], self.attn_dropout, self.args.num_temporal_head, False
        elif self_type == 'a_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.seq_lens[1], self.attn_dropout, self.args.num_temporal_head, False
        elif self_type == 'v_final':
            embed_dim, attn_dropout, num_heads, position_embedding = self.seq_lens[2], self.attn_dropout, self.args.num_temporal_head, False
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  position_embedding=position_embedding)

    def forward(self, text, audio, vision):
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = vision.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        # (V,A) --> L
        h_l = self.trans_l_mem(proj_x_l)
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l, h_l_with_as, h_l_with_vs], dim=2)
        h_ls_n = self.trans_l_final(h_ls.permute(1,2,0)).permute(0,2,1)
        # (L,V) --> A
        h_a = self.trans_a_mem(proj_x_a)
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a, h_a_with_ls, h_a_with_vs], dim=2)
        h_as_n = self.trans_a_final(h_as.permute(1,2,0)).permute(0,2,1)
        # (L,A) --> V
        h_v = self.trans_v_mem(proj_x_v)
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v, h_v_with_ls, h_v_with_as], dim=2)
        h_vs_n = self.trans_v_final(h_vs.permute(1,2,0)).permute(0,2,1)
        return h_ls.transpose(0, 1), h_as.transpose(0, 1), h_vs.transpose(0, 1), h_ls_n, h_as_n, h_vs_n,


class GRUencoder(nn.Module):
    """Pad for utterances with variable lengths and maintain the order of them after GRU"""
    def __init__(self, embedding_dim, utterance_dim, num_layers):
        super(GRUencoder, self).__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=utterance_dim,
                          bidirectional=True, num_layers=num_layers)

    def forward(self, utterance, utterance_lens):
        """Server as simple GRU Layer.
        Args:
            utterance (tensor): [utter_num, max_word_len, embedding_dim]
            utterance_lens (tensor): [utter_num]
        Returns:
            transformed utterance representation (tensor): [utter_num, max_word_len, 2 * utterance_dim]
        """
        utterance_embs = utterance.transpose(0,1)
    
        # SORT BY LENGTH.
        sorted_utter_length, indices = torch.sort(utterance_lens, descending=True)
        _, indices_unsort = torch.sort(indices)
        
        s_embs = utterance_embs.index_select(1, indices)

        # PADDING & GRU MODULE & UNPACK.
        utterance_packed = pack_padded_sequence(s_embs, sorted_utter_length.cpu())
        utterance_output = self.gru(utterance_packed)[0]
        utterance_output = pad_packed_sequence(utterance_output, total_length=utterance.size(1))[0]

        # UNSORT BY LENGTH.
        utterance_output = utterance_output.index_select(1, indices_unsort)
        return utterance_output.transpose(0,1)


class C_GATE(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, drop):
        super(C_GATE, self).__init__()

        # BI-GRU to get the historical context.
        self.gru = GRUencoder(embedding_dim, hidden_dim, num_layers)
        # Calculate the gate.
        self.cnn = nn.Conv1d(in_channels= 2 * hidden_dim, out_channels=1, kernel_size=3, stride=1, padding=1)
        # Linear Layer to get the representation.
        self.fc = nn.Linear(hidden_dim * 2 + embedding_dim, hidden_dim)
        # Utterance Dropout.
        self.dropout_in = nn.Dropout(drop)
        
    def forward(self, utterance, utterance_lens):
        """Returns:
            utterance_rep: [utter_num, utterance_dim]
        """
        # add_zero = torch.zeros(size=[utterance.shape[0], 1], requires_grad=False).type_as(utterance_mask).to(utterance_mask.device)
        # utterance_mask = torch.cat((utterance_mask, add_zero), dim=1)
        # utterance_lens = torch.argmin(utterance_mask, dim=1)

        # Bi-GRU
        transformed_ = self.gru(utterance, utterance_lens) # [batch_size, seq_len, 2 * hidden_dim]
        # CNN_GATE MODULE.
        gate = torch.sigmoid(self.cnn(transformed_.transpose(1, 2)).transpose(1, 2))  # [batch_size, seq_len, 1]
        # CALCULATE GATE OUTPUT.
        gate_x = torch.tanh(transformed_) * gate # [batch_size, seq_len, 2 * hidden_dim]
        # SPACE TRANSFORMS
        utterance_rep = torch.tanh(self.fc(torch.cat([utterance, gate_x], dim=-1))) # [batch_size, seq_len, hidden_dim]
        # MAXPOOLING LAYERS
        utterance_rep = torch.max(utterance_rep, dim=1)[0] # [batch_size, hidden_dim]
        # UTTERANCE DROPOUT
        utterance_rep = self.dropout_in(utterance_rep) # [utter_num, utterance_dim]
        return utterance_rep


class GATE_F(nn.Module):
    def __init__(self, args):
        super(GATE_F, self).__init__()
        
        self.text_encoder = C_GATE(args.fusion_t_in, args.fusion_t_hid, args.fusion_gru_layers, args.fusion_drop)
        self.audio_encoder = C_GATE(args.fusion_a_in, args.fusion_a_hid, args.fusion_gru_layers, args.fusion_drop)
        self.vision_encoder = C_GATE(args.fusion_v_in, args.fusion_v_hid, args.fusion_gru_layers, args.fusion_drop)

        # Classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module('linear_trans_norm', nn.BatchNorm1d(args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid))
        self.classifier.add_module('linear_trans_hidden', nn.Linear(args.fusion_t_hid + args.fusion_a_hid + args.fusion_v_hid, args.cls_hidden_dim))
        self.classifier.add_module('linear_trans_activation', nn.LeakyReLU())
        self.classifier.add_module('linear_trans_drop', nn.Dropout(args.cls_dropout))
        self.classifier.add_module('linear_trans_final', nn.Linear(args.cls_hidden_dim, args.output_dim))

    def forward(self, text_x, audio_x, vision_x):
        text_x, text_lengths = text_x
        audio_x, audio_lengths = audio_x
        vision_x, vision_lengths = vision_x

        text_rep = self.text_encoder(text_x, text_lengths)
        audio_rep = self.audio_encoder(audio_x, audio_lengths)
        vision_rep = self.vision_encoder(vision_x, vision_lengths)

        utterance_rep = torch.cat((text_rep, audio_rep, vision_rep), dim=1)
        return self.classifier(utterance_rep)


class LinearTrans(nn.Module):
    def __init__(self, args, modality='text'):
        super(LinearTrans, self).__init__()
        if modality == 'text':
            in_dim, out_dim = args.dst_feature_dim_nheads[0] * 3, args.feature_dims[0]
        elif modality == 'audio':
            in_dim, out_dim = args.dst_feature_dim_nheads[0] * 3, args.feature_dims[1]
        elif modality == 'vision':
            in_dim, out_dim = args.dst_feature_dim_nheads[0] * 3, args.feature_dims[2]

        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        return self.linear(x)
