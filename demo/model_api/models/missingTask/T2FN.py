"""
paper: Tensor Fusion Network for Multimodal Sentiment Analysis
From: https://github.com/A2Zadeh/TensorFusionNetwork
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from model_api.models.subNets.AlignSubNet import AlignSubNet
from model_api.models.subNets.BertTextEncoder import BertTextEncoder

class T2FN(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''

    def __init__(self, args):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(T2FN, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.output_dim = 1
        self.align_subnet = AlignSubNet(args, mode='avg_pool')
        self.text_out= args.text_out
        self.post_fusion_dim = args.post_fusion_dim

        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = args.dropouts

        # BERT SUBNET FOR TEXT
        
        self.text_model = BertTextEncoder(pretrained=args.pretrained, finetune=args.use_bert_finetune)

        # define the pre-fusion subnetworks
        # self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.audio_lstm = nn.LSTM(self.audio_in, self.audio_hidden, num_layers=1, dropout=self.audio_prob, bidirectional=False, batch_first=True)
        # self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.video_lstm = nn.LSTM(self.video_in, self.video_hidden, num_layers=1, dropout=self.video_prob, bidirectional=False, batch_first=True)
        # self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)
        self.text_lstm = nn.LSTM(self.text_in, self.text_hidden, num_layers=1, dropout=self.text_prob, bidirectional=False, batch_first=True)
        
        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_hidden + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, self.output_dim)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, text, audio, vision):
        '''
        Args:
        '''
        text_x, audio_x, video_x = text[0], audio[0], vision[0]
        text_lengths,audio_lengths,video_lengths = text[1], audio[1], vision[1]
        text_x,audio_x,video_x = self.align_subnet(text_x, audio_x, video_x,text_lengths,audio_lengths,video_lengths)
        
        text_x = self.text_model(text_x)
        audio_h = self.audio_lstm(audio_x)[0]
        video_h = self.video_lstm(video_x)[0]
        text_h = self.text_lstm(text_x)[0]
        batch_size = audio_x.data.shape[0]
        seq_len = audio_x.data.shape[1]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one = torch.ones(size=[batch_size,seq_len, 1], requires_grad=False).type_as(audio_h).to(text_x.device)
        _audio_h = torch.cat((add_one, audio_h), dim=2)
        _video_h = torch.cat((add_one, video_h), dim=2)
        _text_h = torch.cat((add_one, text_h), dim=2)

        fusion_tensor = torch.zeros(size=[batch_size,(self.audio_hidden + 1) * (self.video_hidden + 1) * (self.text_hidden + 1)], requires_grad=False).type_as(audio_h).to(text_x.device)
        for t in range(seq_len):
            fusion_tensor_t = torch.bmm(_audio_h[:,t,:].unsqueeze(2), _video_h[:,t,:].unsqueeze(1))
            fusion_tensor_t = fusion_tensor_t.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
            fusion_tensor_t = torch.bmm(fusion_tensor_t, _text_h[:,t,:].unsqueeze(1)).view(batch_size, -1)
            fusion_tensor += fusion_tensor_t
        
        Tmp = (self.audio_hidden) * (self.video_hidden) * (self.text_hidden) / torch.tensor(max((self.audio_hidden),(self.video_hidden),(self.text_hidden)))
        Tmp = torch.sqrt(Tmp)
        # ||M||* <= Tmp ||M||F

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)

        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped), inplace=True)
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1), inplace=True)
        output = self.post_fusion_layer_3(post_fusion_y_2)
        
        if self.output_dim == 1: # regression
            output = torch.sigmoid(output)
            output = output * self.output_range + self.output_shift

        self.norm = self.M_xing = torch.norm(fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1) * (self.text_hidden + 1)), dim=1)
        self.loss2 = (Tmp * self.norm).mean()

        return output
