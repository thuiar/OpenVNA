import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model_api.configs import BaseConfig
from model_api.models.subnets.AlignSubNet import AlignSubNet
from model_api.models.subnets.BertTextEncoder import BertTextEncoder


class T2FN(nn.Module):
    def __init__(self, config: BaseConfig):
        '''
        Args:
            feature_dims - a length-3 tuple, contains (audio_dim, vision_dim, text_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, vision_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super().__init__()

        # dimensions are specified in the order of audio, vision and text
        self.text_in, self.audio_in, self.vision_in = config.feature_dims
        self.text_hidden, self.audio_hidden, self.vision_hidden = config.hidden_dims
        self.output_dim = config.output_dim

        self.text_out= config.text_out
        self.post_fusion_dim = config.post_fusion_dim

        self.audio_prob, self.vision_prob, self.text_prob, self.post_fusion_prob = config.dropouts

        # Alignment Network
        self.alignment_network = AlignSubNet(config, mode='avg_pool')

        # BERT SUBNET FOR TEXT
        self.text_model = BertTextEncoder(pretrained=config.pretrained_bert_model, finetune=config.finetune_bert)

        # define the pre-fusion subnetworks
        self.audio_lstm = nn.LSTM(self.audio_in, self.audio_hidden, num_layers=1, bidirectional=False, batch_first=True)
        self.vision_lstm = nn.LSTM(self.vision_in, self.vision_hidden, num_layers=1, bidirectional=False, batch_first=True)
        self.text_lstm = nn.LSTM(self.text_in, self.text_hidden, num_layers=1, bidirectional=False, batch_first=True)
        
        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_hidden + 1) * (self.vision_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, self.output_dim)
        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.Tensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.Tensor([-3]), requires_grad=False)

    def forward(self, text, audio, vision, audio_lengths, vision_lengths):
        '''
        Args:
            audio: tensor of shape (batch_size, sequence_len, audio_in)
            vision: tensor of shape (batch_size, sequence_len, vision_in)
            text: tensor of shape (batch_size, sequence_len, text_in)
        '''
        text, audio, vision = self.alignment_network(text, audio, vision, 
                                                           audio_lengths, vision_lengths)
        
        text = self.text_model(text)
        audio_h = self.audio_lstm(audio)[0]
        vision_h = self.vision_lstm(vision)[0]
        text_h = self.text_lstm(text)[0]
        batch_size = audio.data.shape[0]
        seq_len = audio.data.shape[1]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one = torch.ones(size=[batch_size,seq_len, 1], requires_grad=False).type_as(audio_h).to(text.device)
        _audio_h = torch.cat((add_one, audio_h), dim=2)
        _vision_h = torch.cat((add_one, vision_h), dim=2)
        _text_h = torch.cat((add_one, text_h), dim=2)

        fusion_tensor = torch.zeros(size=[batch_size,(self.audio_hidden + 1) * (self.vision_hidden + 1) * (self.text_hidden + 1)], requires_grad=False).type_as(audio_h).to(text.device)
        for t in range(seq_len):
            fusion_tensor_t = torch.bmm(_audio_h[:,t,:].unsqueeze(2), _vision_h[:,t,:].unsqueeze(1))
            fusion_tensor_t = fusion_tensor_t.view(-1, (self.audio_hidden + 1) * (self.vision_hidden + 1), 1)
            fusion_tensor_t = torch.bmm(fusion_tensor_t, _text_h[:,t,:].unsqueeze(1)).view(batch_size, -1)
            fusion_tensor += fusion_tensor_t
        
        Tmp = (self.audio_hidden) * (self.vision_hidden) * (self.text_hidden) / torch.tensor(max((self.audio_hidden),(self.vision_hidden),(self.text_hidden)))
        Tmp = torch.sqrt(Tmp)
        # ||M||* <= Tmp ||M||F

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)

        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped), inplace=True)
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1), inplace=True)
        output = self.post_fusion_layer_3(post_fusion_y_2)
        
        if self.output_dim == 1: # regression
            output = torch.sigmoid(output)
            output = output * self.output_range + self.output_shift

        self.norm = torch.norm(fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.vision_hidden + 1) * (self.text_hidden + 1)), dim=1)
        reg_loss = (Tmp * self.norm).mean()

        return output, reg_loss
