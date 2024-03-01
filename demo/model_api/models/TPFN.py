import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter

from model_api.configs import BaseConfig
from model_api.models.subnets.BertTextEncoder import BertTextEncoder
from model_api.models.subnets.AlignSubNet import AlignSubNet

# The fusion network and the classification network
class TPFN(nn.Module):
    def __init__(self, config: BaseConfig) -> None:
        super().__init__()
        self.config = config
        self.text_drop, self.audio_drop, self.video_drop = config.dropouts

        # dimension of the input modality
        self.time_steps = config.seq_lens[0]

        # dim of Z
        self.len = sum(config.hidden_dims)
        # dim of Z_e
        self.extended_dim = config.time_window * self.len + 1

        self.output_dim = config.output_dim
        self.rank = config.rank
        self.time_window = config.time_window
        self.idx = [config.stride*i for i in range(self.time_steps//config.stride)]

        # Alignment Network
        self.alignment_network = AlignSubNet(config, mode='avg_pool')
        # BERT SUBNET FOR TEXT
        self.text_model = BertTextEncoder(pretrained=config.pretrained_bert_model, finetune=config.finetune_bert)
        # preprocess network
        self.audio_subnet = SubNet(config.feature_dims[1], config.hidden_dims[1], self.audio_drop)
        self.video_subnet = SubNet(config.feature_dims[2], config.hidden_dims[2], self.video_drop)
        self.text_subnet = SubNet(config.feature_dims[0], config.hidden_dims[0], self.text_drop)
        # define the factors for Z and Z_e
        self.linear1 = nn.Linear(self.len, self.rank*self.output_dim)
        self.linear2 = nn.Linear(self.extended_dim, self.rank*self.output_dim)
        # Untrainable weight for summation on rank
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))  # trainable = False

        xavier_normal_(self.fusion_weights)

    def forward(self, text, audio, vision, audio_lengths, video_lengths):
        '''
        Args:
            audio: tensor of shape (batch_size, sequence_len, audio_in)
            vision: tensor of shape (batch_size, sequence_len, video_in)
            text: tensor of shape (batch_size, sequence_len, text_in)
        '''
        text, audio, vision = self.alignment_network(text, audio, vision, 
                                                           audio_lengths, video_lengths)
        text = self.text_model(text)
        # get pre-training vectors for audio, video and text
        audio_h = self.audio_subnet(audio)
        video_h = self.video_subnet(vision)
        text_h = self.text_subnet(text)
        # build Z
        z = torch.cat((audio_h, video_h, text_h), dim=2)  # (B, T, len)
        assert z.data.shape[2] == self.len
        # calculate Z_e
        z_extended = time_extend([z], self.config.device, time_window=self.time_window)
        assert z_extended.data.shape[2] == self.extended_dim
        # calculate M, and sum up according to the stride
        for i in self.idx:
            if i == 0:
                z_fusion = self.linear1(z[:, i, :]) * self.linear2(z_extended[:, i, :]) / len(self.idx)
            else:
                z_fusion += self.linear1(z[:, i, :]) * self.linear2(z_extended[:, i, :]) / len(self.idx)
        # summation on rank
        z_fusion = z_fusion.view(-1, self.rank, self.output_dim)
        z_output = torch.matmul(self.fusion_weights, z_fusion).squeeze()
        output = z_output.view(-1, self.output_dim)
        # calculate norm of Z for regularization 
        norm = torch.norm(z.view(-1, self.time_steps*self.len), dim=1)
        norm = torch.sum(norm) / audio_h.shape[0]
        # res = {
        #     'M': output,
        #     'feature':z_fusion
        # }
        return output, norm


# The pre-process network for every modality
class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout=0.2, num_layers=1, bidirectional=False):
        super(SubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        all_states, _ = self.rnn(x)
        state_shape = all_states.shape
        all_states = torch.reshape(all_states, [-1, state_shape[-1]])
        all_h = self.dropout(all_states)
        all_y = torch.reshape(all_h, [state_shape[0], state_shape[1], -1])
        return all_y


# Extend the feature Z to Z_e for every time step
def time_extend(x, device, time_window=2):
    batch_size = x[0].data.shape[0]
    time_step = x[0].data.shape[1]
    ones = Variable(torch.ones(batch_size, 1).to(device), requires_grad=False)  # (B, 1)
    for i in range(time_step):
        y_i = ones
        for j in range(time_window):
            for xt in x:
                try: y_i = torch.cat((y_i, xt[:, i + j, :]), 1)
                except: y_i = torch.cat((y_i, xt[:, i + j - time_step, :]), 1)
        y_i = y_i.view(batch_size, 1, -1)
        try: y = torch.cat((y, y_i), 1)
        except: y = y_i
    return y

