import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.init import xavier_normal

from model_api.models.subNets.AlignSubNet import AlignSubNet
from model_api.models.subNets.FeatureNets import SubNet
from model_api.models.subNets.BertTextEncoder import BertTextEncoder
# The pre-process network for every modality
class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout=0.2, num_layers=1, bidirectional=False):
        super(SubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
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

# The fusion network and the classification network
class TPFN(nn.Module):
    def __init__(self, args):
        super(TPFN, self).__init__()
        self.args = args
        self.text_model = BertTextEncoder(pretrained=args.pretrained, finetune=args.use_bert_finetune)
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        t_hidden, a_hidden, v_hidden = args.hidden_dims
        tdr, adr, vdr = args.dropouts
        stride=args.stride
        self.time_steps = args.seq_lens[0]
        # dimension of the input modality

        # number of hidden units in the pre-process LSTM
        self.audio_hidden = a_hidden
        self.video_hidden = v_hidden
        self.text_hidden = t_hidden

        # dim of Z
        self.len = t_hidden + a_hidden + v_hidden
        # dim of Z_e
        self.extended_dim = args.time_window * self.len + 1

        self.output_dim = args.output_dim
        self.rank = args.rank
        self.time_window = args.time_window
        self.idx = [stride*i for i in range(self.time_steps//stride)]

        # dropout probability in LSTM
        self.audio_drop = adr
        self.video_drop = vdr
        self.text_drop = tdr
        self.align_subnet = AlignSubNet(args, mode='avg_pool')
        # preprocess network
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_drop)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_drop)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, self.text_drop)

        # self.DTYPE = torch.cuda.FloatTensor
        # define the factors for Z and Z_e
        self.linear1 = nn.Linear(self.len, self.rank*self.output_dim)
        self.linear2 = nn.Linear(self.extended_dim, self.rank*self.output_dim)
        # Untrainable weight for summation on rank
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))  # trainable = False

        # if is_cuda:
        #     self.DTYPE = torch.cuda.FloatTensor
        #     self.linear1.cuda()
        #     self.linear2.cuda()
        #     self.fusion_weights.cuda()

        xavier_normal(self.fusion_weights)

    def forward(self, text, audio, vision):
        text_x, audio_x, video_x = text[0], audio[0], vision[0]
        text_lengths,audio_lengths,video_lengths = text[1], audio[1], vision[1]
        text_x,audio_x,video_x = self.align_subnet(text_x, audio_x, video_x,text_lengths,audio_lengths,video_lengths)
        text_x = self.text_model(text_x)
        # get pre-training vectors for audio, video and text
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        # build Z
        z = torch.cat((audio_h, video_h, text_h), dim=2)  # (B, T, len)
        assert z.data.shape[2] == self.len
        # calculate Z_e
        z_extended = time_extend([z], self.args.device, time_window=self.time_window)
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
        self.norm = torch.norm(z.view(-1, self.time_steps*self.len), dim=1)
        self.norm = torch.sum(self.norm) / audio_h.shape[0]
        return output