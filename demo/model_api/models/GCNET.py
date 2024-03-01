""" Warnings: The original gcnet model is used on video (taking utterances as unit).
        In our reproducement, the gcnet is used on utterance (taking each spoken words as unit).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model_api.configs import BaseConfig
from torch.autograd import Variable
from torch_geometric.nn import RGCNConv, GraphConv


class GCNET(nn.Module):

    def __init__(self, config: BaseConfig):
        super(GCNET, self).__init__()
        tdim, adim, vdim = config.feature_dims
        self.config = config
        self.lstm = nn.LSTM(input_size=adim+tdim+vdim, hidden_size=config.hidden, num_layers=2, bidirectional=True, dropout=config.dropout)

        ## gain graph models for 'temporal' and 'speaker'
        n_relations = 3
        self.graph_net_temporal = GraphNetwork(2*config.hidden, n_relations, config.time_attn, config.hidden // 2, config.dropout)
        n_relations = config.n_speakers ** 2
        self.graph_net_speaker = GraphNetwork(2*config.hidden, n_relations, config.time_attn, config.hidden // 2, config.dropout)

        ## classification and reconstruction
        D_h = 2 * config.hidden + config.hidden // 2
        self.smax_fc  = nn.Linear(D_h, config.output_dim)
        self.linear_rec = nn.Linear(D_h, adim+tdim+vdim)

    def forward(self, inputfeats, qmask, umask, seq_lengths):
        outputs, _ = self.lstm(inputfeats)
        outputs = outputs.unsqueeze(2)

        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths, self.config.n_speakers, 
                                                             self.config.windowp, self.config.windowf, 'temporal', self.config.device)
        assert len(edge_type_mapping) == 3
        hidden1 = self.graph_net_temporal(features, edge_index, edge_type, seq_lengths, umask)
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths, self.config.n_speakers, 
                                                             self.config.windowp, self.config.windowf, 'speaker', self.config.device)
        assert len(edge_type_mapping) == self.config.n_speakers ** 2
        hidden2 = self.graph_net_speaker(features, edge_index, edge_type, seq_lengths, umask)
        hidden = hidden1 + hidden2

        ## for classification
        hidden_ = F.max_pool1d(hidden.permute(1,2,0), hidden.size(0), hidden.size(0)).squeeze()
        log_prob = self.smax_fc(hidden_) # [seqlen, batch, n_classes]

        ## for reconstruction
        rec_outputs = [self.linear_rec(hidden)]

        return log_prob, rec_outputs, hidden


class GraphNetwork(nn.Module):
    def __init__(self, num_features, num_relations, time_attn, hidden_size=64, dropout=0.5):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNetwork, self).__init__()
        self.time_attn = time_attn
        self.hidden_size = hidden_size

        ## graph modeling
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations)
        self.conv2 = GraphConv(hidden_size, hidden_size)

        ## nodal attention
        D_h = num_features+hidden_size
        self.grufusion = nn.LSTM(input_size=D_h, hidden_size=D_h, num_layers=2, bidirectional=True, dropout=dropout)

        ## sequence attention
        self.matchatt = MatchingAttention(2*D_h, 2*D_h, att_type='general2')
        self.linear = nn.Linear(2*D_h, D_h)


    def forward(self, features, edge_index, edge_type, seq_lengths, umask):
        '''
        features: input node features: [num_nodes, in_channels]
        edge_index: [2, edge_num]
        edge_type: [edge_num]
        '''

        ## graph model: graph => outputs
        out = self.conv1(features, edge_index, edge_type) # [num_features -> hidden_size]
        out = self.conv2(out, edge_index) # [hidden_size -> hidden_size]
        outputs = torch.cat([features, out], dim=-1) # [num_nodes, num_features(16)+hidden_size(8)]

        ## change utterance to conversation: (outputs->outputs)
        outputs = outputs.reshape(-1, outputs.size(1)) # [num_utterance, dim]
        outputs = utterance_to_conversation(outputs, seq_lengths, umask, features.device) # [seqlen, batch, dim]
        outputs = outputs.reshape(outputs.size(0), outputs.size(1), 1, -1) # [seqlen, batch, ?, dim]

        ## outputs -> outputs:
        seqlen = outputs.size(0)
        batch = outputs.size(1)
        outputs = torch.reshape(outputs, (seqlen, batch, -1)) # [seqlen, batch, dim]
        outputs = self.grufusion(outputs)[0] # [seqlen, batch, dim]

        ## outputs -> hidden:
        ## sequence attention => [seqlen, batch, d_h]
        if self.time_attn:
            alpha = []
            att_emotions = []
            for t in outputs: # [bacth, dim]
                # att_em: [batch, mem_dim] # alpha_: [batch, 1, seqlen]
                att_em, alpha_ = self.matchatt(outputs, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0)) # [1, batch, mem_dim]
                alpha.append(alpha_[:,0,:]) # [batch, seqlen]
            att_emotions = torch.cat(att_emotions, dim=0) # [seqlen, batch, mem_dim]
            hidden = F.relu(self.linear(att_emotions)) # [seqlen, batch, D_h]
        else:
            alpha = []
            hidden = F.relu(self.linear(outputs)) # [seqlen, batch, D_h]

        return hidden # [seqlen, batch, D_h]


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type()) # [batch, seq_len]

        if self.att_type=='dot':
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # [batch, mem_dim, seqlen]
            x_ = self.transform(x).unsqueeze(1) # [batch, 1, mem_dim]
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2) # [batch, mem_dim, seq_len]
            M_ = M_ * mask_ # [batch, mem_dim, seqlen]
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1) # attention value: [batch, 1, seqlen]
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2) # [batch, 1, seqlen]
            alpha_masked = alpha_*mask.unsqueeze(1) # [batch, 1, seqlen]
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # [batch, 1, 1]
            alpha = alpha_masked/alpha_sum # normalized attention: [batch, 1, seqlen]
            # alpha = torch.where(alpha.isnan(), alpha_masked, alpha) 
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # [batch, 1, seqlen]

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # [batch, mem_dim]
        return attn_pool, alpha
    
def edge_perms(l, window_past, window_future):
    """
    Target:
        Method to construct the edges considering the past and future window.
    
    Input: 
        l: seq length
        window_past, window_future: context lengths

    Output:
        all_perms: all connected edges
    """
    all_perms = set()
    array = np.arange(l)
    for j in range(l): # j: start index
        perms = set()
        
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]
        
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


## accurate graph building process [single relation graph]
def batch_graphify(features, qmask, lengths, n_speakers, window_past, window_future, graph_type, device):
    """
    Target: prepare the data format required for the GCN network.
    Different batches have no edge connection.

    qmask: save speaker items [Batch, Time] !!!! [tensor]
    features: [Time, Batch, ?, Feat] => for node initialization [tensor]
    lengths: each conversation has its own lens [int]
    window_past, window_future: context lens [int]

    'one_nms', 'one_ms', 'two_nms', 'two_ms':
    one/two means one speaker per time; or two speakers per time
    ms/nms means modality-specific and non modality-specific
    """
    ## define edge_type_mapping
    order_types = ['past', 'now', 'future']
    assert n_speakers == 1, 'Note: n_speakers must == 1'
    if n_speakers == 1: speaker_types = ['00']
    
    ## only for single relation graph
    assert graph_type in ['temporal', 'speaker'] 
    merge_types = set()
    if graph_type == 'temporal':
        for ii in range(len(order_types)):
            merge_types.add(f'{order_types[ii]}')
    elif graph_type == 'speaker':
        for ii in range(len(speaker_types)):
            merge_types.add(f'{speaker_types[ii]}')
    
    edge_type_mapping = {}
    for ii, item in enumerate(merge_types):
        edge_type_mapping[item] = ii

    ## qmask to cup()
    qmask = qmask.cpu().data.numpy().astype(int)

    ## build graph
    node_features = []
    edge_index, edge_type = [], []
    length_sum = 0 # for unique node index
    batch_size = features.size(1)
    for j in range(batch_size):
        # gain node_features
        node_feature = features[:int(lengths[j].item()), j, :, :] # [Time, Batch, ?, Feat] -> [Time, ?, Feat]
        node_feature = torch.reshape(node_feature, (-1, node_feature.size(-1))) # [Time*?, Feat]
        node_features.append(node_feature) # [Time*?, Feat]
        
        # make sure different conversations have no connection
        perms1 = edge_perms(int(lengths[j].item()), window_past, window_future)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += int(lengths[j].item()) # add node number [no repeated nodes]
        
        ## change perms1 and perms2
        for item1, item2 in zip(perms1, perms2):

            # gain edge_index [actual edge]
            edge_index.append([item2[0], item2[1]])
            
            # gain edge_type
            (jj, ii) = (item1[0], item1[1])

            ## item1: gain time order
            jj_time = int(jj)
            ii_time = int(ii)
            if ii_time > jj_time:
                order_type = 'past'
            elif ii_time == jj_time:
                order_type = 'now'
            else:
                order_type = 'future'

            jj_speaker = qmask[j, jj_time]
            ii_speaker = qmask[j, ii_time]
            speaker_type = f'{ii_speaker}{jj_speaker}'

            if graph_type == 'speaker':  edge_type_name = f'{speaker_type}'
            if graph_type == 'temporal': edge_type_name = f'{order_type}'
            edge_type.append(edge_type_mapping[edge_type_name])

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.tensor(edge_index).transpose(0, 1)
    edge_type = torch.tensor(edge_type)

    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    
    return node_features, edge_index, edge_type, edge_type_mapping

def utterance_to_conversation(outputs, seq_lengths, umask, device):
    input_conversation_length = seq_lengths.clone().detach() # [6, 24, 13, 9]
    start_zero = input_conversation_length.data.new(1).zero_() # [0]
    
    input_conversation_length = input_conversation_length.to(device)
    start_zero = start_zero.to(device)

    max_len = int(max(seq_lengths))# [int]
    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0) # [0,  6, 30, 43]

    outputs = torch.stack([pad(outputs.narrow(0, int(s), int(l)), max_len, device) # [seqlen, batch, dim]
                                for s, l in zip(start.data.tolist(),
                                input_conversation_length.data.tolist())], 0).transpose(0, 1)
    return outputs


def pad(tensor, length, device):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).to(device)])
        else:
            return var
    else:
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).to(device)])
        else:
            return tensor