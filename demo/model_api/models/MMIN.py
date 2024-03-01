import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_api.models.subnets.BertTextEncoder import BertTextEncoder


class MMIN(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        # Modality Encoder.
        text_dim, audio_dim, vision_dim = config.feature_dims
        self.text_model = BertTextEncoder(pretrained=config.pretrained_bert_model, finetune=config.finetune_bert)
        self.netA = LSTMEncoder(audio_dim, config.embd_size_a, embd_method=config.embd_method_a)
        self.netL = TextCNN(text_dim, config.embd_size_t)
        self.netV = LSTMEncoder(vision_dim, config.embd_size_v, config.embd_method_v)

        # Training Period one. Classifier 
        cls_layers_p1 = list(map(lambda x: int(x), config.cls_layers_p1.split(',')))
        cls_input_size_p1 = config.embd_size_a + config.embd_size_v + config.embd_size_t
        self.netC1 = FcClassifier(cls_input_size_p1, cls_layers_p1, output_dim=config.output_dim, dropout=config.dropout_rate_p1)

        # Training Period two. AutoEncoder Model
        ae_layers = list(map(lambda x: int(x), config.ae_layers.split(',')))
        ae_input_dim = config.embd_size_a + config.embd_size_v + config.embd_size_t
        self.netAE = ResidualAE(ae_layers, config.n_blocks, ae_input_dim, dropout=0, use_bn=False)
        self.netAE_cycle = ResidualAE(ae_layers, config.n_blocks, ae_input_dim, dropout=0, use_bn=False)
        # Training Period two. Classifier 
        cls_layers_p2 = list(map(lambda x: int(x), config.cls_layers_p2.split(',')))
        cls_input_size_p2 = ae_layers[-1] * config.n_blocks
        self.netC2 = FcClassifier(cls_input_size_p2, cls_layers_p2, output_dim=config.output_dim, dropout=config.dropout_rate_p2)

    def forward(self):
        pass

    def forward_o(self, text, text_reverse, audio, audio_reverse, vision, vision_reverse, mode='train'):
        # get utt level representattion
        feat_A = self.netA(audio)
        feat_T = self.netL(self.text_model(text))
        feat_V = self.netV(vision)
        # fusion miss
        feat_fusion = torch.cat([feat_A, feat_T, feat_V], dim=-1)
        # calc reconstruction of teacher's output
        recon_fusion, latent = self.netAE(feat_fusion)
        recon_cycle, _ = self.netAE_cycle(recon_fusion)
        # get fusion outputs for missing modality
        logits = self.netC(latent)
        self.pred = self.logits

        # for training 
        if mode == 'train':
            with torch.no_grad():
                T_embd_A = self.pretrained_encoder.netA(audio_reverse)
                text_reverse = self.text_model(text_reverse)
                T_embd_L = self.pretrained_encoder.netL(text_reverse)
                T_embd_V = self.pretrained_encoder.netV(vision_reverse)
                T_embds = torch.cat([T_embd_A, T_embd_L, T_embd_V], dim=-1)
        
        loss_mse = F.mse_loss(T_embds, recon_fusion)
        loss_cycle = F.mse_loss(feat_fusion, recon_cycle)

        return logits, loss_mse, loss_cycle


class LSTMEncoder(nn.Module):
    ''' one directional LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, embd_method='last'):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        assert embd_method in ['maxpool', 'attention', 'last']
        self.embd_method = embd_method
        
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)

    def embd_attention(self, r_out, h_n):
        
        hidden_reps = self.attention_layer(r_out)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1]  =  [batch_size, seq_len, hidden_size]
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)       # [batch_size, hidden_size]
        return sentence_vector

    def embd_maxpool(self, r_out, h_n):
        in_feat = r_out.transpose(1,2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)

    def embd_last(self, r_out, h_n):
        return h_n.squeeze(0)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x)
        embd = getattr(self, 'embd_'+self.embd_method)(r_out, h_n)
        return embd


class TextCNN(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, out_channels=128, kernel_heights=[3,4,5], dropout=0.5):
        super().__init__()

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
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
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


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class FcClassifier(nn.Module):
    def __init__(self, input_dim, layers, output_dim, dropout=0.3, use_bn=False):
        super().__init__()

        self.all_layers = []
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            if use_bn:
                self.all_layers.append(nn.BatchNorm1d(layers[i]))
            if dropout > 0:
                self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        
        if len(layers) == 0:
            layers.append(input_dim)
            self.all_layers.append(Identity())
        
        self.fc_out = nn.Linear(layers[-1], output_dim)
        self.module = nn.Sequential(*self.all_layers)
    
    def forward(self, x):
        return self.fc_out(self.module(x))


class ResidualAE(nn.Module):
    def __init__(self, layers, n_blocks, input_dim, dropout=0.5, use_bn=False):
        super(ResidualAE, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.input_dim = input_dim
        self.transition = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        for i in range(n_blocks):
            setattr(self, 'encoder_' + str(i), self.get_encoder(layers))
            setattr(self, 'decoder_' + str(i), self.get_decoder(layers))
    
    def get_encoder(self, layers):
        all_layers = []
        input_dim = self.input_dim
        for i in range(0, len(layers)):
            all_layers.append(nn.Linear(input_dim, layers[i]))
            all_layers.append(nn.LeakyReLU())
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(layers[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
            input_dim = layers[i]
        # delete the activation layer of the last layer
        decline_num = 1 + int(self.use_bn) + int(self.dropout > 0)
        all_layers = all_layers[:-decline_num]
        return nn.Sequential(*all_layers)
    
    def get_decoder(self, layers):
        all_layers = []
        decoder_layer = copy.deepcopy(layers)
        decoder_layer.reverse()
        decoder_layer.append(self.input_dim)
        for i in range(0, len(decoder_layer)-2):
            all_layers.append(nn.Linear(decoder_layer[i], decoder_layer[i+1]))
            all_layers.append(nn.ReLU()) # LeakyReLU
            if self.use_bn:
                all_layers.append(nn.BatchNorm1d(decoder_layer[i]))
            if self.dropout > 0:
                all_layers.append(nn.Dropout(self.dropout))
        
        all_layers.append(nn.Linear(decoder_layer[-2], decoder_layer[-1]))
        return nn.Sequential(*all_layers)

    
    def forward(self, x):
        x_in = x
        x_out = x.clone().fill_(0)
        latents = []
        for i in range(self.n_blocks):
            encoder = getattr(self, 'encoder_' + str(i))
            decoder = getattr(self, 'decoder_' + str(i))
            x_in = x_in + x_out
            latent = encoder(x_in)
            x_out = decoder(latent)
            latents.append(latent)
        latents = torch.cat(latents, dim=-1)
        return self.transition(x_in+x_out), latents
