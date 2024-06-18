import numpy as np
import torch
import time
import torch.nn as nn

__all__ = ['AlignSubNet']

class AlignSubNet(nn.Module):
    def __init__(self, args, mode):
        """
        mode: the way of aligning
            avg_pool, ctc, conv1d
        """
        super(AlignSubNet, self).__init__()
        assert mode in ['avg_pool', 'ctc', 'conv1d']
        self.args =args
        in_dim_t, in_dim_a, in_dim_v = args.feature_dims
        seq_len_t, seq_len_a, seq_len_v = args.seq_lens
        self.dst_len = seq_len_t
        self.mode = mode

        self.ALIGN_METHODS = {
            'avg_pool': self.__avg_pool,
            'ctc': self.__ctc,
        }

        if mode == 'ctc':
            self.ctc_t = CTCModule(in_dim_t, self.dst_len)
            self.ctc_a = CTCModule(in_dim_a, self.dst_len)
            self.ctc_v = CTCModule(in_dim_v, self.dst_len)

    def get_seq_len(self):
        return self.dst_len
    
    def __ctc(self, text_x, audio_x, video_x, **kwargs):
        # TODO: need verification
        text_x = self.ctc_t(text_x) if text_x.size(1) != self.dst_len else text_x
        audio_x = self.ctc_a(audio_x) if audio_x.size(1) != self.dst_len else audio_x
        video_x = self.ctc_v(video_x) if video_x.size(1) != self.dst_len else video_x
        return text_x, audio_x, video_x

    def __avg_pool(self, text_x, audio_x, video_x, audio_lengths, video_lengths):
        # def _align_2(x, divisor, dividend):
        #     if x.shape[1] == self.dst_len:
        #         return x
        #     pool_size = torch.ceil(dividend / divisor)
        #     dividend = pool_size * divisor
        #     pad_len = ((dividend > x.shape[1]) * (dividend - x.shape[1])).max().cpu()
        #     pad_x = torch.zeros((x.shape[0], int(pad_len), x.shape[2])).to(self.args.device)
        #     x = torch.cat([x, pad_x], dim=1)
        #     res = torch.zeros(x.shape[0], self.dst_len-1, x.shape[2]).to(self.args.device)
        #     for i, item in enumerate(x):
        #         split_index = torch.cumsum(pool_size[i].repeat(self.dst_len-1), dim=0).cpu().long()
        #         split_array = torch.tensor_split(item, split_index, dim=0)[:(self.dst_len-1)]
        #         res[i] = torch.stack(split_array).mean(axis=1)
        #     res = torch.cat([torch.zeros((x.shape[0], 1, x.shape[2])).to(self.args.device), res], dim=1)
        #     return res
        def _align(x,divisor,dividend):
            if x.shape[1] == self.dst_len:
                return x
            divisor[divisor == 0] += 1
            pad_len = divisor - dividend % divisor
            pool_size = dividend // divisor + 1
            tmp_x = torch.zeros(x.shape[0], self.dst_len, x.shape[2]).to(self.args.device)
            for index, item in enumerate(x):
                if pad_len[index] == divisor[index]:
                    pad_len[index] = 0
                    pool_size[index] -= 1 
                pad_x = torch.zeros((int(pad_len[index]), item.shape[1])).to(self.args.device)
                item = item[0:dividend[index],:]
                item = torch.cat([item,pad_x], dim=0).view(int(divisor[index]), int(pool_size[index]), -1).mean(dim=1)
                item = torch.cat([torch.zeros(1, item.shape[1]).to(self.args.device),item], dim=0)
                item = torch.cat([item,torch.zeros(self.dst_len - item.shape[0], item.shape[1]).to(self.args.device)], dim=0)
                tmp_x[index] = item
            return tmp_x
        text_lengths = torch.tensor([t[1].sum() for t in text_x], device=self.args.device)
        # text_x = _align(text_x, text_lengths-2, text_lengths)
        audio_x = _align(audio_x, text_lengths-2, audio_lengths)
        video_x = _align(video_x, text_lengths-2, video_lengths)
        return text_x, audio_x, video_x
    
    def forward(self, text_x, audio_x, video_x, audio_lengths, video_lengths):
        # already aligned
        if text_x.size(1) == audio_x.size(1) == video_x.size(1):
            return text_x, audio_x, video_x
        return self.ALIGN_METHODS[self.mode](text_x, audio_x, video_x, audio_lengths=audio_lengths, video_lengths=video_lengths)

class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len):
        '''
        This module is performing alignment from A (e.g., audio) to B (e.g., text).
        :param in_dim: Dimension for input modality A
        :param out_seq_len: Sequence length for output modality B
        From: https://github.com/yaohungt/Multimodal-Transformer
        '''
        super(CTCModule, self).__init__()
        # Use LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len+1, num_layers=2, batch_first=True) # 1 denoting blank
        
        self.out_seq_len = out_seq_len
        
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''
        # NOTE that the index 0 refers to blank. 
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        prob_pred_output_position_inclu_blank = self.softmax(pred_output_position_inclu_blank) # batch_size x in_seq_len x out_seq_len+1
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :, 1:] # batch_size x in_seq_len x out_seq_len
        prob_pred_output_position = prob_pred_output_position.transpose(1,2) # batch_size x out_seq_len x in_seq_len
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x) # batch_size x out_seq_len x in_dim
        
        # pseudo_aligned_out is regarded as the aligned A (w.r.t B)
        # return pseudo_aligned_out, (pred_output_position_inclu_blank)
        return pseudo_aligned_out
        
