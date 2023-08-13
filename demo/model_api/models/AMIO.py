"""
AIO -- All Model in One
"""
from model_api.models.missingTask import *
import torch.nn as nn


__all__ = ['AMIO']

MODEL_MAP = {
    # Proposed Method.
    'tpfn': TPFN,
    't2fn': T2FN,
    'tfr_net': TFR_NET
}

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        lastModel = MODEL_MAP[args.model_name]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x):
        return self.Model(text_x, audio_x, video_x)