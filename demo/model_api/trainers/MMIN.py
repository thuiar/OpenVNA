import torch
from model_api.trainers.base_trainer import BaseTrainer

class MMIN_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def do_valid(self, model, batch_data, phase='phase_two'):
        vision = batch_data['vision'].float()
        audio = batch_data['audio']
        text = batch_data['text_bert']
        if phase == 'phase_one':
            feat_A = model.netA(audio)
            feat_T = model.netL(model.text_model(text))
            feat_V = model.netV(vision)
            prediction = model.netC1(torch.cat([feat_A, feat_T, feat_V], dim=-1))
        else:
            feat_A = model.netA(audio)
            feat_T = model.netL(model.text_model(text))
            feat_V = model.netV(vision)
            _, latent = model.netAE(torch.cat([feat_A, feat_T, feat_V], dim=-1))
            prediction = model.netC2(latent)
        return prediction
