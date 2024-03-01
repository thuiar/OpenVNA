import torch
from model_api.trainers.base_trainer import BaseTrainer

class CTFN_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
  
    def do_valid(self, model, batch_data):
        vision = batch_data['vision']
        audio = batch_data['audio']
        text = batch_data['text_bert']
        vision_lengths =batch_data['vision_lengths']
        audio_lengths = batch_data['audio_lengths']

        text, audio, vision = model.align_subnet(text, audio, vision, 
                                                audio_lengths, vision_lengths)
        
        text = model.text_model(text)

        a2t_fake_a, a2t_fake_t, bimodal_at, bimodal_ta = self.trans_fusion(model.a2t_model, None, audio, text, need_grad=False)
        audio_a2t = self.specific_modal_fusion(audio, a2t_fake_a, bimodal_ta)
        text_a2t = self.specific_modal_fusion(text, a2t_fake_t, bimodal_at)
        a2v_fake_a, a2v_fake_v, bimodal_av, bimodal_va = self.trans_fusion(model.a2v_model, None, audio, vision, need_grad=False)
        audio_a2v = self.specific_modal_fusion(audio, a2v_fake_a, bimodal_va)
        vision_a2v = self.specific_modal_fusion(vision, a2v_fake_v, bimodal_av)
        v2t_fake_v, v2t_fake_t, bimodal_vt, bimodal_tv = self.trans_fusion(model.v2t_model, None, vision, text, need_grad=False)
        vision_v2t = self.specific_modal_fusion(vision, v2t_fake_v, bimodal_tv)
        text_v2t = self.specific_modal_fusion(text, v2t_fake_t, bimodal_vt)

        prediction = model.sa_model(audio, text, vision, audio_a2t, text_a2t, vision_v2t, text_v2t, audio_a2v, vision_a2v)
    
        return prediction
    
    def train_transnet(self, trans_net, optimizer, source, target):
        optimizer.zero_grad()
        fake_target, _ = trans_net[0](source)
        recon_source, _ = trans_net[1](fake_target)
        g_loss = torch.mean((source-recon_source)**2)
        g_loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        fake_target, _ = trans_net[1](target)
        recon_source, _ = trans_net[0](fake_target)
        g_loss = torch.mean((target-recon_source)**2)
        g_loss.backward()
        optimizer.step()

    def trans_fusion(self, trans_net, optimizer, source, target, need_grad=True):
        if need_grad:
            optimizer.zero_grad()
            fake_target, bimodal_12 = trans_net[0](source)
            fake_source, bimodal_21 = trans_net[1](target)

        else:
            trans_net.eval()
            with torch.no_grad():
                fake_target, bimodal_12 = trans_net[0](source)
                fake_source, bimodal_21 = trans_net[1](target)

        return fake_source, fake_target, bimodal_12, bimodal_21

    def specific_modal_fusion(self, true_data, fake_data, mid_data):
        alphas = torch.sum(torch.abs(true_data - fake_data), (1, 2))
        alphas_sum = torch.sum(alphas)
        alphas = torch.div(alphas, alphas_sum).unsqueeze(-1).unsqueeze(-1)
        return torch.mul(alphas, mid_data[-1])