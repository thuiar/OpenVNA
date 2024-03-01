import torch
from model_api.models.subnets.AlignSubNet import AlignSubNet
from model_api.models.subnets.BertTextEncoder import BertTextEncoder
from model_api.trainers.base_trainer import BaseTrainer

class GCNET_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.alignment_network = AlignSubNet(config, mode='avg_pool').to(config.device)
        self.text_model = BertTextEncoder(pretrained=config.pretrained_bert_model, finetune=config.finetune_bert).to(config.device)
        

    def do_valid(self, model, batch_data):
        vision = batch_data['vision']
        audio = batch_data['audio']
        text = batch_data['text_bert']
        vision_lengths = batch_data['vision_lengths']
        audio_lengths = batch_data['audio_lengths']
        
        # Align audio/vision modality to spoken words.
        text, audio, vision = self.alignment_network(text, audio, vision, 
                                                audio_lengths, vision_lengths)
        # The umask, seq_lengths is corresponding to the text mask (after alignments)
        umask = text[:,1,:].to(self.config.device)
        qmask = torch.zeros_like(umask).to(self.config.device) # This framework considers one speaker scenario only.
        seq_lengths = torch.sum(umask, dim=-1).to(self.config.device)
        
        text = self.text_model(text) # Bert Text Encoder.
        audio, text, vision = audio.transpose(0, 1), text.transpose(0, 1), vision.transpose(0, 1)
        inputfeats = torch.cat([audio, text, vision], dim=-1)

        prediction, _, _ = model(inputfeats, qmask, umask, seq_lengths)
                    
        return prediction