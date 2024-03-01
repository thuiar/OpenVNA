from model_api.trainers.base_trainer import BaseTrainer

class EMT_DLFR_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        
    def do_valid(self, model, batch_data):
        text = batch_data['text_bert']
        audio = batch_data['audio']
        vision = batch_data['vision'].float()
        vision_lengths = batch_data['vision_lengths']
        audio_lengths = batch_data['audio_lengths']
        
        outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
        return outputs['pred']