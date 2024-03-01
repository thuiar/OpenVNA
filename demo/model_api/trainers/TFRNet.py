from model_api.trainers.base_trainer import BaseTrainer

class TFRNet_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
    
    def do_valid(self, model, batch_data):
        vision = batch_data['vision'].float()
        audio = batch_data['audio']
        text = batch_data['text_bert']
        vision_lengths = batch_data['vision_lengths']
        audio_lengths = batch_data['audio_lengths']
        prediction = model((text, None), (audio, None), (vision, None), audio_lengths, vision_lengths, mode='eval')
        return prediction
