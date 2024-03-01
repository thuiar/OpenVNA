from model_api.trainers.base_trainer import BaseTrainer

class NIAT_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
     
    def do_valid(self, model, batch_data):
        vision = batch_data['vision']
        audio = batch_data['audio']
        text = batch_data['text_bert']
        vision_lengths = batch_data['vision_lengths']
        audio_lengths = batch_data['audio_lengths']
        text, audio, vision = model.alignment_network(text, audio, vision, 
                                                audio_lengths, vision_lengths)
        fusion_feature = model.fusion(text, audio, vision)
        prediction = model.classifier(fusion_feature)
        return prediction
