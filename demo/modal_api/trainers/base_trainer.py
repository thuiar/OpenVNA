import torch
from modal_api.configs.base_config import BaseConfig
from modal_api.models import get_model
from modal_api.models.subnets.AlignSubNet import AlignSubNet
import numpy as np
class BaseTrainer(object):
    """
    This is an example of a trainer class.
    Either inherit this class and override the functions or copy, paste and modify the functions of your own class.
    """
    def __init__(self, config: BaseConfig) -> None:
        self.config = config
        self.model = get_model(config).to(config.device)
        self.model.eval()
        self.align_net = None
        if self.config.feature_aligned:
            if self.config.seq_lens[0] == self.config.seq_lens[1] == self.config.seq_lens[2]:
                self.align_net = None
            else:
                self.align_net = AlignSubNet(config, "avg_pool")

    @torch.no_grad()
    def test(self, feature) -> dict:
        """
        Test after training one seed.
        Override this function if needed.
        """        
        text, audio, vision, audio_lengths, vision_lengths = \
            self.prepare_batch_data(feature)
        if self.align_net:
            text, audio, vision = self.align_net(text, audio, vision, audio_lengths, vision_lengths)
        outputs = self.model(text, audio, vision)
        return outputs
    
    def prepare_batch_data(self, batch_data):
        """
        Preprocess batch data.
        Override this function if the model requires additional data.
        """
        text = torch.from_numpy(batch_data['text_bert']).to(self.config.device)
        audio = torch.from_numpy(batch_data['audio']).float().to(self.config.device)
        vision = torch.from_numpy(batch_data['vision']).float().to(self.config.device)

        if self.align_net:
            audio_lengths = torch.from_numpy(np.array([batch_data['audio_lengths']])).to(self.config.device)
            vision_lengths = torch.from_numpy(np.array([batch_data['vision_lengths']])).to(self.config.device)
        else:
            audio_lengths, vision_lengths = None, None
        return text, audio, vision, audio_lengths, vision_lengths

    def load_model(self, load_path=None):
        if load_path is None:
            load_path = self.model_save_path
        self.model.load_state_dict(torch.load(load_path, map_location=self.config.device), strict=False)
   
    @torch.no_grad()
    def robust_test(self,feature):
        self.load_model(self.config.test_model_path)
        self.model.to(self.config.device)
        test_results = self.test(feature)
        test_results = round(float(test_results.detach().cpu().numpy()[0][0]), 4)
        return test_results
