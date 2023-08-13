import torch
import argparse
import numpy as np
from model_api.models.AMIO import AMIO
from model_api.config.config_regression import ConfigRegression
from config import MODEL_PATH,DEVICE,CUDA_VISIBLE_DEVICES
from model_api.config.MMIN import Options
from model_api.models import create_model
import numpy as np

class MMIN_MODEL():
    def __init__(self):
        seed = 1111
        opt = Options().parse(1111)  
        test_model_path = f"{MODEL_PATH}/{str.lower(opt.model)}/{seed}"
        opt.isTrain = False
        self.model = create_model(opt)      # create a model given opt.model and other options
        self.model.setup(opt)               # regular setup: load and print networks; create schedulers
        self.model.load_networks(test_model_path)
        self.model.eval()

    def eval(self,feature):
        self.model.set_input(feature)
        self.model.test()
        pred = np.around(self.model.pred.detach().cpu().numpy().squeeze(),decimals=4)
        return pred

class AMIO_MODEL():
    def __init__(self,mode):
        self.args = parse_args()
        self.args.model_name = str.lower(mode)
        self.args.model_path = MODEL_PATH
        config = ConfigRegression(self.args)
        self.args = config.get_config()
        self.args.use_bert_finetune = False
        # device
        if DEVICE == 'cuda':
            self.args.device = torch.device(f'{DEVICE}:{CUDA_VISIBLE_DEVICES}')
        else:
            self.args.device = torch.device(DEVICE)

        # load pretrained model
        self.args.model_save_path = f"{self.args.model_path}/{self.args.model_name}/{self.args.model_name}-{self.args.dataset_name}-{self.args.seed}.pth"
        self.model = AMIO(self.args).to(self.args.device)
        self.model.load_state_dict(torch.load(self.args.model_save_path))
        self.model.to(self.args.device)

    def eval(self,batch_data):
        audio = torch.from_numpy(batch_data['audio']).float().to(self.args.device)
        vision = torch.from_numpy(batch_data['vision']).float().to(self.args.device)
        text = torch.from_numpy(batch_data['text_bert']).to(self.args.device)

        text_lengths = np.argmin(np.concatenate((batch_data['text_bert'][:,1,:], np.zeros((batch_data['text_bert'][:,1,:].shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_lengths = torch.from_numpy(text_lengths).to(self.args.device)
        audio_lengths = torch.from_numpy(batch_data['audio_lengths']).to(self.args.device)
        vision_lengths = torch.from_numpy(batch_data['vision_lengths']).to(self.args.device)
        
        outputs = self.model((text,text_lengths), (audio,audio_lengths), (vision,vision_lengths))

        outputs = np.around(outputs.detach().cpu().numpy().squeeze(),decimals=4)
        return outputs
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='tfr_net',
                        help='support tfr_net/t2fn/tpfn')
    parser.add_argument('--dataset_name', type=str, default='mosi',
                        help='support mosi/mosei')
    parser.add_argument('--seed', type=int, default=1111,
                        help='')
    return parser.parse_args()

if __name__ == '__main__':
    pass
