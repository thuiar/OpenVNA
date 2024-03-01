import torch
import argparse
import pickle
from model_api.configs import *
from model_api.models import get_model
from model_api.trainers import get_trainer
from config import MODEL_PATH, CUDA_VISIBLE_DEVICES

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MOSI', choices=['MOSI'],
                        help='Video Understanding Dataset Name.')
    return parser.parse_args()

class ModelApi():
    def __init__(self) -> None:
        super(ModelApi, self).__init__()
        models = ['T2FN', 'TPFN', 'CTFN', 'MMIN', 'TFRNet', 'GCNET', 'NIAT', 'EMT_DLFR']
        args = parse_args()
        args.device = [CUDA_VISIBLE_DEVICES]

        for model in models:
            args.model = model
            setattr(self, f'{str.lower(model)}_config', get_config(**vars(args)))
            setattr(self, f'{str.lower(model)}_model', get_model(getattr(self,f'{str.lower(model)}_config')))
            setattr(self, f'{str.lower(model)}_trainer', get_trainer(getattr(self,f'{str.lower(model)}_config')))
            (getattr(self,f'{str.lower(model)}_model')).load_state_dict(torch.load(f"{MODEL_PATH}/{str.lower(args.dataset)}_{str.lower(model)}.pth"))
            (getattr(self, f'{str.lower(model)}_model')).to(self.t2fn_config.device)
            (getattr(self, f'{str.lower(model)}_model')).eval()

    def run_t2fn(self, data):
        results = self.t2fn_trainer.do_valid(self.t2fn_model, data)
        return results.cpu().detach().numpy().tolist()
    
    def run_tpfn(self, data):
        results = self.tpfn_trainer.do_valid(self.tpfn_model, data)
        return results.cpu().detach().numpy().tolist()
    
    def run_mmin(self, data):
        results = self.mmin_trainer.do_valid(self.mmin_model, data)
        return results.cpu().detach().numpy().tolist()
    
    def run_ctfn(self, data):
        results = self.ctfn_trainer.do_valid(self.ctfn_model, data)
        return results.cpu().detach().numpy().tolist()
    
    def run_tfrnet(self, data):
        results = self.tfrnet_trainer.do_valid(self.tfrnet_model, data)
        return results.cpu().detach().numpy().tolist()
    
    def run_gcnet(self, data):
        results = self.gcnet_trainer.do_valid(self.gcnet_model, data)
        return results.cpu().detach().numpy().tolist()
    
    def run_niat(self, data):
        results = self.niat_trainer.do_valid(self.niat_model, data)
        return results.cpu().detach().numpy().tolist()
    
    def run_emt_dlfr(self, data):
        results = self.emt_dlfr_trainer.do_valid(self.emt_dlfr_model, data)
        return results.cpu().detach().numpy().tolist()
  
if __name__ == "__main__":
    model = ModelApi()
    with open(f'/home/sharing/disk1/zhangbaozheng/code/Robust-MSA/model_api/assets/temp/example.pkl', "rb") as f:
        batch_data = pickle.load(f)
    
    