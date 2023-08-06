from modal_api.configs import get_config
from modal_api.trainers import get_trainer
from modal_api.trainers import MMIN

def single(mode,feature):
    if mode == 'MMIN':
        return MMIN.robust_test(feature)
    elif mode == "TFR_Net":
        cfg = get_config("TFR_Net", dataset_name = "MOSI")
        cfg.device = "cuda:0"
        cfg.test_model_path = "/home/sharing/disk1/zhangbaozheng/encoder/emnlp/tfr_net/tfr_net-mosi-1111.pth"
        trainer = get_trainer(cfg)
        return trainer.robust_test(feature)
    elif mode == "T2FN":
        cfg = get_config(mode, dataset_name = "MOSI")
        cfg.device = "cuda:0"
        cfg.test_model_path = "/home/sharing/disk1/zhangbaozheng/encoder/emnlp/tfr_net/tfr_net-mosi-1111.pth"
        trainer = get_trainer(cfg)
        return trainer.robust_test(feature)
    elif mode == "TPFN":
        cfg = get_config(mode, dataset_name = "MOSI")
        cfg.device = "cuda:0"
        cfg.test_model_path = "/home/sharing/disk1/zhangbaozheng/encoder/emnlp/tfr_net/tfr_net-mosi-1111.pth"
        trainer = get_trainer(cfg)
        return trainer.robust_test(feature)
