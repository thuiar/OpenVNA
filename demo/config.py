from pathlib import Path

WEB_SERVER_PORT = 4096
LOG_FILE_PATH = Path("logs/OpenVNA.log")
MEDIA_PATH = Path("media")
MEDIA_SERVER_PORT = 8192
# DEVICE = torch.device("cuda")
DEVICE = "cuda"
CUDA_VISIBLE_DEVICES = "0"

MODEL_PATH = "/home/sharing/disk1/zhangbaozheng/encoder/emnlp"

# custom status codes
ERROR_CODE = 400
SUCCESS_CODE = 200

WAV2VEC_MODEL_NAME = "/home/sharing/disk1/zhangbaozheng/code/Robust-MSA/model_api/assets/jonatasgrosman/wav2vec2-large-xlsr-53-english"
PRETRAINED_MODEL = '/home/zhangbaozheng/paper_code/Robust_Framework/pretrained_model/bert_en'

# APP_SETTINGS = 'config.ProductionConfig'
APP_SETTINGS = 'config.DevelopmentConfig'

class DevelopmentConfig(object):
    DEBUG = True
    JSON_AS_ASCII = False # Chinese

class ProductionConfig(object):
    DEBUG = False
    JSON_AS_ASCII = False # Chinese