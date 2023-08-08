from pathlib import Path

WEB_SERVER_PORT = 4096
LOG_FILE_PATH = Path("logs/Robust-MSA.log")
MEDIA_PATH = Path("media")
MEDIA_SERVER_PORT = 8192
# DEVICE = torch.device("cuda")
DEVICE = "cuda"
CUDA_VISIBLE_DEVICES = "0"

# custom status codes
ERROR_CODE = 400
SUCCESS_CODE = 200

WAV2VEC_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

# APP_SETTINGS = 'config.ProductionConfig'
APP_SETTINGS = 'config.DevelopmentConfig'

class DevelopmentConfig(object):
    DEBUG = True
    JSON_AS_ASCII = False # Chinese

class ProductionConfig(object):
    DEBUG = False
    JSON_AS_ASCII = False # Chinese