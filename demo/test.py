from model_api.run import AMIO_MODEL
import pickle
import numpy as np

feat_original_path = "/home/sharing/disk3/Datasets/MMSA-Standard/MOSI/Processed/unaligned_v171_a25_l50.pkl"
with open(feat_original_path, 'rb') as f:
    feature = pickle.load(f)

single_feature = {
    "text_bert":feature['test']['text_bert'][0:2],
    "text":feature['test']['text'][0:2],
    "vision":feature['test']['vision'][0:2],
    "audio":feature['test']['audio'][0:2],
    "audio_lengths":np.array(feature['test']['audio_lengths'][0:2]),
    "vision_lengths":np.array(feature['test']['vision_lengths'][0:2]),
}

model = AMIO_MODEL('TFR_Net')

result = model.eval(single_feature)
print()