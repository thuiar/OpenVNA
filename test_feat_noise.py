from noise_api.feature_noise import feature_noise
import pickle
import numpy as np


# with open("test/feature_aligned.pkl", "rb") as f:
#     data = pickle.load(f)
# text, vision, audio = data["text_bert"], data["vision"], data["audio"]
# text, vision, audio = np.expand_dims(text, 0), np.expand_dims(vision, 0), np.expand_dims(audio, 0)

with open("noise_api/examples/feature_unaligned.pkl", "rb") as f:
    data = pickle.load(f)

text, vision, audio, vision_lengths, audio_lengths = np.expand_dims(data["text_bert"], 0), \
    np.expand_dims(data["vision"], 0), np.expand_dims(data["audio"], 0), [data['vision'].shape[0]], [data['audio'].shape[0]]

text_m, audio_m, vision_m, *_ = feature_noise("random_drop_unaligned", text, audio, vision, audio_lengths, vision_lengths, 0.1, [1,2,3])