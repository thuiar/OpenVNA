from typing import Literal

import numpy as np
from typeguard import typechecked

@typechecked
def feature_noise(
    mode: Literal["feat_random_drop", "feat_structural_drop"],
    text: np.ndarray,       # (batch, 3, seq_len)
    audio: np.ndarray,      # (batch, seq_len, feat_dim)
    vision: np.ndarray,     # (batch, seq_len, feat_dim)
    audio_lengths,
    vision_lengths,
    missing_rate,
    seeds,
):
    if mode == "feat_random_drop":
        assert audio_lengths is not None and vision_lengths is not None
        if type(missing_rate) == np.float64 or type(missing_rate) == float:
            assert missing_rate >= 0 and missing_rate <= 1
            missing_rate = [missing_rate] * 3
        elif type(missing_rate) == list:
            assert len(missing_rate) == 3
            assert all([mr >= 0 and mr <= 1 for mr in missing_rate])
        input_mask_text = text[:,1,:]
        input_len_text = np.sum(input_mask_text, axis=1).astype(np.int32)
        input_mask_audio = np.array([np.array([1] * length + [0] * (audio.shape[1] - length)) for length in audio_lengths])
        input_len_audio = np.array(audio_lengths)
        input_mask_vision = np.array([np.array([1] * length + [0] * (vision.shape[1] - length)) for length in vision_lengths])
        input_len_vision = np.array(vision_lengths)
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        text_mask_multiseed, audio_mask_multiseed, vision_mask_multiseed = None, None, None

        for missing_seed in seeds:
            np.random.seed(missing_seed)
            missing_mask_t = (np.random.uniform(size=input_mask_text.shape) > missing_rate[0]) * input_mask_text
            missing_mask_a = (np.random.uniform(size=input_mask_audio.shape) > missing_rate[1]) * input_mask_audio
            missing_mask_v = (np.random.uniform(size=input_mask_vision.shape) > missing_rate[2]) * input_mask_vision

            # ensure CLS and SEP tokens are not missing
            missing_mask_t[:,0] = 1
            missing_mask_t[np.arange(text.shape[0]), input_len_text-1] = 1

            text_m = missing_mask_t * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask_text - missing_mask_t) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask_a, axis=2) * audio
            vision_m = np.expand_dims(missing_mask_v, axis=2) * vision

            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1)

            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)

            text_mask_multiseed = missing_mask_t if text_mask_multiseed is None else np.concatenate((text_mask_multiseed, missing_mask_t), axis=0)
            audio_mask_multiseed = missing_mask_a if audio_mask_multiseed is None else np.concatenate((audio_mask_multiseed, missing_mask_a), axis=0)
            vision_mask_multiseed = missing_mask_v if vision_mask_multiseed is None else np.concatenate((vision_mask_multiseed, missing_mask_v), axis=0)

        return text_m_multiseed, audio_m_multiseed, vision_m_multiseed, text_mask_multiseed, audio_mask_multiseed, vision_mask_multiseed
    
    elif mode == "feat_structural_drop":
        assert audio_lengths is not None and vision_lengths is not None
        if type(missing_rate) == float:
            assert missing_rate >= 0 and missing_rate <= 1
            missing_rate = [missing_rate] * 3
        if type(missing_rate) == list:
            assert len(missing_rate) == 3
            assert all([mr >= 0 and mr <= 1 for mr in missing_rate])
        input_mask_text = text[:,1,:]
        input_len_text = np.sum(input_mask_text, axis=1).astype(np.int32)
        input_mask_audio = np.array([np.array([1] * length + [0] * (audio.shape[1] - length)) for length in audio_lengths])
        input_len_audio = np.array(audio_lengths)
        input_mask_vision = np.array([np.array([1] * length + [0] * (vision.shape[1] - length)) for length in vision_lengths])
        input_len_vision = np.array(vision_lengths)
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        text_mask_multiseed, audio_mask_multiseed, vision_mask_multiseed = None, None, None

        # Calculate the length of missing blocks.
        missing_block_t_len = np.around((input_len_text - 2) * missing_rate).astype(np.int32)
        missing_block_a_len = np.around(input_len_audio * missing_rate).astype(np.int32)
        missing_block_v_len = np.around(input_len_vision * missing_rate).astype(np.int32)
        missing_mask_t = input_mask_text.copy()
        missing_mask_a = input_mask_audio.copy()
        missing_mask_v = input_mask_vision.copy()
        for missing_seed in seeds:
            np.random.seed(missing_seed)

            for i, _ in enumerate(missing_mask_t):
                start_p = np.random.randint(low=0, high=input_len_text[i] - missing_block_t_len[i])
                missing_mask_t[i, start_p:start_p+missing_block_t_len[i]] = 0
            
            for i, _ in enumerate(missing_mask_a):
                if input_len_audio[i] > missing_block_a_len[i]:
                    start_p = np.random.randint(low=0, high=input_len_audio[i] - missing_block_a_len[i])
                    missing_mask_a[i, start_p:start_p+missing_block_a_len[i]] = 0
                else:
                    missing_mask_a[i,:] = 0

            for i, _ in enumerate(missing_mask_v):
                if input_len_vision[i] > missing_block_v_len[i]:
                    start_p = np.random.randint(low=0, high=input_len_vision[i] - missing_block_v_len[i])
                    missing_mask_v[i, start_p:start_p+missing_block_v_len[i]] = 0
                else:
                    missing_mask_v[i,:] = 0
            text_m = missing_mask_t * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask_text - missing_mask_t) # UNK token: 100.
            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1) 
            audio_m = np.expand_dims(missing_mask_a, axis=2) * audio
            vision_m = np.expand_dims(missing_mask_v, axis=2) * vision
            
            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)
        return text_m_multiseed, audio_m_multiseed, vision_m_multiseed, text_mask_multiseed, audio_mask_multiseed, vision_mask_multiseed
    
    else:   # temporal_drop
        raise NotImplementedError