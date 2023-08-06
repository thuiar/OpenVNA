from typing import Literal

import numpy as np
from typeguard import typechecked


class staticNoise():

    def __init__(self, config: dict):

        self.seed_list = config['seeds_list']
        self.config = config
        self.strategy_map = {
            'temporal_feature_missing': self.__FRAME_DROP,
            'static_block_drop': self.__BLOCK_DROP,
        }
        self.process_func = self.strategy_map[config['noise_type']]

    def __FRAME_DROP(self, text, vision, audio):
        """ config: missing_rate.
        """
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        mask_multiseed = None

        for missing_seed in self.seed_list:
            np.random.seed(missing_seed)
            missing_mask = (np.random.uniform(size=input_mask.shape) > self.config['missing_rate']) * input_mask
            
            assert missing_mask.shape == input_mask.shape

            for i, instance in enumerate(missing_mask):
                instance[0] = instance[input_len[i] - 1] = 1
            
            text_m = missing_mask * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask, axis=2) * audio
            vision_m = np.expand_dims(missing_mask, axis=2) * vision

            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1)

            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)
            
            mask_multiseed = missing_mask if mask_multiseed is None else np.concatenate((mask_multiseed, missing_mask), axis=0)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, mask_multiseed, mask_multiseed, mask_multiseed

    def __BLOCK_DROP(self, text, vision, audio):
        """ config: missing_rate
        """
        input_mask = text[:,1,:]
        input_len = np.argmin(np.concatenate((input_mask, np.zeros((input_mask.shape[0], 1))), axis=1), axis=1) # 防止mask全一导致长度为0
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        mask_multiseed = None
        
        missing_block_len = np.around((input_len - 2) * self.config['missing_rate']).astype(np.int)
        missing_mask = input_mask.copy()

        for missing_seed in self.seed_list:
            np.random.seed(missing_seed)
            
            # 构造 missing_mask 方法不同，frame_drop 后续操作均一致.
            for i, instance in enumerate(missing_mask):
                start_p = np.random.randint(low=1, high=input_len[i] - missing_block_len[i])
                missing_mask[i, start_p:start_p+missing_block_len[i]] = 0
            
            text_m = missing_mask * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask) # UNK token: 100.
            audio_m = np.expand_dims(missing_mask, axis=2) * audio
            vision_m = np.expand_dims(missing_mask, axis=2) * vision

            text_m = np.concatenate((np.expand_dims(text_m, 1), text[:,1:,:]), axis=1) 

            text_m_multiseed = text_m if text_m_multiseed is None else np.concatenate((text_m_multiseed, text_m), axis=0)
            audio_m_multiseed = audio_m if audio_m_multiseed is None else np.concatenate((audio_m_multiseed, audio_m), axis=0)
            vision_m_multiseed = vision_m if vision_m_multiseed is None else np.concatenate((vision_m_multiseed, vision_m), axis=0)
            
            mask_multiseed = missing_mask if mask_multiseed is None else np.concatenate((mask_multiseed, missing_mask), axis=0)

        return text_m_multiseed, vision_m_multiseed, audio_m_multiseed, mask_multiseed, mask_multiseed, mask_multiseed


@typechecked
def feature_noise(
    mode: Literal["temporal_drop", "random_drop_aligned", "random_drop_unaligned"],
    text: np.ndarray,       # (batch, 3, seq_len)
    audio: np.ndarray,      # (batch, seq_len, feat_dim)
    vision: np.ndarray,     # (batch, seq_len, feat_dim)
    audio_lengths: np.ndarray | list,
    vision_lengths: np.ndarray | list,
    missing_rate: float | list[float],
    seeds: list[int],
):
    if mode == "random_drop_aligned":
        assert audio.shape[1] == vision.shape[1] == text.shape[2]
        assert type(missing_rate) == float and missing_rate >= 0 and missing_rate <= 1
        input_mask = text[:,1,:]
        input_len = np.sum(input_mask, axis=1)
        text_m_multiseed, audio_m_multiseed, vision_m_multiseed = None, None, None
        text_mask_multiseed, audio_mask_multiseed, vision_mask_multiseed = None, None, None

        for missing_seed in seeds:
            np.random.seed(missing_seed)
            missing_mask_t = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask
            missing_mask_a = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask
            missing_mask_v = (np.random.uniform(size=input_mask.shape) > missing_rate) * input_mask

            # ensure CLS and SEP tokens are not missing
            missing_mask_t[:,0] = 1
            missing_mask_a[:,0] = 1
            missing_mask_v[:,0] = 1
            missing_mask_t[np.arange(text.shape[0]), input_len-1] = 1
            missing_mask_a[np.arange(audio.shape[0]), input_len-1] = 1
            missing_mask_v[np.arange(vision.shape[0]), input_len-1] = 1

            text_m = missing_mask_t * text[:,0,:] + (100 * np.ones_like(text[:,0,:])) * (input_mask - missing_mask_t) # UNK token: 100.
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
    elif mode == "random_drop_unaligned":
        assert audio_lengths is not None and vision_lengths is not None
        if type(missing_rate) == float:
            assert missing_rate >= 0 and missing_rate <= 1
            missing_rate = [missing_rate] * 3
        if type(missing_rate) == list:
            assert len(missing_rate) == 3
            assert all([mr >= 0 and mr <= 1 for mr in missing_rate])
        input_mask_text = text[:,1,:]
        input_len_text = np.sum(input_mask_text, axis=1).astype(np.int)
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
            missing_mask_a[:,0] = 1
            missing_mask_v[:,0] = 1
            missing_mask_t[np.arange(text.shape[0]), input_len_text-1] = 1
            missing_mask_a[np.arange(audio.shape[0]), input_len_audio-1] = 1
            missing_mask_v[np.arange(vision.shape[0]), input_len_vision-1] = 1

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
    else:   # temporal_drop
        assert type(missing_rate) == float and missing_rate >= 0 and missing_rate <= 1