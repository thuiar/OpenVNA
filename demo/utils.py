import json
import logging
import pickle
import shlex
import shutil
import subprocess
from logging.handlers import RotatingFileHandler
from pathlib import Path
from shutil import rmtree

import librosa
import numpy as np
import torch
from transformers import (BertTokenizerFast, Wav2Vec2ForCTC, Wav2Vec2Processor,
                          Wav2Vec2CTCTokenizer)
from ctc_segmentation import (
    CtcSegmentationParameters,
    ctc_segmentation,
    determine_utterance_segments,
    prepare_token_list,
)
# from espnet2.bin.asr_align import CTCSegmentation
# from espnet2.bin.asr_inference import Speech2Text
# from espnet_model_zoo.downloader import ModelDownloader
# from espnet2.tasks.asr import ASRTask

from config import *


def init_logger():
    # The ctc_segmentation module, which runs in a subprocess, calls
    # logging.debug(), which in turn calls logging.basicConfig(), which adds a
    # handler to the root logger if none exists. This is explicitly suggested
    # against in python docs here: https://docs.python.org/3/library/logging.html#logging.basicConfig
    # In this case, it results in duplicate logging messages from an extra
    # handler if we init logger with custom names other than the root logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt = '[%(asctime)s] - %(levelname)s - %(message)s',
        datefmt = "%Y-%m-%d %H:%M:%S"
    )
    fh = RotatingFileHandler(LOG_FILE_PATH, maxBytes=2e7, backupCount=5)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # set numba logger to warning to prevent flooding messages when calling librosa.load()
    logging.getLogger('numba').setLevel(logging.WARNING)
    return logger

def clear_media_folder():
    logger = logging.getLogger()
    logger.info("Cleaning temp files...")
    for path in MEDIA_PATH.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)

def execute_cmd(cmd: str) -> bytes:
    args = shlex.split(cmd)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError("ffmpeg", out, err)
    return out

@torch.no_grad()
def do_asr(
    audio_file : str | Path, 
) -> str:
    try:
        sample_rate = 16000
        speech, _ = librosa.load(audio_file, sr=sample_rate)
        processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_MODEL_NAME).to(DEVICE)
        
        features = processor(
            speech, 
            sampling_rate=sample_rate,
            return_tensors="pt", 
            padding="longest"
        )
        with torch.no_grad():
            logits = model(features.input_values.to(DEVICE)).logits.cpu()[0]

        predicted_ids = torch.argmax(logits, dim=-1)
        asr_text = processor.decode(predicted_ids)
        
        # sample_rate = 16000
        # speech, _ = librosa.load(audio_file, sr=sample_rate)
        # d = ModelDownloader(cachedir="/home/sharing/mhs/espnet_models")
        # # model = d.download_and_unpack("espnet/simpleoier_librispeech_asr_train_asr_conformer7_hubert_ll60k_large_raw_en_bpe5000_sp")
        # model = d.download_and_unpack("espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char")

        # model_infer = Speech2Text(
        #     asr_train_config = model["asr_train_config"],
        #     asr_model_file = model["asr_model_file"],
        #     device = "cpu",
        # )
        # asr_text, *_ = model_infer(speech.squeeze())[0]

        return asr_text
    except Exception as e:
        raise e

@torch.no_grad()
def do_alignment(
    audio_file : str | Path, 
    transcript : str,
) -> list[dict]:
    try:
        speech, _ = librosa.load(audio_file, sr=16000)
        processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL_NAME)
        model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_MODEL_NAME).to(DEVICE)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(WAV2VEC_MODEL_NAME)
        features = processor(
            speech, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="longest"
        )
        with torch.no_grad():
            logits = model(features.input_values.to(DEVICE)).logits.cpu()[0]
            probs = torch.nn.functional.log_softmax(logits,dim=-1)

        # Tokenize transcripts
        transcripts = transcript.split()
        vocab = tokenizer.get_vocab()
        inv_vocab = {v:k for k,v in vocab.items()}
        unk_id = vocab["<unk>"]
        tokens = []
        for transcript in transcripts:
            assert len(transcript) > 0
            tok_ids = tokenizer(transcript.lower())['input_ids']
            tok_ids = np.array(tok_ids,dtype=np.int32)
            tokens.append(tok_ids[tok_ids != unk_id])
        
        # Do align
        char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
        config = CtcSegmentationParameters(char_list=char_list)
        config.index_duration = speech.shape[0] / probs.size()[0] / 16000
        
        ground_truth_mat, utt_begin_indices = prepare_token_list(config, tokens)
        timings, char_probs, state_list = ctc_segmentation(config, probs.numpy(), ground_truth_mat)
        segments = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcripts)
        return [{"text" : t, "start" : p[0], "end" : p[1], "conf" : np.exp(p[2])} for t,p in zip(transcripts, segments)]

        # speech, _ = librosa.load(audio_file, sr=16000)
        # d = ModelDownloader(cachedir="/home/sharing/mhs/espnet_models")
        # # model = d.download_and_unpack("espnet/simpleoier_librispeech_asr_train_asr_conformer7_hubert_ll60k_large_raw_en_bpe5000_sp")
        # model = d.download_and_unpack("espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char")
        # asr_model, asr_train_args = ASRTask.build_model_from_file(
        #     config_file = model["asr_train_config"],
        #     model_file = model["asr_model_file"],
        #     device = "cpu",
        # )
        # speech = torch.tensor(speech).unsqueeze(0).to("cpu")
        # lengths = torch.tensor(speech.shape[1]).unsqueeze(0).to("cpu")
        # enc, _ = asr_model.encode(speech=speech, speech_lengths=lengths)
        # probs = asr_model.ctc.log_softmax(enc).detach().squeeze(0).cpu()

        # char_list = asr_model.token_list[:-1]
        # index_duration = speech.shape[1] / probs.shape[0] / 16000
        # config = CtcSegmentationParameters(
        #     char_list=char_list,
        #     index_duration=index_duration,
        #     # replace_spaces_with_blanks=True,
        #     # blank_transition_cost_zero=True,
        #     # preamble_transition_cost_zero=False,
        # )
        # tokens = []
        # unk = char_list.index("<unk>")
        # for ch in transcript:
        #     token_id = char_list.index(ch)
        #     token_id = np.array(token_id)
        #     tokens.append(token_id[token_id != unk])

        # ground_truth_mat, utt_begin_indices = prepare_token_list(config, tokens)
        # timings, char_probs, state_list = ctc_segmentation(config, probs.numpy(), ground_truth_mat)
        # segments = determine_utterance_segments(config, utt_begin_indices, char_probs, timings, transcript)

        # res = []
        # for i,(t,p) in enumerate(zip(transcript, segments)):
        #     # add 0.11s offset to end timings of chinese characters
        #     res.append({
        #         "text" : t, 
        #         "start" : p[0] if i == 0 else p[0] + 0.11, 
        #         "end" : min(p[1] + 0.11, speech.shape[1]/16000), 
        #         "conf" : np.exp(p[2])
        #     })
        # print(res)
        # return res
    except Exception as e:
        raise e

def data_defence(video_id: str, defence_methods: list[str]) -> bool:
    logger = logging.getLogger()
    modified_video_path = MEDIA_PATH / video_id / "modified_video.mp4"
    defended_video_path = MEDIA_PATH / video_id / "defended_video.mp4"
    defended_video_tmp = MEDIA_PATH / video_id / "defended_video_tmp.mp4"
    defended = False
    shutil.copyfile(modified_video_path, defended_video_path)
    for method in defence_methods:
        if method == "a_denoise":
            logger.info("Audio Denoising...")
            cmd = f"ffmpeg -i {defended_video_path} -af afftdn=nr=40:nf=-20:tn=1 -c:v copy -y {defended_video_tmp}"
            execute_cmd(cmd)
            shutil.copyfile(defended_video_tmp, defended_video_path)
            defended_video_tmp.unlink()
            defended = True
        elif method == "v_reconstruct":
            logger.info("Video MCI...")
            # cmd = f"ffmpeg -i {defended_video_path} -vf blackframe=0,metadata=select:key=lavfi.blackframe.pblack:value=90:function=less,minterpolate=mi_mode=mci -c:a copy -y {defended_video_tmp}"
            cmd = f'ffmpeg -i {defended_video_path} -vf blackdetect=d=0.1:pic_th=0.90:pix_th=0.10 -c:a copy -y {defended_video_tmp}'
            execute_cmd(cmd)
            shutil.copyfile(defended_video_tmp, defended_video_path)
            defended_video_tmp.unlink()
            defended = True
    return defended

def feature_defence(video_id: str, defence_methods: list[str], data_defended: bool, word_ids: list[int]) -> bool:
    modified_feature = MEDIA_PATH / video_id / "feat_modified.pkl"
    defended_feature = MEDIA_PATH / video_id / "feat_defended.pkl"
    video_edit_file = MEDIA_PATH / video_id / "edit_video.json"
    audio_edit_file = MEDIA_PATH / video_id / "edit_audio.json"
    defended = False
    if not data_defended:
        shutil.copyfile(modified_feature, defended_feature)
    for method in defence_methods:
        if method == "f_interpol":
            need_vf_defend = False if "v_reconstruct" in defence_methods else True
            with open(defended_feature, "rb") as f:
                feat = pickle.load(f)
            with open(video_edit_file, "r") as f:
                video_edit = json.load(f)
            with open(audio_edit_file, "r") as f:
                audio_edit = json.load(f)
            v_edit_ids = [v[0] for v in video_edit]
            a_edit_ids = [a[0] for a in audio_edit]
            v_edit_mask = []
            a_edit_mask = []
            for v in word_ids:
                if v in v_edit_ids:
                    v_edit_mask.append(1)
                else:
                    v_edit_mask.append(0)
                if v in a_edit_ids:
                    a_edit_mask.append(1)
                else:
                    a_edit_mask.append(0)
            # audio interpolation
            i, start_idx, end_idx = 0, -1, -1
            while i < len(a_edit_mask):
                if a_edit_mask[i] == 1:
                    start_idx = i - 1
                    while i < len(a_edit_mask) and a_edit_mask[i] == 1:
                        i += 1
                    end_idx = i
                    start_f = feat['audio'][start_idx]
                    end_f = feat['audio'][end_idx]
                    delta = end_f - start_f
                    for idx in range(1, end_idx - start_idx):
                        feat['audio'][start_idx + idx] = start_f + delta * idx / float(end_idx - start_idx)
                i += 1
            # video interpolation
            if need_vf_defend:
                i, start_idx, end_idx = 0, -1, -1
                while i < len(v_edit_mask):
                    if v_edit_mask[i] == 1:
                        start_idx = i - 1
                        while i < len(v_edit_mask) and v_edit_mask[i] == 1:
                            i += 1
                        end_idx = i
                        start_f = feat['vision'][start_idx]
                        end_f = feat['vision'][end_idx]
                        delta = end_f - start_f
                        for idx in range(1, end_idx - start_idx):
                            feat['vision'][start_idx + idx] = start_f + delta * idx / float(end_idx - start_idx)
                    i += 1
            defended = True
    return defended

def get_word_ids(text_file : str, tokenizer : BertTokenizerFast) -> list[int]:
    text = open(text_file, "r").read()
    encoding = tokenizer(text.split(), is_split_into_words=True)
    return encoding.word_ids()

def toTorch(feature, device):
    feature['vision'] = (torch.from_numpy(feature['vision'])).to(device)
    feature['audio'] = (torch.from_numpy(feature['audio'])).to(device)
    feature['text_bert'] = (torch.from_numpy(feature['text_bert'])).to(device)
    feature['vision_lengths'] =(torch.from_numpy(feature['vision_lengths'])).to(device)
    feature['audio_lengths'] = (torch.from_numpy(feature['audio_lengths'])).to(device)
    return feature

def pad_or_truncate(data : dict, text_len : int = 50, audio_len : int = 1432, vision_len : int = 143) -> dict:
    data['audio_lengths'] = data['audio'].shape[0]
    data['vision_lengths'] = data['vision'].shape[0]

    if data['text'].shape[0] > text_len:
        data['text'] = data['text'][:text_len]
        data['text_bert'] = data['text_bert'][:, :text_len]
    elif data['text'].shape[0] < text_len:
        data['text'] = np.pad(data['text'], ((0, text_len - data['text'].shape[0]), (0, 0)), 'constant')
        data['text_bert'] = np.pad(data['text_bert'], ((0, 0), (0, text_len - data['text_bert'].shape[1])), 'constant')

    if data['audio'].shape[0] > audio_len:
        data['audio'] = data['audio'][:audio_len]
        data['audio_lengths']  = audio_len
    elif data['audio'].shape[0] < audio_len:
        data['audio'] = np.pad(data['audio'], ((0, audio_len - data['audio'].shape[0]), (0, 0)), 'constant')

    if data['vision'].shape[0] > vision_len:
        data['vision'] = data['vision'][:vision_len]
        data['vision_lengths']  = vision_len
    elif data['vision'].shape[0] < vision_len:
        data['vision'] = np.pad(data['vision'], ((0, vision_len - data['vision'].shape[0]), (0, 0)), 'constant')

    data['text'] = np.expand_dims(data['text'], axis=0)
    data['text_bert'] = np.expand_dims(data['text_bert'], axis=0)
    data['audio'] = np.expand_dims(data['audio'], axis=0)
    data['vision'] = np.expand_dims(data['vision'], axis=0)

    data['audio_lengths'] = np.array([data['audio_lengths']])
    data['vision_lengths'] = np.array([data['vision_lengths']])
    
    return data
