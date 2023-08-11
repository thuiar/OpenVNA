import datetime
import json
import logging
import os
import random
import shutil
from pathlib import Path
from model_api.utils.real_noise import real_noise
# import transformers
from flask import Flask, request
from flask_cors import CORS
from MSA_FET import FeatureExtractionTool, get_default_config
from transformers import BertTokenizerFast

from config import *
from utils import *
from model_api.run import AMIO_MODEL,MMIN_MODEL

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
app.config.from_object(APP_SETTINGS)
app.config['mmin'] = MMIN_MODEL()
app.config['tpfn'] = AMIO_MODEL('tpfn')
app.config['t2fn'] = AMIO_MODEL('t2fn')
app.config['tfr_net'] = AMIO_MODEL('tfr_net')

CORS(app, supports_credentials=True)

logger = logging.getLogger()

@app.route('/test')
def test():
    logger.info("API called: /test")
    return {"code": SUCCESS_CODE, "msg": "success"}


@app.route('/uploadVideo', methods=['POST'])
def upload_video():
    logger.info("API called: /uploadVideo")
    try:
        file = request.files.get('video')
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        video_id = now + str(random.randint(0, 999)).rjust(3, '0')
        save_path = Path(MEDIA_PATH) / video_id
        save_path.mkdir(parents=True, exist_ok=False)
        original_file = save_path / f"original_{file.filename}"
        file.save(original_file)
        # convert to mp4 using ffmpeg
        logger.info(f"Uploaded video, ID: {video_id}")
        logger.info(f"Converting video {video_id} to mp4...")
        cmd = f"ffmpeg -i {original_file} -c:v libx264 -c:a aac -y {save_path / 'raw_video.mp4'}"
        execute_cmd(cmd)
        logger.info("Conversion done.")
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "id": video_id, "path": str(save_path)}


@app.route('/callASR', methods=['POST'])
def call_ASR():
    logger.info("API called: /callASR")
    try:
        video_id = json.loads(request.get_data())['videoID']
        video_file = Path(MEDIA_PATH) / video_id / "raw_video.mp4"
        # extract audio
        audio_save_path = Path(MEDIA_PATH) / video_id / "audio.wav"
        logger.info(f"Extracting audio from {video_id}...")
        cmd = f"ffmpeg -i {video_file} -vn -acodec pcm_s16le -ac 1 -y {audio_save_path}"
        execute_cmd(cmd)
        logger.info("Extraction done.")
        # do ASR
        logger.info(f"Running ASR for {video_id}...")
        transcript = do_asr(audio_save_path)
        logger.info("ASR done.")
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "result": transcript}


@app.route('/uploadTranscript', methods=['POST'])
def upload_transcript():
    logger.info("API called: /uploadTranscript")
    try:
        data = json.loads(request.get_data())
        transcript = data['transcript']
        video_id = data['videoID']
        audio_save_path = MEDIA_PATH / video_id / "audio.wav"
        text_save_path = MEDIA_PATH / video_id / "transcript.txt"
        video_path = MEDIA_PATH / video_id / "raw_video.mp4"
        annotated_video_path = MEDIA_PATH / video_id / "annotated_video.mp4"
        with open(text_save_path, 'w') as f:
            f.write(transcript)
        # do alignment
        logger.info(f"Running alignment for {video_id}...")
        # aligned_results = do_alignment(audio_save_path, transcript, WAV2VEC_PROCESSER, WAV2VEC_TOKENIZER, WAV2VEC_MODEL)
        aligned_results = do_alignment(audio_save_path, transcript)
        with open(MEDIA_PATH / video_id / "aligned_results.json", 'w') as f:
            json.dump(aligned_results, f)
        logger.info("Alignment done.")
        # annotate video with OpenFace
        # logger.info("Annotating video with OpenFace...")
        # res = openfaceExtractor.get_annotated_video(str(video_path), str(annotated_video_path))
        # logger.debug(res) # BUG: OpenFace won't set return code >0 on error
        # logger.info("Annotation done.")
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "align": aligned_results}

@app.route('/videoEditAligned', methods=['POST'])
def edit_video_aligned():
    logger.info("API called: /videoEditAligned")
    try:
        data = json.loads(request.get_data())
        video_id = data['videoID']
        words = data['words']
        asrNosie = data['asrNosie']
        raw_video_path = MEDIA_PATH / video_id / "raw_video.mp4"
        modified_video_path = MEDIA_PATH / video_id / "modified_video.mp4"
        video_edit_file = MEDIA_PATH / video_id / "edit_video.json"
        audio_edit_file = MEDIA_PATH / video_id / "edit_audio.json"
        
        with open(MEDIA_PATH / video_id / "aligned_results.json") as f:
            aligned_results = json.load(f)
        # re-organize
        video_modify, audio_modify, text_modify = {
            'v_mode': [], 'v_start': [], 'v_end': [], 'v_option': []}, {'a_mode': [], 'a_start': [], 'a_end': [], 'a_option': []}, []
        for word in words:
            if word[1] == 'v':
                video_modify['v_mode'].append(word[2])
                video_modify['v_start'].append(float(word[0][0]))
                video_modify['v_end'].append(float(word[0][1]))
                video_modify['v_option'].append(word[3])
            elif word[1] == 'a':
                if word[2] == 'mute':
                    word[2] = 'volume'
                    word[3] = 0
                audio_modify['a_mode'].append(word[2])
                audio_modify['a_start'].append(float(word[0][0]))
                audio_modify['a_end'].append(float(word[0][1]))
                audio_modify['a_option'].append(word[3])
            elif word[1] == 't':
                text_modify.append([word[0], word[2], word[3]])
        # dump edit to files, to be used by feature interpolation
        with open(video_edit_file, 'w') as f:
            json.dump(video_modify, f)
        with open(audio_edit_file, 'w') as f:
            json.dump(audio_modify, f)
        
        if len(video_modify['v_mode']) or len(audio_modify['a_mode']):
            real_noise(
                in_file=raw_video_path,
                out_file=modified_video_path,
                mode="exact",
                v_mode=video_modify['v_mode'],
                v_start=video_modify['v_start'],
                v_end=video_modify['v_end'],
                v_option=video_modify['v_option'],
                a_mode=audio_modify['a_mode'],
                a_start=audio_modify['a_start'],
                a_end=audio_modify['a_end'],
                a_option=audio_modify['a_option']
            )
        else:
            shutil.copyfile(raw_video_path, modified_video_path)

        if len(audio_modify['a_mode']) and asrNosie:
            # extract audio
            audio_save_path = Path(MEDIA_PATH) / video_id / "audio.wav"
            logger.info(f"Extracting audio from {video_id}...")
            cmd = f"ffmpeg -i {modified_video_path} -vn -acodec pcm_s16le -ac 1 -y {audio_save_path}"
            execute_cmd(cmd)
            logger.info("Extraction done.")
            # do ASR
            logger.info(f"Running ASR for {video_id}...")
            transcript = do_asr(audio_save_path)
            text_save_path = MEDIA_PATH / video_id / "transcript.txt"
            with open(text_save_path, 'w') as f:
                f.write(transcript)

            # do alignment
            logger.info(f"Running alignment for {video_id}...")
            aligned_results = do_alignment(audio_save_path, transcript)
            logger.info("Alignment done.")
        else:
            # edit text
            for t in text_modify:
                word_id = t[0][2]
                if t[1] == 'replace':
                    aligned_results[word_id]['text'] = t[2]
                elif t[1] == 'remove':
                    aligned_results[word_id]['text'] = 'REMOVED'  # [UNK]
        
        transcript = " ".join([w['text'] for w in aligned_results])
        with open(MEDIA_PATH / video_id / "transcript_modified.txt", 'w') as f:
            f.write(transcript)
        with open(MEDIA_PATH / video_id / "aligned_modified.json", 'w') as f:
            json.dump(aligned_results, f)
    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "align": aligned_results}


@app.route('/runMSAAligned', methods=['POST'])
def run_msa_aligned():
    logger.info("API called: /runMSAAligned")
    try:
        data = json.loads(request.get_data())
        video_id = data['videoID']
        models = data['models']
        defence = data['defence']
        video_original_path = MEDIA_PATH / video_id / "raw_video.mp4"
        video_modified_path = MEDIA_PATH / video_id / "modified_video.mp4"
        video_defended_path = MEDIA_PATH / video_id / "defended_video.mp4"
        feat_original_path = MEDIA_PATH / video_id / "feat_original.pkl"
        feat_modified_path = MEDIA_PATH / video_id / "feat_modified.pkl"
        feat_defended_path = MEDIA_PATH / video_id / "feat_defended.pkl"
        trans_original_path = MEDIA_PATH / video_id / "transcript.txt"
        trans_modified_path = MEDIA_PATH / video_id / "transcript_modified.txt"
        weights_root_path = Path(__file__).parent / "assets" / "weights"
        # init
        cfg = get_default_config('aligned')
        cfg['text']['device'] = DEVICE
        cfg['align']['device'] = DEVICE
        cfg['video']['fps'] = 30
        cfg['video']['args'] = {
            "hogalign": False,
            "simalign": False,
            "nobadaligned": False,
            "landmark_2D": True,
            "landmark_3D": False,
            "pdmparams": False,
            "head_pose": False,
            "action_units": True,
            "gaze": False,
            "tracked": False
        }
        fet = FeatureExtractionTool(cfg, verbose=0)
        bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        # data defence
        logger.info(f"Data-Level Defence for {video_id}...")
        data_defended = data_defence(video_id, defence)

        # extract original features
        logger.info(f"Extracting features from original video {video_id}...")
        f_original = fet.run_single(
            video_original_path, feat_original_path, text_file=trans_original_path)

        # extract modified features
        logger.info(f"Extracting features from modified video {video_id}...")
        f_modified = fet.run_single(
            video_modified_path, feat_modified_path, text_file=trans_modified_path)

        # extract defended features
        if data_defended:
            logger.info(
                f"Extracting features from defended video {video_id}...")
            fet.run_single(video_defended_path, feat_defended_path,
                           text_file=trans_modified_path)
        # explain original features
        word_ids_original = get_word_ids(trans_original_path, bert_tokenizer)
        last_word_id = -1
        remove_ids = []
        for i, v in enumerate(word_ids_original):
            if v is None:  # start or end token
                continue
            if v == last_word_id:  # same word, multiple tokens
                remove_ids.append(i)
            last_word_id = v
        for m in ['audio', 'vision', 'text']:
            # remove repeated feature
            f_original[m] = np.delete(f_original[m], remove_ids, axis=0)
            f_original[m] = f_original[m][1:-1]  # remove start and end token
        # explain modified & defended features
        word_ids_modified = get_word_ids(trans_modified_path, bert_tokenizer)
        # feature defence
        logger.info(f"Feature-Level Defence for {video_id}...")
        feature_defended = feature_defence(
            video_id, defence, data_defended, word_ids_modified)
        last_word_id = -1
        remove_ids = []
        for i, v in enumerate(word_ids_modified):
            if v is None:  # start or end token
                continue
            if v == last_word_id:  # same word, multiple tokens
                remove_ids.append(i)
            last_word_id = v
        for m in ['audio', 'vision']:
            f_modified[m] = np.delete(f_modified[m], remove_ids, axis=0)
            f_modified[m] = f_modified[m][1:-1]
            if data_defended or feature_defended:
                with open(feat_defended_path, "rb") as f:
                    f_defended = pickle.load(f)
                f_defended[m] = np.delete(f_defended[m], remove_ids, axis=0)
                f_defended[m] = f_defended[m][1:-1]

        feat = {
            "original": {
                "loudness": f_original['audio'][:, 0].tolist(),
                "alphaRatio": f_original['audio'][:, 1].tolist(),
                "mfcc1": f_original['audio'][:, 6].tolist(),
                "mfcc2": f_original['audio'][:, 7].tolist(),
                "mfcc3": f_original['audio'][:, 8].tolist(),
                "mfcc4": f_original['audio'][:, 9].tolist(),
                "pitch": f_original['audio'][:, 10].tolist(),
                "HNR": f_original['audio'][:, 13].tolist(),
                "F1frequency": f_original['audio'][:, 16].tolist(),
                "F1bandwidth": f_original['audio'][:, 17].tolist(),
                "F1amplitude": f_original['audio'][:, 18].tolist(),
                "AU01_r": f_original['vision'][:, -35].tolist(),
                "AU02_r": f_original['vision'][:, -34].tolist(),
                "AU04_r": f_original['vision'][:, -33].tolist(),
                "AU05_r": f_original['vision'][:, -32].tolist(),
                "AU06_r": f_original['vision'][:, -31].tolist(),
                "AU07_r": f_original['vision'][:, -30].tolist(),
                "AU09_r": f_original['vision'][:, -29].tolist(),
                "AU10_r": f_original['vision'][:, -28].tolist(),
                "AU12_r": f_original['vision'][:, -27].tolist(),
                "AU14_r": f_original['vision'][:, -26].tolist(),
                "AU15_r": f_original['vision'][:, -25].tolist(),
                "AU17_r": f_original['vision'][:, -24].tolist(),
                "AU20_r": f_original['vision'][:, -23].tolist(),
                "AU23_r": f_original['vision'][:, -22].tolist(),
                "AU25_r": f_original['vision'][:, -21].tolist(),
                "AU26_r": f_original['vision'][:, -20].tolist(),
                "AU45_r": f_original['vision'][:, -19].tolist(),
            },
            "modified": {
                "loudness": f_modified['audio'][:, 0].tolist(),
                "alphaRatio": f_modified['audio'][:, 1].tolist(),
                "mfcc1": f_modified['audio'][:, 6].tolist(),
                "mfcc2": f_modified['audio'][:, 7].tolist(),
                "mfcc3": f_modified['audio'][:, 8].tolist(),
                "mfcc4": f_modified['audio'][:, 9].tolist(),
                "pitch": f_modified['audio'][:, 10].tolist(),
                "HNR": f_modified['audio'][:, 13].tolist(),
                "F1frequency": f_modified['audio'][:, 16].tolist(),
                "F1bandwidth": f_modified['audio'][:, 17].tolist(),
                "F1amplitude": f_modified['audio'][:, 18].tolist(),
                "AU01_r": f_modified['vision'][:, -35].tolist(),
                "AU02_r": f_modified['vision'][:, -34].tolist(),
                "AU04_r": f_modified['vision'][:, -33].tolist(),
                "AU05_r": f_modified['vision'][:, -32].tolist(),
                "AU06_r": f_modified['vision'][:, -31].tolist(),
                "AU07_r": f_modified['vision'][:, -30].tolist(),
                "AU09_r": f_modified['vision'][:, -29].tolist(),
                "AU10_r": f_modified['vision'][:, -28].tolist(),
                "AU12_r": f_modified['vision'][:, -27].tolist(),
                "AU14_r": f_modified['vision'][:, -26].tolist(),
                "AU15_r": f_modified['vision'][:, -25].tolist(),
                "AU17_r": f_modified['vision'][:, -24].tolist(),
                "AU20_r": f_modified['vision'][:, -23].tolist(),
                "AU23_r": f_modified['vision'][:, -22].tolist(),
                "AU25_r": f_modified['vision'][:, -21].tolist(),
                "AU26_r": f_modified['vision'][:, -20].tolist(),
                "AU45_r": f_modified['vision'][:, -19].tolist(),
            }
        }
        if data_defended or feature_defended:
            feat['defended'] = {
                "loudness": f_defended['audio'][:, 0].tolist(),
                "alphaRatio": f_defended['audio'][:, 1].tolist(),
                "mfcc1": f_defended['audio'][:, 6].tolist(),
                "mfcc2": f_defended['audio'][:, 7].tolist(),
                "mfcc3": f_defended['audio'][:, 8].tolist(),
                "mfcc4": f_defended['audio'][:, 9].tolist(),
                "pitch": f_defended['audio'][:, 10].tolist(),
                "HNR": f_defended['audio'][:, 13].tolist(),
                "F1frequency": f_defended['audio'][:, 16].tolist(),
                "F1bandwidth": f_defended['audio'][:, 17].tolist(),
                "F1amplitude": f_defended['audio'][:, 18].tolist(),
                "AU01_r": f_defended['vision'][:, -35].tolist(),
                "AU02_r": f_defended['vision'][:, -34].tolist(),
                "AU04_r": f_defended['vision'][:, -33].tolist(),
                "AU05_r": f_defended['vision'][:, -32].tolist(),
                "AU06_r": f_defended['vision'][:, -31].tolist(),
                "AU07_r": f_defended['vision'][:, -30].tolist(),
                "AU09_r": f_defended['vision'][:, -29].tolist(),
                "AU10_r": f_defended['vision'][:, -28].tolist(),
                "AU12_r": f_defended['vision'][:, -27].tolist(),
                "AU14_r": f_defended['vision'][:, -26].tolist(),
                "AU15_r": f_defended['vision'][:, -25].tolist(),
                "AU17_r": f_defended['vision'][:, -24].tolist(),
                "AU20_r": f_defended['vision'][:, -23].tolist(),
                "AU23_r": f_defended['vision'][:, -22].tolist(),
                "AU25_r": f_defended['vision'][:, -21].tolist(),
                "AU26_r": f_defended['vision'][:, -20].tolist(),
                "AU45_r": f_defended['vision'][:, -19].tolist(),
            }
        # run MSA models
        logger.info(f"Running MSA models for {video_id}...")
        res = {
            "original": {},
            "modified": {},
        }
        if data_defended or feature_defended:
            res["defended"] = {}

        audio_fet = FeatureExtractionTool(config="model_api/config/opensmile.json")
        vision_fet = FeatureExtractionTool(config="model_api/config/openface.json")

        with open(feat_defended_path, "rb") as f:
            original_bert = pickle.load(f)
        original_feature = {
            "text": original_bert['text'],
            "text_bert": original_bert['text_bert'],
            "audio": (audio_fet.run_single(video_original_path))['audio'],
            "vision": (vision_fet.run_single(video_original_path))['vision']
        }
        original_feature = pad_or_truncate(original_feature)

        with open(feat_defended_path, "rb") as f:
            modified_bert = pickle.load(f)
        modified_feature = {
            "text": modified_bert['text'],
            "text_bert": modified_bert['text_bert'],
            "audio": (audio_fet.run_single(video_modified_path))['audio'],
            "vision": (vision_fet.run_single(video_modified_path))['vision']
        }
        modified_feature = pad_or_truncate(modified_feature)

        if data_defended or feature_defended:
            with open(feat_defended_path, "rb") as f:
                defended_bert = pickle.load(f)
            defended_feature = {
                "text": defended_bert['text'],
                "text_bert": defended_bert['text_bert'],
                "audio": (audio_fet.run_single(video_defended_path))['audio'],
                "vision": (vision_fet.run_single(video_defended_path))['vision']
            }
            defended_feature = pad_or_truncate(defended_feature)

        for m in models:
            if data_defended or feature_defended:
                for key in original_feature:
                    original_feature[key] = np.concatenate((original_feature[key], modified_feature[key], defended_feature[key]), axis=0)
                r = app.config[str.lower(m)].eval(original_feature)
                res["original"][m] = float(r[0])
                res["modified"][m] = float(r[1])
                res["defended"][m] = float(r[2])
            else:
                for key in original_feature:
                    original_feature[key] = np.concatenate((original_feature[key], modified_feature[key]), axis=0)
                r = app.config[str.lower(m)].eval(original_feature)
                res["original"][m] = float(r[0])
                res["modified"][m] = float(r[1])

    except Exception as e:
        logger.exception(e)
        return {"code": ERROR_CODE, "msg": str(e)}
    return {"code": SUCCESS_CODE, "msg": "success", "result": res, "feature": feat}
