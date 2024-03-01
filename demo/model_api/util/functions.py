import json
import numpy as np
import torch 
import random
import pynvml
import math
import logging
from typeguard import typechecked
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger('OpenVNA')

@typechecked
def execute_cmd(cmd: str) -> bytes:
    args = shlex.split(cmd)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError("ffmpeg", out, err)
    return out

@typechecked
def get_audio_length(file : str | Path) -> float:
    """
    Get length of audio stream.

    Args:
        file: Path to media file.

    Returns:
        duration: Audio duration.
    """
    cmd = f"ffprobe -select_streams a -show_entries stream=duration -of json -i {str(file)}"
    out = execute_cmd(cmd)
    prob_result = json.loads(out.decode('utf-8'))
    return float(prob_result['streams'][0]['duration'])

@typechecked
def get_video_length(file : str | Path) -> float:
    """
    Get length of video stream.

    Args:
        file: Path to media file.

    Returns:
        duration: Video duration.
    """
    cmd = f"ffprobe -select_streams v -show_entries stream=duration -of json -i {str(file)}"
    out = execute_cmd(cmd)
    prob_result = json.loads(out.decode('utf-8'))
    return float(prob_result['streams'][0]['duration'])

@typechecked
def get_video_size(file : str | Path) -> tuple[int, int]:
    """
    Get height and width of video stream.

    Args:
        file: Path to media file.

    Returns:
        height: Video height.
        width: Video width.
    """
    cmd = f"ffprobe -select_streams v -show_entries stream=height,width -of json -i {str(file)}"
    out = execute_cmd(cmd)
    prob_result = json.loads(out.decode('utf-8'))
    return prob_result['streams'][0]['height'], prob_result['streams'][0]['width']

@typechecked
def randint_k_order(a: int, b: int, k: int = 2) -> int:
    """
    Return a random integer within the range [a, b] with higher probability for lower numbers.
    
    Args:
        a: Lower bound.
        b: Upper bound.
        k: Order of probability. When k = 2, it has a triangle distribution. For higher values of k it becomes more and more heavily weighted towards a.
    """
    assert a < b
    assert k >= 1
    return int(math.floor(a + (b - a + 1) * (1.0 - random.random()**(1.0 / k))))

def count_parameters(model):
    res = 0
    for p in model.parameters():
        if p.requires_grad:
            res += p.numel()
            # print(p)
    return res

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def assign_gpu(gpu_ids, memory_limit=1e16):
    if len(gpu_ids) == 0 and torch.cuda.is_available():
        # find most free gpu
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        dst_gpu_id, min_mem_used = 0, memory_limit
        for g_id in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        logger.info(f'Found gpu {dst_gpu_id}, used memory {min_mem_used}.')
        gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()
    # logger.info("Let's use %d GPUs!" % len(gpu_ids))
    device = torch.device('cuda:%d' % int(gpu_ids[0]) if using_cuda else 'cpu')
    return device

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def AIRrobustness(y, x= None, dx = 0.1) -> np.float32:
    """
    Compute the absolute and effective robustness of a model given accuracy values under various level of 
    data imperfection, which is the area under the curve of the accuracy-imperfection curve.

    Args:
        y: Accuracy values under various level of data imperfection. Must be a 1-D array.
        x: Data imperfection levels, range from 0 to 1. If None, the sample points are assumed to be evenly 
            spaced along the x-axis. Default is None.
        dx: The spacing between sample points if x is None. Default is 0.1.
    
    Return:
        A float value of absolute robustness.
    """
    abs_robustness = np.trapz(y, x=x, dx=dx)
    eff_robustness = y[0] - abs_robustness
    return abs_robustness, eff_robustness

def audio_pad(raw_item,audio_len=1432):
    if raw_item['audio'].shape[0] > audio_len:
        raw_item['audio_lengths'] = audio_len
        raw_item['audio'] = raw_item['audio'][:audio_len]
    elif raw_item['audio'].shape[0] < audio_len:
        raw_item['audio_lengths'] = raw_item['audio'].shape[0] 
        raw_item['audio'] = np.pad(raw_item['audio'], ((0, audio_len - raw_item['audio'].shape[0]), (0, 0)), 'constant')
    return raw_item

def text_pad(raw_item, text_len=50):
    if raw_item['text_bert'].shape[1] > text_len:
        raw_item['text_lengths'] = text_len
        raw_item['text_bert'] = raw_item['text_bert'][:, :text_len]
        raw_item['text'] = raw_item['text'][:text_len]
    elif raw_item['text_bert'].shape[1] < text_len:
        raw_item['text_lengths'] = raw_item['text_bert'].shape[1]
        raw_item['text_bert'] = np.pad(raw_item['text_bert'], ((0, 0), (0, text_len - raw_item['text_bert'].shape[1])), 'constant')
        raw_item['text'] = np.pad(raw_item['text'], ((0, text_len - raw_item['text'].shape[0]), (0, 0)), 'constant')
    return raw_item

def vision_pad(raw_item,vision_len = 143):
    if raw_item['vision'].shape[0] > vision_len:
        raw_item['vision_lengths'] = vision_len
        raw_item['vision'] = raw_item['vision'][:vision_len]
    elif raw_item['vision'].shape[0] < vision_len:
        raw_item['vision_lengths'] = raw_item['vision'].shape[0]
        raw_item['vision'] = np.pad(raw_item['vision'], ((0, vision_len - raw_item['vision'].shape[0]), (0, 0)), 'constant')
    return raw_item
