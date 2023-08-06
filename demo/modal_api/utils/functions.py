import json
import logging
import math
import random
import shlex
import subprocess
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
import numpy as np
import torch
from typeguard import typechecked


@typechecked
def set_logger(
    log_dir: Path = Path("logs"), 
    verbose_level: int = 1
) -> logging.Logger:
    # base logger
    logger = logging.getLogger('Robust-MSA') 
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # file handler
    fh = RotatingFileHandler(log_dir / 'Robust-MSA.log', maxBytes=2e7, backupCount=2)
    fh_formatter = logging.Formatter(
            fmt= '%(asctime)s - %(name)s [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    return logger


@typechecked
def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@typechecked
def execute_cmd(cmd: str) -> bytes:
    args = shlex.split(cmd)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError("ffmpeg", out, err)
    return out


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

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)