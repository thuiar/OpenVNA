import logging
import random
import shutil
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from typeguard import typechecked

from model_api.utils.functions import (execute_cmd, get_audio_length, get_video_length,
                        get_video_size, randint_k_order, set_logger)

logger = set_logger(verbose_level=1)


class RealNoiseConfig(NamedTuple):
    mode: Literal["percent", "exact"] = "percent"
    v_mode: list[Literal["blank", "avgblur", "gblur", "impulse_value",
                            "occlusion", "color", "color_channel_swapping", "color_inversion"]] = ['occlusion']
    v_start: list[float] = [0.0]
    v_end: list[float] = [1.0]
    v_option: list = [(0.3, 0.3, 0.4, 0.4)]
    a_mode: list[Literal["volume", "mute", "coloran", "background", "sudden", "lowpass", "reverb"]] = ['background']
    a_start: list[float] = [0.0]
    a_end: list[float] = [1.0]
    a_option: list = [('random', 1.0)]


@typechecked
def real_noise(
    in_file: str | Path,
    out_file: str | Path,
    mode: Literal["percent", "exact"] = "percent",
    v_mode: list[Literal["blank", "avgblur", "gblur", "impulse_value",
                            "occlusion", "color", "color_channel_swapping", "color_inversion"]] = [],
    v_start: list[float] = [],
    v_end: list[float] = [],
    v_option: list = [],
    a_mode: list[Literal["volume", "mute", "coloran", "background", "sudden", "lowpass", "reverb"]] = [],
    a_start: list[float] = [],
    a_end: list[float] = [],
    a_option: list = [],
    seed: int | None = None,
) -> None:
    """
    Add real-world visual and acoustic noise.

    The configuration of noise is given as list of mode, start time, end time, and option. The length of list must be same for each parameter. 
    Use `real_noise_config` function to generate configurations for this function.
    If use this function directly, please read options below.

    Args:
        in_file (str | Path): Input video file path.
        out_file (str | Path): Output video file path.
        mode (str): "percent" or "exact". Whether start and end time are given as percentage exact timestamp in seconds. This also affects some of the options.
        v_mode (list[str]): List of visual noise mode. Supported modes: "blank", "avgblur", "gblur", "impulse_value", "occlusion", "color", "color_channel_swapping", "color_inversion".
        v_start (list[float]): List of start time of visual noise.
        v_end (list[float]): List of end time of visual noise.
        v_option (list): List of option for visual noise. See "Options" below for details.
        a_mode (list[str]): List of acoustic noise mode. Supported modes: "volume", "coloran", "background", "sudden", "lowpass", "reverb"
        a_start (list[float]): List of start time of acoustic noise.
        a_end (list[float]): List of end time of acoustic noise.
        a_option (list): List of option for acoustic noise. See "Options" below for details.
        seed (int | None): Random seed. If None, will not set seed explicitly.

    Options:
        Blank: 
            No options required. 
        avgBlur:
            (SizeX, SizeY) of the avgBlur filter.
            SizeX, SizeY: Set horizontal and vertical radius size.
        gBlur: 
            Sigma of Gaussian blur. 
        Impulse valued noise (salt and pepper noise):
            Noise strength for specific pixel component。
        Occlusion: 
            (x, y, w, h) of occlusion box. If mode is "percent", values are given as percentage of video size. Otherwise, they are given as pixels.
        Color:
            (contrast, brightness, saturation, gamma_r, gamma_g, gamma_b) of color filter. 
            contrast: [-2.0, 2.0]. FFmpeg default: 1.0
            brightness: [-1.0, 1.0]. FFmpeg default: 0.0
            saturation: [0.0, 3.0]. FFmpeg default: 1.0
            gamma_r: [0.1, 10.0]. FFmpeg default: 1.0
            gamma_g: [0.1, 10.0]. FFmpeg default: 1.0
            gamma_b: [0.1, 10.0]. FFmpeg default: 1.0
        Color Channel Swapping:
            (channel_1, channel_2) of two channels to be swapped. Supported values are ["r", "g", "b"].
            Currently only support swapping two channels.
        Color Inversion:
            No options required.
        Volume: 
            Volume range from 0 to 1. Values greater than 1 will amplify the audio. 
        Low Pass:
            Frequency of low pass filter, given in Hz. 
        Color Audio Noise:
            (color, amplitude) of the constructed noise.
            color: Which color noise is applied to the audio. Supported Values are ["white", "pink", "brown", "blue", "violet", "grey", "brown", "violet"].
            amplitude: Specify the amplitude (0.0 - 1.0) of the generated audio stream. Default value is 1.0.
        Background: 
            (filename, volume) of background noise. Filename is the name of background noise file. Supported Values are ["random", "metro", "office", "park", "restaurant", "traffic", "white", "music_soothing", "music_tense", "song_gentle", "song_rock"]. 
            If filename is "random", a random background noise will be selected. Volume is a float number greater than 0. 
            Background noise is applied to the specified time range. Currently, the background noise is not looped. Thus if the noise is shorter, it will not cover the whole time range. A random part of the noise will be selected if the noise file is longer.
            Most noise files are about 5 minutes long, except for the music and songs which are about 3 minutes long.
        Sudden:
            (filename, volume) of sudden noise. Filename is the name of sudden noise file. Supported Values are ["random", "beep", "glass", "thunder", "dog"]. 
            If filename is "random", a random background noise will be selected. Volume is a float number greater than 0. 
            Sudden noise is applied at the specified timestamp. It will not be looped and the end time is ignored. 
        Reverb:
            (filename, length) of IR files. Reverb is done by applying Frequency Impulse Response filter. Filename is the name of IR file. Supported Values are ["hall", "room"]. 
            Length is the Impulse Response filter length range from 0 to 1. A value of 1 means whole IR is processed. Note that the length value does not equal to the strength of reverb. It depends on the waveform of the IR file, which is usually non-linear.
            The Reverb effect is applied to the whole audio, thus the start and end time is ignored. 
    """

    # assertions
    assert len(v_mode) == len(v_start) == len(v_end) == len(v_option)
    assert len(a_mode) == len(a_start) == len(a_end) == len(a_option)
    if mode == "percent":
        assert all([0.0 <= s <= 1.0 for s in v_start])
        assert all([0.0 <= e <= 1.0 for e in v_end])
        assert all([0.0 <= s <= 1.0 for s in a_start])
        assert all([0.0 <= e <= 1.0 for e in a_end])
    elif mode == "exact":
        assert all([0.0 <= s for s in v_start])
        assert all([0.0 <= e for e in v_end])
        assert all([0.0 <= s for s in a_start])
        assert all([0.0 <= e for e in a_end])
    assert Path(in_file).is_file()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    tmp_file_path = Path(out_file).parent / "tmp"
    tmp_file_path.mkdir(parents=True, exist_ok=True)
    background_noise_path = Path(__file__).parent.parent / "assets" / "noise" / "background"
    sudden_noise_path = Path(__file__).parent.parent / "assets" / "noise" / "sudden"
    reverb_path = Path(__file__).parent.parent / "assets" / "noise" / "IRs"
    assert Path(background_noise_path).is_dir()
    assert Path(sudden_noise_path).is_dir()

    video_length = get_video_length(in_file)
    video_height, video_width = get_video_size(in_file)
    if mode == "percent":
        for i in range(len(v_start)):
            v_start[i] *= video_length
            v_end[i] *= video_length
        for i in range(len(a_start)):
            a_start[i] *= video_length
            a_end[i] *= video_length

    if seed is not None:
        random.seed(seed)
    
    # video noise
    if len(v_mode) > 0:
        cmd = f"/usr/bin/ffmpeg -i {str(in_file)} -filter_complex \"[0:v]"
        for i, (m, s, e, o) in enumerate(zip(v_mode, v_start, v_end, v_option)):
            if i != 0:
                cmd += f"[v{i-1}]"
            if m == "blank": # Video Noise: blank screen
                cmd += f"drawbox=enable='between(t,{s},{e})':color=black:t=fill[v{i}];"
            elif m == "avgblur": # Video Noise
                cmd += f"avgblur=sizeX={o[0]}:sizeY={o[1]}:enable='between(t,{s},{e})'[v{i}];"
            elif m == "gblur": # Video Noise: Gaussian blur
                cmd += f"gblur=sigma={o}:enable='between(t,{s},{e})'[v{i}];"
            elif m == "impulse_value": # Video Noise: impulse_value noise.
                cmd += f"noise=alls={o}:allf=t:enable='between(t,{s},{e})'[v{i}];"
            elif m == "occlusion": # Video Noise: occlusion
                if mode == "percent":
                    cmd += f"drawbox=enable='between(t,{s},{e})':x={o[0]*video_width}:y={o[1]*video_height}:w={o[2]*video_width}:h={o[3]*video_height}:color=black:t=fill[v{i}];"
                else:
                    cmd += f"drawbox=enable='between(t,{s},{e})':x={o[0]}:y={o[1]}:w={o[2]}:h={o[3]}:color=black:t=fill[v{i}];"
            elif m == "color": # Video Noise: color error
                cmd += f"eq=contrast={o[0]}:brightness={o[1]}:saturation={o[2]}:gamma_r={o[3]}:gamma_g={o[4]}:gamma_b={o[5]}:enable='between(t,{s},{e})'[v{i}];"
            elif m == "color_channel_swapping": # Video Noise: color channel error
                cmd += f"colorchannelmixer={o[0]}{o[0]}=0:{o[0]}{o[1]}=1:{o[1]}{o[0]}=1:{o[1]}{o[1]}=0:enable='between(t,{s},{e})'[v{i}];"
            elif m == "color_inversion": # Video Noise: color invertion
                cmd += f"negate=negate_alpha=1:enable='between(t,{s},{e})'[v{i}];"

        cmd = cmd.rstrip(";") + f"\" -map \"[v{len(v_mode)-1}]\" "
    else:
        cmd = f"/usr/bin/ffmpeg -i {str(in_file)} -c:v copy -map 0:v "
    
    # audio noise
    additive_noise = []
    if len(a_mode) > 0:
        cmd_a = f"-filter_complex \"[0:a]"
        i = 0
        for m, s, e, o in zip(a_mode, a_start, a_end, a_option):
            if m == "volume": # Audio Noise: volume
                if i != 0:
                    cmd_a += f"[a{i-1}]"
                cmd_a += f"volume=enable='between(t,{s},{e})':volume={o}[a{i}];"
                i += 1
            elif m == "lowpass": # Audio Noise: low pass filter
                if i != 0:
                    cmd_a += f"[a{i-1}]"
                cmd_a += f"lowpass=f={o}:enable='between(t,{s},{e})'[a{i}];"
                i += 1
            elif m == "coloran":
                if o[0] == "random":
                    rc = random.choice(["white", "pink", "brown", "blue", "violet", "grey", "brown", "violet"])
                else:
                    rc = o[0]
                cmd_nc = f"/usr/bin/ffmpeg -f lavfi -i anoisesrc=d={e-s}:c={rc}:r=16000:a={o[1]} -y {tmp_file_path}/an_{len(additive_noise)}.wav"
                logger.debug(cmd_nc)
                execute_cmd(cmd_nc)
                additive_noise.append((m, s, e, o))
            elif m == "background" or m == "sudden" or m == "reverb": # requires additional audio inputs
                additive_noise.append((m, s, e, o))
        if len(a_mode) == len(additive_noise):
            cmd_a = f"-c:a copy -map 0:a -y {str(out_file)}"
        else:
            cmd_a = cmd_a.rstrip(";") + f"\" -map \"[a{i-1}]\" -y {str(out_file)}"
        cmd += cmd_a
    else:
        cmd += f"-c:a copy -map 0:a -y {str(out_file)}"
    # execute
    logger.debug(cmd)
    execute_cmd(cmd)

    # additive noise
    if len(additive_noise) > 0:
        shutil.copyfile(str(out_file), str(out_file) + ".tmp")
        cmd = f"/usr/bin/ffmpeg -i {str(out_file) + '.tmp'} "
        # inputs
        noise_file_list = []
        for i, (m, s, e, o) in enumerate(additive_noise):
            if m == "coloran":
                cmd += f"-i {str(Path(tmp_file_path, f'an_{i}.wav'))} "
            if m == "background":
                if o[0] == "random":
                    noise_file = random.choice(list(Path.iterdir(background_noise_path)))
                else:
                    noise_file = Path(background_noise_path) / (o[0]+".wav")
                assert noise_file.is_file()
                noise_file_list.append(noise_file)
                cmd += f"-i {str(noise_file)} "
            if m == "sudden":
                if o[0] == "random":
                    noise_file = random.choice(list(Path.iterdir(sudden_noise_path)))
                else:
                    noise_file = Path(sudden_noise_path) / (o[0]+".wav")
                assert Path(noise_file).is_file()
                noise_file_list.append(noise_file)
                cmd += f"-i {str(noise_file)} "
            if m == "reverb":
                ir_file = Path(reverb_path) / (o[0]+".wav")
                noise_file_list.append(ir_file)
                cmd += f"-i {str(ir_file)} "
        # resample, trim & delay
        cmd += f"-filter_complex \"[0:a]aresample=16000[resample0];"
        for i, (m, s, e, o) in enumerate(additive_noise):
            if m == "coloran":
                cmd += f"[{i+1}:a]aresample=16000,adelay={s*1000}|{s*1000}[resample{i+1}];"
            if m == "background":
                audio_length = get_audio_length(noise_file_list[i])
                if audio_length < e-s:
                    cmd += f"[{i+1}:a]aresample=16000,adelay={s*1000}|{s*1000}[resample{i+1}];"
                else:
                    rnd_start = random.uniform(0, audio_length - (e-s))
                    cmd += f"[{i+1}:a]aresample=16000,atrim={rnd_start}:{rnd_start+e-s},adelay={s*1000}|{s*1000}[resample{i+1}];"
            elif m == "sudden":
                cmd += f"[{i+1}:a]aresample=16000,adelay={s*1000}|{s*1000}[resample{i+1}];"
            elif m == "reverb":
                cmd += f"[{i+1}:a]aresample=16000[resample{i+1}];"
        cmd += "[resample0]"
        # mix & fir
        for i, (m, s, e, o) in enumerate(additive_noise):
            if i != 0:
                cmd += f"[a{i-1}]"
            if m == "coloran":
                cmd += f"[resample{i+1}]amix=inputs=2:duration=first:dropout_transition=0[a{i}];"
            if m == "background":
                cmd += f"[resample{i+1}]amix=inputs=2:duration=first:weights='1 {o[1]}':dropout_transition=0[a{i}];"
            elif m == "sudden":
                cmd += f"[resample{i+1}]amix=inputs=2:duration=first:weights='1 {o[1]}':dropout_transition=0[a{i}];"
            elif m == "reverb":
                cmd += f"[resample{i+1}]afir=dry=10:wet=10:length={o[1]}[a{i}];"
        cmd = cmd.rstrip(";") + f"\" -map 0:v -c:v copy -map \"[a{len(additive_noise)-1}]\" -y {str(out_file)}"
        # execute
        logger.debug(cmd)
        execute_cmd(cmd)
        Path(str(out_file) + '.tmp').unlink(missing_ok=True)
        shutil.rmtree(tmp_file_path)

@typechecked
def real_noise_config(
    in_file: str | Path,
    mode: Literal["random_full", "random_time", "word_random", "word_manual"] = "random_full",
    v_noise_list: list[Literal["blank", "avgblur", "gblur", "impulse_value", "occlusion", "color",
                                "rgb2bgr", "color_inversion"]] = ["gblur", "impulse_value", "occlusion"],
    v_noise_num: int | tuple[int, int] = (1, 3),
    v_noise_ratio: float | tuple[float, float] = (0.1, 0.5),
    v_noise_intensity: float | tuple[float, float] | list = (0.5, 1.0),
    a_noise_list: list[Literal["volume", "mute", "colorna", "lowpass", "background", "sudden",
                                "reverb"]] = ["mute", "background", "sudden"],
    a_noise_num: int | tuple[int, int] = (1, 3),
    a_noise_ratio: float | tuple[float, float] = (0.1, 0.5),
    a_noise_intensity: float | tuple[float, float] | list = (0.8, 1.0),
    word_table: dict | None = None,
    words: list[str] = [],
    seed: int | None = None,
) -> RealNoiseConfig:
    """
    Generate configs for `real_noise` function. See "Mode Options" below for detailed usage. 

    General Args:
        in_file (str | Path): Input file path.
        mode (str): Mode of config generation. See below for details.
        seed (int | None): Random seeds. If None, will not set seed explicitly.

    Mode Options:
        "random_full": 
            Randomly select from given noise types and apply to random parts of the video. 
            "v/a_noise_list": List of noise types to be randomly chosen from.
            "v/a_noise_num": Number of noise to be applied in total. If a two-tuple is given, a random number will be selected within the range.
            "v/a_noise_ratio": Ratio of video/audio length to be covered by noise. If a two-tuple is given, a random number will be selected within the range.
            "v/a_noise_intensity": A fixed float between 0 and 1, or a tuple of two floats indicating a range of intensity to be randomly selected from. See "Noise Intensity" below for details.
        "random_time":
            Apply given noise types to random parts of the video. 
            "v/a_noise_list": List of noise types to be applied.
            "v/a_noise_num": Ignored. For noise num is the length of noise list.
            "v/a_noise_ratio": Ratio of video/audio length to be covered by noise. If a two-tuple is given, a random number will be selected within the range.
            "v/a_noise_intensity": List of intensities for each noise applied, or a single value or a random range. See "Noise Intensity" below for details.
        "word_random":
            Apply random noise types to specific words in the video. (Not Implemented)
            "word_table": Word table containing start/end time of each word spoken in the video. 
            "words": (index, mode) Index of words in "word_table" to be applied with noise. mode can be "a", "v" or "av".
            "v/a_noise_list": List of noise types to be randomly chosen from.
            "v/a_noise_intensity": A fixed float between 0 and 1, or a tuple of two floats indicating a range of intensity to be randomly selected from. See "Noise Intensity" below for details.
        "word_manual":
            Apply given noise types to specific words in the video. (Not Implemented)
            "word_table": Word table containing start/end time of each word spoken in the video. 
            "words": List of words to be applied with noise.
            "v/a_noise_list": List of noise types to be applied. Should be the same length as "words".
            "v/a_noise_intensity": List of intensities for each noise applied, or a single value or a random range. See "Noise Intensity" below for details.
    
    Noise Intensity:
        The intensity of noise indicates the strength of the noise. Its value ranges from 0 to 1. Each type of noise has its own definition of intensity. 
        "blank": Not affected.
        "avgblur": SizeX = SizeY = 8 * intensity + 1.
        "gblur": Sigma = 20 * intensity. 
        "impulse_value": Noise strength = 20 * intensity.
        "occlusion": Width/height of occlusion box = video width/height * intensity.
        "color": Contrast = 1 ± intensity. Brightness = ±0.5 * intensity. Saturation = 1 ± intensity. Only one of the three parameters will be applied each time.
        "color_channel_swapping": Not affected.
        "color_inversion": Not affected.
        "volume": Volume = (1 - intensity).
        "mute": Not affected.
        "coloran": Specify the amplitude (0.0 - 1.0) of the generated color noise audio stream. 
        "lowpass": Frequency = 1000 * (1 - intensity).
        "background": Volume = 0.5 + 2* intensity.
        "sudden": Volume = 0.5 + 2* intensity.
        "reverb": Impulse Response filter length = intensity ** 2. See "Options" in `real_noise` function for details about the length value.
    """
    # assertions
    assert Path(in_file).is_file()
    if mode in ["word_random", "word_manual"]:
        assert word_table is not None
        assert len(words) > 0
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    video_length = get_video_length(in_file)
    video_height, video_width = get_video_size(in_file)
    config = {
        "mode": "exact",
        "v_mode": [],
        "v_start": [],
        "v_end": [],
        "v_option": [],
        "a_mode": [],
        "a_start": [],
        "a_end": [],
        "a_option": [],
    }

    def _generate_options(mode, intensity):
        if mode == "blank":
            v_option = None
            return v_option
        elif mode == "avgblur":
            sizex = sizey = int(1 + intensity * 8)
            v_option = (sizex, sizey)
            return v_option
        elif mode == "gblur":
            v_option = 20 * intensity
            return v_option
        elif mode == "impulse_value":
            v_option = 10 * intensity
            return v_option
        elif mode == "occlusion":
            w = video_width * intensity
            h = video_height * intensity
            x = random.random() * (video_width - w)
            y = random.random() * (video_height - h)
            v_option = (x, y, w, h)
            return v_option
        elif mode == "color":
            m = random.choices(["contrast", "brightness", "saturation"], k=random.choice([1, 2, 3]))
            contrast, brightness, saturation = 1.0, 0.0, 1.0
            if "contrast" in m:
                contrast = 1 + intensity * random.choice([-1, 1])
            if "brightness" in m:
                brightness = 0.5 * intensity * random.choice([-1, 1])
            if "saturation" in m:
                saturation = 1 + intensity * random.choice([-1, 1])
            v_option = (contrast, brightness, saturation, 1.0, 1.0, 1.0)
            return v_option
        elif mode == "rgb2bgr":
            v_option = ("b", "r")
            return v_option
        elif mode == "color_inversion":
            v_option = None
            return v_option
        elif mode == "volume":
            a_option = (1 - intensity)
            return a_option
        elif mode == "mute":
            return 0
        elif mode == "colorna":
            a_option = ("random", intensity)
            return a_option
        elif mode == "lowpass":
            a_option = 1000 * (1 - intensity)
            return a_option
        elif mode == "background":
            a_option = ("random", 0.5 + 2 * intensity)
            return a_option
        elif mode == "sudden":
            a_option = ("random", 0.5 + 2 * intensity)
            return a_option
        elif mode == "reverb":
            a_option = ("hall", intensity  ** 2)
            return a_option
    
    if mode in ["random_full", "random_time"]:
        if len(v_noise_list) > 0:
            # video noise
            if type(v_noise_num) == tuple:
                v_noise_num = randint_k_order(v_noise_num[0], v_noise_num[1])
            if mode == "random_time":
                v_noise_num = len(v_noise_list)
            if type(v_noise_ratio) == tuple:
                v_noise_ratio = random.uniform(v_noise_ratio[0], v_noise_ratio[1])
            v_noise_lengths = np.random.dirichlet(np.ones(v_noise_num)*10) * video_length * v_noise_ratio
            v_starts = np.random.random_sample(v_noise_num) * video_length * (1 - v_noise_ratio)
            v_starts = np.sort(v_starts) + np.insert(np.cumsum(v_noise_lengths), 0, 0)[:-1]
            v_ends = v_starts + v_noise_lengths
            for i in range(v_noise_num):
                if mode == "random_time":
                    v_mode = v_noise_list[i]
                else:
                    v_mode = random.choice(v_noise_list)
                if type(v_noise_intensity) == tuple:
                    v_intensity = random.uniform(v_noise_intensity[0], v_noise_intensity[1])
                elif type(v_noise_intensity) == list:
                    v_intensity = v_noise_intensity[i]
                else:
                    v_intensity = v_noise_intensity
                v_option = _generate_options(v_mode, v_intensity)
                v_start = v_starts[i]
                v_end = v_ends[i]
                config["v_mode"].append(v_mode if v_mode != "rgb2bgr" else "color_channel_swapping")
                config["v_start"].append(v_start)
                config["v_end"].append(v_end)
                config["v_option"].append(v_option)
        if len(a_noise_list) > 0:
            # audio noise
            if type(a_noise_num) == tuple:
                a_noise_num = randint_k_order(a_noise_num[0], a_noise_num[1])
            if mode == "random_time":
                a_noise_num = len(a_noise_list)
            if type(a_noise_ratio) == tuple:
                a_noise_ratio = random.uniform(a_noise_ratio[0], a_noise_ratio[1])
            a_noise_lengths = np.random.dirichlet(np.ones(a_noise_num)*10) * video_length * a_noise_ratio
            a_starts = np.random.random_sample(a_noise_num) * video_length * (1 - a_noise_ratio)
            a_starts = np.sort(a_starts) + np.insert(np.cumsum(a_noise_lengths), 0, 0)[:-1]
            a_ends = a_starts + a_noise_lengths
            for i in range(a_noise_num):
                if mode == "random_time":
                    a_mode = a_noise_list[i]
                else:
                    a_mode = random.choice(a_noise_list)
                    if "reverb" in config["a_mode"]: # reverb should only occur once
                        while a_mode == "reverb":
                            a_mode = random.choice(a_noise_list)
                if type(a_noise_intensity) == tuple:
                    a_intensity = random.uniform(a_noise_intensity[0], a_noise_intensity[1])
                elif type(a_noise_intensity) == list:
                    a_intensity = a_noise_intensity[i]
                else:
                    a_intensity = a_noise_intensity
                a_option = _generate_options(a_mode, a_intensity)
                a_start = a_starts[i]
                a_end = a_ends[i]
                config["a_mode"].append(a_mode if a_mode != "mute" else "volume")
                config["a_start"].append(a_start)
                config["a_end"].append(a_end)
                config["a_option"].append(a_option)

    if mode in ["word_random", "word_manual"]:
        pass

    logger.debug(config)
    return RealNoiseConfig._make(config.values()) # With python 3.7+, iteration order of dict is guaranteed to be insertion order.
