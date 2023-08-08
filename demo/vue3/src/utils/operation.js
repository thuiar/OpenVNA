import config from "@/config";
import WaveSurfer from "wavesurfer.js";
import Timeline from "wavesurfer.js/dist/plugin/wavesurfer.timeline.js";
import Regions from "wavesurfer.js/dist/plugin/wavesurfer.regions.js";
import CursorPlugin from "wavesurfer.js/dist/plugin/wavesurfer.cursor.js";
import { ElLoading, ElMessage } from "element-plus";
import { uploadVideo } from "@/api/upload";
const hightLightInit = () => {
  $(`#modifiedWaveform > wave > region.wavesurfer-region`).css({
    background: "rgba(181, 198, 241, 0.2)",
  });
  $(`.modifiedCon .textContent span`).css({
    background: "#fff",
  });
};

const highlightOver = (index, background, textBackground) => {
  $(
    `#modifiedWaveform > wave > region.wavesurfer-region:nth-child(${index + 3
    })`
  ).css({
    background,
  });
  $(
    `.pageContainer .modifiedCon .textContent span:nth-child(${index + 1})`
  ).css({
    background: textBackground,
  });
};



const _timeInterval = (pxPerSec) => {
  var retval = 1;
  if (pxPerSec >= 3000) {
    retval = 0.005;
  } else if (pxPerSec >= 1000) {
    retval = 0.01;
  } else if (pxPerSec >= 500) {
    retval = 0.05;
  } else if (pxPerSec >= 200) {
    retval = 0.1;
  } else if (pxPerSec >= 100) {
    retval = 0.4;
  } else if (pxPerSec >= 80) {
    retval = 1;
  } else if (pxPerSec >= 60) {
    retval = 2;
  } else if (pxPerSec >= 40) {
    retval = 1;
  } else if (pxPerSec >= 20) {
    retval = 5;
  } else {
    retval = Math.ceil(0.5 / pxPerSec) * 60;
  }
  return retval;
};
const noiseTip = (from_item_type) => {
  let result
  if (from_item_type == "lowpass") {
    result = "Frequency of low pass filter, given in Hz.";
  } else if (from_item_type == "volume") {
    result = "Volume range from 0 to 1. Values greater than 1 will amplify the audio. ";
  } else if (from_item_type == "coloran") {
    result = '(color, amplitude) of the constructed noise. \
            color: Which color noise is applied to the audio. Supported Values are ["white", "pink", "brown", "blue", "violet", "grey", "brown", "violet"]. \
            amplitude: Specify the amplitude (0.0 - 1.0) of the generated audio stream. Default value is 1.0.';
  } else if (from_item_type == "background") {
    result = '(filename, volume) of background noise. Filename is the name of background noise file. Supported Values are ["random", "metro", "office", "park", "restaurant", "traffic", "white", "music_soothing", "music_tense", "song_gentle", "song_rock"]. \
            If filename is "random", a random background noise will be selected. Volume is a float number greater than 0. \
            Background noise is applied to the specified time range. Currently, the background noise is not looped. Thus if the noise is shorter, it will not cover the whole time range. A random part of the noise will be selected if the noise file is longer.\
            Most noise files are about 5 minutes long, except for the music and songs which are about 3 minutes long.';
  } else if (from_item_type == "sudden") {
    result = '(filename, volume) of sudden noise. Filename is the name of sudden noise file. Supported Values are ["random", "beep", "glass", "thunder", "dog"]. \
            If filename is "random", a random background noise will be selected. Volume is a float number greater than 0. \
            Sudden noise is applied at the specified timestamp. It will not be looped and the end time is ignored. ';
  } else if (from_item_type == "reverb") {
    result = '(filename, length) of IR files. Reverb is done by applying Frequency Impulse Response filter. Filename is the name of IR file. Supported Values are ["hall", "room"]. \
            Length is the Impulse Response filter length range from 0 to 1. A value of 1 means whole IR is processed. Note that the length value does not equal to the strength of reverb. It depends on the waveform of the IR file, which is usually non-linear.\
            The Reverb effect is applied to the whole audio, thus the start and end time is ignored. ';
  } else if (from_item_type == "avgblur") {
    result = '(SizeX, SizeY) of the avgBlur filter. \
        SizeX, SizeY: Set horizontal and vertical radius size.';
  } else if (from_item_type == "gblur") {
    result = 'Sigma of Gaussian blur. ';
  } else if (from_item_type == "impulse_value") {
    result = 'Impulse valued noise (salt and pepper noise): \
        Noise strength for specific pixel component。 ';
  } else if (from_item_type == "occlusion") {
    result = '(x, y, w, h) of occlusion box. If mode is "percent", values are given as percentage of video size. Otherwise, they are given as pixels.';
  } else if (from_item_type == "color") {
    result = '(contrast, brightness, saturation, gamma_r, gamma_g, gamma_b) of color filter. \
            contrast: [-2.0, 2.0]. FFmpeg default: 1.0\
            brightness: [-1.0, 1.0]. FFmpeg default: 0.0\
            saturation: [0.0, 3.0]. FFmpeg default: 1.0\
            gamma_r: [0.1, 10.0]. FFmpeg default: 1.0\
            gamma_g: [0.1, 10.0]. FFmpeg default: 1.0\
            gamma_b: [0.1, 10.0]. FFmpeg default: 1.0';
  } else if (from_item_type == "color_channel_swapping") {
    result = '(channel_1, channel_2) of two channels to be swapped. Supported values are ["r", "g", "b"].\
            Currently only support swapping two channels.';
  } else if (from_item_type == "color_channel_swapping") {
    result = '(channel_1, channel_2) of two channels to be swapped. Supported values are ["r", "g", "b"].\
            Currently only support swapping two channels.';
  }
  return result
}
const echartsDataChange = (methodDetailValue) => {
  if (methodDetailValue == "F1amplitude") {
    return "F1amplitude: Ratio of the energy of the spectral harmonic peak at the first formant's centre frequency to the energy of the spectral peak at F0.";
  } else if (methodDetailValue == "F1bandwidth") {
    return "F1bandwidth: Bandwidth of first formant.";
  } else if (methodDetailValue == "F1frequency") {
    return "F1frequency: Centre frequency of first formant.";
  } else if (methodDetailValue == "HNR") {
    return "HNRdBACF: Harmonics-to-Noise Ratio. Relation of energy in harmonic components to energy in noise-like components.";
  } else if (methodDetailValue == "alphaRatio") {
    return "alphaRatio: Ratio of the summed energy from 50-1000 Hz and 1-5 kHz.";
  } else if (methodDetailValue == "loudness") {
    return "Loudness: Estimate of perceived signal intensity from an auditory spectrum.";
  } else if (methodDetailValue == "mfcc1") {
    return "MFCC1: Mel frequency cepstral coefficients - 1 ";
  } else if (methodDetailValue == "mfcc2") {
    return "MFCC2: Mel frequency cepstral coefficients - 2.";
  } else if (methodDetailValue == "mfcc3") {
    return "MFCC3: Mel frequency cepstral coefficients - 3.";
  } else if (methodDetailValue == "mfcc4") {
    return "MFCC4: Mel frequency cepstral coefficients - 4 ";
  } else if (methodDetailValue == "pitch") {
    return "Pitch: logarithmic F0 on a semitone frequency scale, starting at 27.5 Hz (semitone 0).";
  } else if (methodDetailValue == "AU01_r") {
    return "Action Unit 01: Inner brow raiser.";
  } else if (methodDetailValue == "AU02_r") {
    return "Action Unit 02: Outer brow raiser.";
  } else if (methodDetailValue == "AU04_r") {
    return "Action Unit 04: Brow lowerer.";
  } else if (methodDetailValue == "AU05_r") {
    return "Action Unit 05: Upper lid raiser.";
  } else if (methodDetailValue == "AU06_r") {
    return "Action Unit 06: Cheek raiser.";
  } else if (methodDetailValue == "AU07_r") {
    return "Action Unit 07: Lid rightener.";
  } else if (methodDetailValue == "AU09_r") {
    return "Action Unit 09: Nose wrinkler.";
  } else if (methodDetailValue == "AU10_r") {
    return "Action Unit 10: Upper lip raiser.";
  } else if (methodDetailValue == "AU12_r") {
    return "Action Unit 12: Lip corner puller.";
  } else if (methodDetailValue == "AU14_r") {
    return "Action Unit 14: Dimpler.";
  } else if (methodDetailValue == "AU15_r") {
    return "Action Unit 15: Lip corner depressor.";
  } else if (methodDetailValue == "AU17_r") {
    return "Action Unit 17: Chin raiser";
  } else if (methodDetailValue == "AU20_r") {
    return "Action Unit 20: Lip stretcher.";
  } else if (methodDetailValue == "AU23_r") {
    return "Action Unit 23: Lip tightener.";
  } else if (methodDetailValue == "AU25_r") {
    return "Action Unit 25: Lips part.";
  } else if (methodDetailValue == "AU26_r") {
    return "Action Unit 26: Jaw drop.";
  } else if (methodDetailValue == "AU45_r") {
    return "Action Unit 46: Wink.";
  } else {
    return "";
  }
};
const modifyCard = (editAligned, noiseObj, length) => {
  if (noiseObj.white) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.white &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_white"
      );
    });
    editAligned.push([-1, "a", "noise_white", noiseObj.white]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.white &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_white"
      );
    });
  }
  if (noiseObj.metro) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.metro &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_metro"
      );
    });
    editAligned.push([-1, "a", "noise_metro", noiseObj.metro]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.metro &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_metro"
      );
    });
  }
  if (noiseObj.office) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.office &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_office"
      );
    });
    editAligned.push([-1, "a", "noise_office", noiseObj.office]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.office &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_office"
      );
    });
  }
  if (noiseObj.park) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.park &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_park"
      );
    });
    editAligned.push([-1, "a", "noise_park", noiseObj.park]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.park &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_park"
      );
    });
  }
  if (noiseObj.diner) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.park &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_diner"
      );
    });
    editAligned.push([-1, "a", "noise_diner", noiseObj.diner]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.park &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_diner"
      );
    });
  }
  if (noiseObj.traffic) {
    editAligned = editAligned.filter((item) => {
      return !(
        noiseObj.traffic &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_traffic"
      );
    });
    editAligned.push([-1, "a", "noise_traffic", noiseObj.traffic]);
  } else {
    editAligned = editAligned.filter((item) => {
      return !(
        !noiseObj.traffic &&
        item[0] == -1 &&
        item[1] == "a" &&
        item[2] == "noise_traffic"
      );
    });
  }
  hightLightInit();
  dragBackgroundColor(editAligned, length);
  return editAligned;
};

const waveformCreate = (
  modifiedWaveform,
  regions,
  container = "#modifiedWaveTimeline"
) => {
  return WaveSurfer.create({
    container: modifiedWaveform,
    scrollParent: true,
    hideScrollbar: false,
    waveColor: "#409EFF",
    progressColor: "blue",
    backend: "MediaElement",
    mediaControls: false,
    audioRate: "1",
    plugins: [
      Timeline.create({
        container: container,
        timeInterval: _timeInterval,
      }),
      Regions.create({
        showTime: true,
        regions: regions,
      }),
      CursorPlugin.create({
        showTime: true,
        opacity: 1,
        customShowTimeStyle: {
          "background-color": "#000",
          color: "#fff",
          padding: "5px",
          "font-size": "10px",
        },
      }),
    ],
  });
};
const getVideoDuration = videoFile =>
  new Promise((resolve, reject) => {
    try {
      const url = URL.createObjectURL(videoFile);
      // const url = videoFile;
      const tempAudio = new Audio(url);
      tempAudio.addEventListener("loadedmetadata", () => {
        resolve(tempAudio.duration * 1000000);
      });
    } catch (error) {
      console.log("getVideoDuration error", error);
      throw error;
    }
  });

const waveOver = async (e, waveBackground, textBackground, rightPopup) => {
  if (
    e.target.getAttributeNode("class") &&
    e.target.getAttributeNode("class").value === "wavesurfer-region" &&
    !rightPopup
  ) {
    let index = $(`#modifiedWaveform > wave > region.wavesurfer-region`).index(
      e.target
    );
    await highlightOver(index, waveBackground, textBackground);
  }
};

const uploadFile = async (file) => {
  const loading = ElLoading.service({
    lock: true,
    text: "Video uploading, please wait...",
  });
  let data = new FormData();
  data.append("video", file);
  try {
    var result = await uploadVideo(data);
  } catch (error) {
    return { code: 400 };
  } finally {
    loading.close();
  }
  return result.data;
};

export default {
  hightLightInit,
  highlightOver,
  modifyCard,
  waveformCreate,
  echartsDataChange,
  getVideoDuration,
  waveOver,
  uploadFile,
  noiseTip
};
