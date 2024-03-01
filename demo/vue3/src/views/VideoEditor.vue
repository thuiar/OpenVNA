<template>
  <div class="appContainer">
    <StepProcess :activeStep="activeStep" @trainsActivate="trainsActivate" />
    <div class="firstPageContainer" v-if="activeStep === 0">
      <UploadVideo @transmitMp4="currentFile = $event" />
      <el-button
        class="elButton homeback"
        @click="backhome"
        size="small"
        type="info"
        >Home</el-button
      >
      <el-button
        class="aligned elButton nextbutton"
        @click="nextButton(1)"
        size="small"
        type="primary"
        >Next</el-button
      >
    </div>
    <div class="firstPageContainer" v-if="activeStep === 1">
      <video id="my-player1" class="video-js1" controls data-setup="{}">
        <source :src="McurrentVideoUrl" type="video/mp4" />
      </video>
      <div class="transcriptCon">
        <div class="transcriptTip">Transcript:</div>
        <el-input
          type="textarea"
          :rows="2"
          placeholder="Please enter"
          v-model="textarea"
        ></el-input>
      </div>
      <div>
        <el-button
          class="aligned elButton nextbutton"
          @click="nextButton(2)"
          size="small"
          type="primary"
          >Next</el-button
        >
      </div>
    </div>
    <div class="pageContainer" v-else-if="activeStep === 2">
      <div class="modifiedCon">
        <div class="modifiedTopContainer">
          <div class="previewContainer">
            <div class="previewTop">
              <div class="topTip">Preview</div>
              <div class="topDescribe">
                Click "generate" button to preview changes.
              </div>
            </div>
            <div id="modifiedContainer">
              <video
                id="modified-my-player"
                class="video-js vjs-big-play-centered"
                controls
              >
                <source :src="McurrentVideoUrl" type="video/mp4" />
              </video>
            </div>
            <div class="modifiedWaveform" v-if="modifiedWaveformSwitch">
              <div id="modifiedWaveTimeline" ref="modifiedWaveTimeline"></div>
              <div
                id="modifiedWaveform"
                ref="modifiedWaveform"
                @click="seekToTime(null)"
                @contextmenu.prevent="rightClickItem($event, 'wave')"
                @mouseover="
                  OperationMethod.waveOver(
                    $event,
                    'rgba(255, 228, 196, 0.5)',
                    'rgba(255, 228, 196, 0.5)',
                    rightPopup
                  )
                "
                @mouseout="
                  OperationMethod.waveOver(
                    $event,
                    'rgba(181, 198, 241, 0.2)',
                    '#fff',
                    rightPopup
                  )
                "
              ></div>
            </div>
            <div class="displayTimeContainer">
              <div class="topDescribe">
                Drag & drop <span>Time</span> onto word to get word start and
                end times
              </div>
              <el-form-item style="margin-top: 10px">
                <template #label>
                  <span draggable="true" @dragend="dragGetTime($event)"
                    >Time</span
                  >
                </template>
                <el-input
                  v-model="currentNoiseItem.text"
                  placeholder="Text"
                  class="textTimeInput"
                  readonly
                />
                -
                <el-input
                  v-model="currentNoiseItem.start"
                  placeholder="Start time"
                  class="startTimeInput"
                  readonly
                />
                -
                <el-input
                  v-model="currentNoiseItem.end"
                  placeholder="End time"
                  class="endTimeInput"
                  readonly
                />
              </el-form-item>
            </div>

            <div class="textContent">
              <span
                @mouseenter="
                  textOver(
                    $event,
                    index,
                    'rgba(255, 228, 196, 0.5)',
                    'rgba(255, 228, 196, 0.5)'
                  )
                "
                @mouseleave="
                  textOver($event, index, 'rgba(181, 198, 241, 0.2)', '#fff')
                "
                @contextmenu.prevent="rightClickItem($event, 'text')"
                @dragover="
                  $event.preventDefault(),
                    ($event.dataTransfer.dropEffect = 'move')
                "
                @click="seekToTime(textitem.start)"
                v-for="(textitem, index) in modifiedTextList"
                :key="textitem"
                :id="index"
                >{{ textitem.text }}</span
              >
            </div>
          </div>
          <div class="methodsContainer">
            <div class="methodsTop">
              <div class="topTip">Modify Methods</div>
              <div class="topDescribe">
                <span>Drag & drop</span> methods onto word to take effect
              </div>
            </div>
            <el-card shadow="always" class="methodContent">
              <div class="methodContainer">
                <div class="modalityMethod">
                  <span>Text Modality</span>
                </div>
                <div class="methodState">
                  <span
                    draggable="true"
                    @dragend="dragEnd($event, 't', 'replace', 'off')"
                    >Replace</span
                  >
                </div>
                <div class="methodState">
                  <span
                    draggable="true"
                    @dragend="dragEnd($event, 't', 'remove', '')"
                    >Remove</span
                  >
                </div>
              </div>
            </el-card>
            <div class="topDescribe">
              Fill in the form and click the Add button
            </div>
            <AddNoise @addNoise="addNoise" />
          </div>
        </div>
        <el-checkbox
          v-model="asrNosie"
          label="If selected, text changes will only be affected by audio noise."
          class="asrCheckbox"
          size="large"
        />
        <el-button
          class="aligned elButton generate"
          @click="generate"
          type="danger"
          >Generate</el-button
        >
        <el-button
          class="aligned elButton nextbutton"
          :type="McurrentVideoUrl == currentVideoUrl ? 'info' : 'primary'"
          @click="nextButton(3)"
          >Next</el-button
        >
        <el-divider />
        <TableModified
          :editAligned="editAligned"
          :onlyDisplay="false"
          @deleteNoise="deleteNoise"
        />
      </div>
    </div>
    <div
      class="pageContainer"
      style="position: relative"
      v-else-if="activeStep === 3"
    >
      <select-method
        class="selectMethod"
        :disabled="false"
        @transmitMethods="methodList = $event"
      />
      <el-button
        class="aligned elButton nextbutton"
        style="margintop: 60px"
        @click="nextButton(4)"
        type="primary"
        >Next</el-button
      >
    </div>
    <div class="pageContainer" v-else-if="activeStep === 4">
      <view-result
        :textList="modifiedTextList"
        :videoUrl="McurrentVideoUrl"
        :defenceVideoUrl="defenceVideoUrl"
        :originalVideoUrl="currentVideoUrl"
        :duration="duration"
        :editAligned="editAligned"
        :viewResults="viewResults"
        :methodData="methodData"
        :asrNosie="asrNosie"
      />
    </div>
    <div
      id="popup"
      :style="{
        top: showPositionTop + 'px',
        left: showPositionLeft + 'px',
        position: 'absolute',
      }"
      @mouseleave="
        (modalityDisplay.audio = false),
          (modalityDisplay.gblur = false),
          (modalityDisplay.video = false),
          (modalityDisplay.text = false)
      "
    >
      <div
        v-show="rightPopup"
        :style="{ position: 'absolute' }"
        class="popupContainer"
      >
        <div
          class="modality"
          @mouseover="
            (modalityDisplay.audio = false),
              (modalityDisplay.gblur = false),
              (modalityDisplay.video = true),
              (modalityDisplay.text = false)
          "
        >
          <span>Video</span>
          <el-icon>
            <ArrowRight />
          </el-icon>
        </div>
        <div
          class="modality"
          @mouseover="
            (modalityDisplay.audio = true),
              (modalityDisplay.gblur = false),
              (modalityDisplay.video = false),
              (modalityDisplay.text = false)
          "
        >
          <span>Audio</span>
          <el-icon>
            <ArrowRight />
          </el-icon>
        </div>
        <div
          class="modality"
          @mouseover="
            (modalityDisplay.audio = false),
              (modalityDisplay.gblur = false),
              (modalityDisplay.video = false),
              (modalityDisplay.text = true)
          "
        >
          <span>Text</span>
          <el-icon>
            <ArrowRight />
          </el-icon>
        </div>
        <div
          class="modalityClose"
          @click="modalityModify('clear', 'all')"
          @mouseover="
            (modalityDisplay.audio = false),
              (modalityDisplay.gblur = false),
              (modalityDisplay.video = false),
              (modalityDisplay.text = false)
          "
        >
          Clear
        </div>
      </div>
      <div
        v-show="
          rightPopup &&
          (modalityDisplay.video ||
            modalityDisplay.audio ||
            modalityDisplay.text)
        "
        :style="{
          top:
            (modalityDisplay.audio ? 50 : 0) +
            (modalityDisplay.video ? 10 : 0) +
            (modalityDisplay.text ? 90 : 0) +
            'px',
          left: 140 + 'px',
          position: 'relative',
        }"
        class="popupContainer"
      >
        <div v-show="modalityDisplay.video">
          <div
            class="state"
            @mouseover="modalityDisplay.gblur = false"
            @click="modalityModify('blank', 'v')"
          >
            <span>Blank</span>
            <el-icon></el-icon>
          </div>
          <div class="state" @mouseover="modalityDisplay.gblur = true">
            <span>Gblur</span>
            <el-icon>
              <ArrowRight />
            </el-icon>
          </div>
        </div>
        <div v-show="modalityDisplay.audio">
          <div class="state" @click="modalityModify('mute', 'a')">
            <span>Mute</span>
            <el-icon></el-icon>
          </div>
        </div>
        <div v-show="modalityDisplay.text">
          <div class="state" @click="modalityModify('replace', 't')">
            <span>Replace</span>
            <el-icon></el-icon>
          </div>
          <div class="state" @click="modalityModify('remove', 't')">
            <span>Remove</span>
            <el-icon></el-icon>
          </div>
        </div>
      </div>
      <div
        v-show="rightPopup && modalityDisplay.gblur"
        :style="{ top: -30 + 'px', left: 280 + 'px', position: 'relative' }"
        class="popupContainer"
      >
        <div>
          <div class="state" @click="modalityModify('gblur', 'v', 5)">
            <span style="display: block; width: 70px">Low</span>
            <el-icon></el-icon>
          </div>
          <div class="state" @click="modalityModify('gblur', 'v', 10)">
            <span style="display: block; width: 70px">Medium</span>
            <el-icon></el-icon>
          </div>
          <div class="state" @click="modalityModify('gblur', 'v', 20)">
            <span style="display: block; width: 70px">High</span>
            <el-icon></el-icon>
          </div>
        </div>
      </div>
    </div>
    <el-dialog v-model="dialogReplaceText" title="Text replacement" width="30%">
      <div class="dialogReplaceTextContainer">
        <div class="dialogReplaceItem">
          <div class="dialogReplaceTitle">Word:</div>
          <el-input readonly v-model="dialogReplaceTitle" />
        </div>
        <div class="dialogReplaceItem">
          <div class="dialogReplaceTitle">Alternate:</div>
          <el-input v-model="alternateText" placeholder="Please input" />
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button type="primary" @click="dialogConfirm">Confirm</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>
  
  <script setup>
import { ref, watchEffect, computed, onMounted } from "vue";
import StepProcess from "@/components/StepProcess.vue";
import UploadVideo from "@/components/UploadVideo.vue";
import TableModified from "@/components/TableModified.vue";
import SelectMethod from "@/components/SelectMethod.vue";
import ViewResult from "@/components/ViewResult.vue";
import AddNoise from "@/components/AddNoise.vue";
import { uploadTranscript } from "@/api/upload";
import OperationMethod from "@/utils/operation.js";
import { callASR } from "@/api/detection";
import { videoEditAligned, getFileFromUrl } from "@/api/modify";
import { ElLoading, ElMessage } from "element-plus";
import { runMSAAligned } from "@/api/detection";
import { useRouter } from "vue-router";
const router = useRouter();
const currentFile = ref("");
const McurrentFile = ref("");
const currentVideoUrl = ref("");
const McurrentVideoUrl = ref("");
const activeStep = ref(0);
const modifiedVideoTime = ref(0);
const modifiedWavesurfer = ref(null);
const duration = ref();
const originalID = ref("");
const originalPath = ref("");
const textarea = ref("");
const showPositionTop = ref(0);
const showPositionLeft = ref(0);
const dialogReplaceTitle = ref("");
const alternateText = ref("");
const rightPopup = ref(false);
const editAlignedIndex = ref(-1);
const currentNoiseItem = ref({
  start: "",
  end: "",
  text: "",
});
const methodList = ref({
  defence: ["a_denoise", "v_reconstruct"],
  models: ["T2FN", "TPFN", "CTFN", "MMIN","TFRNet","GCNET", "NIAT", "EMT_DLFR"],
});
const modalityDisplay = ref({
  video: false,
  audio: false,
  text: false,
  gblur: false,
  clear: false,
});
const modifiedPlayer = ref(null);
const modifiedWaveform = ref(null);
const dialogReplaceText = ref(false);
const viewResults = ref(null);
const methodData = ref(null);
const modifiedTextList = ref([]);
const editAligned = ref([]);
const asrNosie = ref(false);
const defenceVideoUrl = ref("");
const regions = computed(() => {
  let dataList = [];
  modifiedTextList.value.forEach((item) => {
    let itemdata = {
      start: 0,
      end: 0,
      attributes: {
        label: "",
      },
      data: {
        note: "",
      },
      loop: false,
      drag: false,
      resize: false,
      color: "rgba(181, 198, 241, 0.2)",
      handleStyle: false,
    };
    itemdata.start = item.start;
    itemdata.end = item.end;
    dataList.push(itemdata);
  });
  return dataList;
});
const modifiedWaveformSwitch = ref(true);
const modifiedEvent = () => {
  modifiedWavesurfer.value = OperationMethod.waveformCreate(
    modifiedWaveform.value,
    regions.value
  );
  modifiedWavesurfer.value.load(McurrentVideoUrl.value);
};
const trainsActivate = (e) => {
  if(activeStep.value>e){
    activeStep.value = e;
  }
};

const nextButton = async (index) => {
  
  if (index == 1) {
    if (typeof currentFile.value == "undefined" || currentFile.value == "") {
      ElMessage({
        message: "Please select a file.",
        type: "warning",
      });
      return;
    }
    let result = await OperationMethod.uploadFile(currentFile.value);
    if (result.code === 200) {
      try {
        var response = await getFileFromUrl(
          `${result.id}/raw_video.mp4`,
          "raw_video.mp4"
        );
        McurrentVideoUrl.value = URL.createObjectURL(response);
      } catch (error) {
        return;
      }
      originalID.value = result.id;
      originalPath.value = result.path;
      let ASRResult = await ASRClick(result.id);
      if (ASRResult.code != 200) {
        return;
      }
      try {
        var response = await getFileFromUrl(
          `${result.id}/raw_video.mp4`,
          "raw_video.mp4"
        );
      } catch (error) {
        return;
      }
      duration.value =
        (await OperationMethod.getVideoDuration(response)) / 1000000;
      McurrentFile.value = [response];
    } else {
      return;
    }
  } else if (index == 2) {
    if (textarea.value) {
      if (/\d/.test(textarea.value)) {
        ElMessage({
          message: "Please substitute numbers with text",
          type: "warning",
        });
        return;
      }
      const loading = ElLoading.service({
        lock: true,
        text: "Aligning video with transcript...",
      });
      try {
        var result = await uploadTranscript({
          videoID: originalID.value,
          transcript: textarea.value,
        });
        loading.close();
      } catch (error) {
        loading.close();
        return;
      }
      modifiedTextList.value = result.data.align;
      currentVideoUrl.value = McurrentVideoUrl.value;
    }
    setTimeout(() => {
      modifiedEvent();
    }, 5);
  } else if (index == 3) {
    if (McurrentVideoUrl.value === currentVideoUrl.value) {
      ElMessage({
        message: "After modifying the video, please click generate.",
        type: "warning",
      });
      return;
    }
  } else if (index == 4) {
    if (methodList.value.models.length == 0) {
      ElMessage({
        message: "Please select at least one msa model.",
        type: "warning",
      });
      return;
    } else {
      methodData.value = methodList.value;
      methodData.value["videoID"] = originalID.value;
      const loading = ElLoading.service({
        lock: true,
        text: "Processing, please wait...",
      });
      try {
        let result = await runMSAAligned(methodData.value);
        defenceVideoUrl.value = await getFileFromUrl(
          `${originalID.value}/defended_video.mp4`,
          "defended_video.mp4"
        );
        defenceVideoUrl.value = URL.createObjectURL(defenceVideoUrl.value);
        viewResults.value = result.data;
        loading.close();
      } catch (error) {
        loading.close();
        return;
      }
    }
  }
  activeStep.value++;
};
const backhome = () => {
  router.push({ path: "/" });
};
const ASRClick = async (originalID) => {
  const loading = ElLoading.service({
    lock: true,
    text: "Speech Recognition in progress...",
  });
  try {
    const ASRResult = await callASR({ videoID: originalID });
    textarea.value = ASRResult.data.result;
  } catch (error) {
    return { code: 400 };
  } finally {
    loading.close();
  }
  return { code: 200 };
};
const watchVideo = (videoTime) => {
  if (
    videoTime > modifiedTextList.value[modifiedTextList.value.length - 1].end
  ) {
    OperationMethod.hightLightInit();
    return;
  }
  let watchSwitch = true;
  for (let index = 0; index < modifiedTextList.value.length; index++) {
    if (
      modifiedTextList.value[index].start < videoTime &&
      videoTime <= modifiedTextList.value[index].end
    ) {
      OperationMethod.highlightOver(
        index - 1,
        "rgba(181, 198, 241, 0.2)",
        "#fff"
      );
      OperationMethod.highlightOver(
        index,
        "rgba(255, 228, 196, 0.5)",
        "rgba(255, 228, 196, 0.5)"
      );
      watchSwitch = false;
      break;
    }
  }
  if (watchSwitch) {
    OperationMethod.hightLightInit();
  }
};
const modifiedVideoTimeEvent = (that) => {
  let count = 10;
  let timer = setInterval(() => {
    modifiedVideoTime.value = that.currentTime();
    modifiedWavesurfer.value.seekAndCenter(
      modifiedVideoTime.value / duration.value > 1
        ? 1
        : modifiedVideoTime.value / duration.value
    );
    watchVideo(modifiedVideoTime.value);
    count == 0 ? clearInterval(timer) : count--;
  }, 25);
};
const seekToTime = (textitem) => {
  if (textitem == null) {
    setTimeout(() => {
      modifiedPlayer.value.currentTime(
        modifiedWavesurfer.value.getCurrentTime() + 0.000001
      );
    }, 5);
  } else {
    modifiedPlayer.value.currentTime(textitem + 0.000001);
  }
};
const dragEnd = (e, type, method, lastkey) => {
  var element = document.elementFromPoint(e.clientX, e.clientY);
  if ($(element).parent().attr("class") == "textContent") {
    editAlignedIndex.value = Number($(element).attr("id"));
    let time = [
      modifiedTextList.value[editAlignedIndex.value].start,
      modifiedTextList.value[editAlignedIndex.value].end,
      editAlignedIndex.value,
    ];
    time[0] = time[0].toFixed(4);
    time[1] = time[1].toFixed(4);
    editAligned.value = editAligned.value.filter((item) => {
      return !(item[0][2] == time[2] && item[1] == type && item[2] == method);
    });
    let item = [time, type, method, lastkey];
    if (method == "replace") {
      dialogReplaceText.value = true;
      dialogReplaceTitle.value =
        modifiedTextList.value[editAlignedIndex.value].text;
      return;
    }
    editAligned.value.push(item);
  }
};
const dragGetTime = (e) => {
  var element = document.elementFromPoint(e.clientX, e.clientY);
  if ($(element).parent().attr("class") == "textContent") {
    let index = Number($(element).attr("id"));
    currentNoiseItem.value = modifiedTextList.value[index];
    currentNoiseItem.value.start = currentNoiseItem.value.start.toFixed(4);
    currentNoiseItem.value.end = currentNoiseItem.value.end.toFixed(4);
  }
};
const deleteNoise = (index) => {
  editAligned.value.splice(index, 1);
};
const modifiedVideoPlayer = () => {
  modifiedPlayer.value = videojs(
    "modified-my-player",
    {},
    function onPlayerReady() {
      this.on("play", function (e) {
        modifiedVideoTimeEvent(this);
      });
      this.on("timeupdate", function (e) {
        modifiedVideoTimeEvent(this);
      });
      this.on("seeked", function () {
        OperationMethod.hightLightInit();
      });
    }
  );
};
watchEffect(
  () => {
    if (activeStep.value == 2) {
      let player = videojs.getPlayer("modified-my-player");
      if (player) {
        player.dispose();
      }
      modifiedVideoPlayer();
    }
  },
  {
    flush: "post",
  }
);

const mouseDown = (e) => {
  var state = true;
  try {
    e.path.forEach((item) => {
      if ($(item).attr("id") == "popup") {
        state = false;
      }
    });
  } catch (error) {
    return;
  }
  if (state && modifiedTextList.value.length) {
    rightPopup.value = false;
    OperationMethod.hightLightInit();
  }
};
const dialogConfirm = () => {
  dialogReplaceText.value = false;
  rightPopup.value = false;
  let time = [
    modifiedTextList.value[editAlignedIndex.value].start,
    modifiedTextList.value[editAlignedIndex.value].end,
    editAlignedIndex.value,
  ];
  time[0] = time[0].toFixed(4);
  time[1] = time[1].toFixed(4);
  let item = [
    time,
    "t",
    "replace",
    [modifiedTextList.value[editAlignedIndex.value].text, alternateText.value],
  ];
  
  editAligned.value.push(item);
};
const addNoise = (e) => {
  editAligned.value.push(e);
};

const textOver = async (e, index, background, textBackground) => {
  if (!rightPopup.value) {
    await OperationMethod.highlightOver(index, background, textBackground);
  }
};

const rightClickItem = async (e, type) => {
  const element = document.getElementById(`${e.target.id}`);
  showPositionTop.value = element.offsetTop + 80 || 0;
  showPositionLeft.value = element.offsetLeft + element.offsetWidth + 25 || 0;
  rightPopup.value = true;
  await OperationMethod.hightLightInit();
  let index = $(
    type == "wave"
      ? "#modifiedWaveform > wave > region.wavesurfer-region"
      : ".pageContainer .modifiedCon .textContent span"
  ).index(e.target);
  editAlignedIndex.value = index;
  await OperationMethod.highlightOver(
    index,
    "rgba(255, 228, 196, 0.5)",
    "rgba(255, 228, 196, 0.5)"
  );
  modalityDisplay.value.clear = false;
  editAligned.value.forEach((item) => {
    if (item[0][2] == index) {
      modalityDisplay.value.clear = true;
    }
  });
  modalityDisplay.value.clear = modalityDisplay.value.clear ? true : false;
};
const generate = async () => {
  if (editAligned.value.length == 0) {
    ElMessage({
      message: "Please modify the video first.",
      type: "warning",
    });
    return;
  }
  let data = {
    videoID: originalID.value,
    words: editAligned.value,
    asrNosie: asrNosie.value,
  };
  const loading = ElLoading.service({
    lock: true,
    text: "Generating modified video...",
  });
  try {
    var result = await videoEditAligned(data);
  } catch (error) {
    loading.close();
    return;
  }
  if (result.data.code == 200) {
    modifiedTextList.value = result.data.align;
    McurrentFile.value = "";
    try {
      var response = await getFileFromUrl(
        `${originalID.value}/modified_video.mp4`,
        "modified_video.mp4"
      );
      McurrentVideoUrl.value = URL.createObjectURL(response);
    } catch (error) {
      loading.close();
      return;
    }
    McurrentFile.value = [response];
    modifiedPlayer.value.dispose();
    let html =
      '<video id="modified-my-player" class="video-js vjs-big-play-centered" controls="true"><source src="' +
      McurrentVideoUrl.value +
      '" type="video/mp4" /></video>';
    document.getElementById("modifiedContainer").innerHTML = html;
    modifiedVideoPlayer();
    modifiedWaveformSwitch.value = false;
    setTimeout(() => {
      modifiedWaveformSwitch.value = true;
      setTimeout(() => {
        modifiedEvent();
      }, 2);
    }, 2);
    loading.close();
  }
};
const modalityModify = (e, t, last = "") => {
  OperationMethod.hightLightInit();
  let edit_aligned = editAligned.value.filter((item) => {
    return item[0][2] == editAlignedIndex.value && (item[1] == t || "all" == t)
      ? false
      : true;
  });
  if (e != "clear") {
    let lastkey = "";
    if (t == "v") {
      lastkey = e == "gblur" ? last : "";
    } else if (e == "replace") {
      dialogReplaceText.value = true;
      dialogReplaceTitle.value =
        modifiedTextList.value[editAlignedIndex.value].text;
      return;
    }
    let time = [
      modifiedTextList.value[editAlignedIndex.value].start,
      modifiedTextList.value[editAlignedIndex.value].end,
      editAlignedIndex.value,
    ];
    time[0] = time[0].toFixed(4);
    time[1] = time[1].toFixed(4);
    let item = [time, t, e, lastkey];
    edit_aligned.push(item);
  } else {
    rightPopup.value = false;
    return;
  }
  editAligned.value = edit_aligned;
  rightPopup.value = false;
};
onMounted(async () => {
  window.addEventListener("mousedown", mouseDown);
  // let time = await OperationMethod.getVideoDuration('1.mp4')
  // console.log(time)
});
</script>
  
  <style scoped lang="scss">
.appContainer {
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  width: 850px;
  overflow: hidden;
}

.firstPageContainer {
  position: relative;
}
.firstPageContainer,
.pageContainer {
  padding-top: 80px;
  width: 800px;
  padding-bottom: 100px;
}
.modifiedCon {
  position: relative;
}

.firstPageContainer .elButton {
  width: 100px;
  height: 30px;
}
.firstPageContainer .aligned {
  margin-left: 150px;
}
.modifiedTopContainer {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
}
.topTip {
  font-size: 18px;
  color: #353535;
}
.topDescribe {
  font-size: 14px;
  color: #727272;
  margin-top: 5px;
}
.topDescribe span {
  font-weight: bold;
}
.video-js1 {
  margin-top: 15px;
  width: 800px;
  height: 500px;
}
.video-js {
  margin-top: 15px;
}
#app
  .firstPageContainer
  button.el-button.elButton.el-button--default.el-button--small {
  margin-left: 0;
}
.firstPageContainer .transcriptCon {
  width: 800px;
  display: flex;
  margin-top: 15px;
}
.firstPageContainer .transcriptTip {
  width: 100px;
}
.nextbutton {
  margin-top: 35px;
  position: absolute;
  width: 95px;
  right: 0;
}
.homeback {
  margin-top: 35px;
  position: absolute;
  width: 95px;
  left: 0;
}
.pageContainer .videoframe {
  overflow-x: scroll;
  margin-top: 20px;
  width: 800px;
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  overflow-y: hidden;
  box-shadow: 0 0 1px 1px rgb(211, 211, 211);
  border-radius: 5px;
  height: 120px;
}
.modifiedWaveform {
  width: 470px;
  position: relative;
  margin-top: 10px;
  box-shadow: 0 0 1px 1px rgb(211, 211, 211);
  border-radius: 5px;
  overflow-y: hidden;
}
#waveform > wave::-webkit-scrollbar {
  height: 10px;
}
#waveform > wave::-webkit-scrollbar-track {
  background-color: rgb(255, 255, 255);
  box-shadow: inset 0 0 2px rgba(0, 0, 0, 0.2);
}
#waveform > wave::-webkit-scrollbar-thumb {
  background-color: rgb(207, 207, 207);
  border-radius: 10px;
}
.displayTimeContainer {
  display: flex;
  flex-direction: column;
  margin-top: 10px;
  width: 470px;
}
.textContent {
  width: 470px;
  font-size: 18px;
  line-height: 25px;
  margin-top: -10px;
}
.textContent span {
  display: inline-block;
  margin-top: 5px;
  margin-right: 10px;
  user-select: none;
}
.textContent span:hover {
  background-color: rgba(255, 228, 196, 0.5);
  cursor: pointer;
}
.popupContainer {
  box-shadow: 1px 1px 1px 1px rgb(189, 189, 189);
  width: 130px;
  background: #fff;
  padding: 10px;
  line-height: 30px;
  color: rgb(77, 77, 77);
  font-size: 16px;
  z-index: 99;
}
.popupContainer .modality {
  width: 100%;
  display: flex;
  flex-direction: row;
  justify-content: space-around;
  line-height: 40px;
  align-items: center;
  font-size: 15px;
}
.popupContainer .modality span {
  display: inline-block;
  width: 70px;
  color: rgb(73, 73, 73);
}
.popupContainer .state {
  display: flex;
  flex-direction: row;
  justify-content: space-around;
  align-items: center;
  width: 100%;
  font-size: 14px;
  color: rgb(105, 105, 105);
}
.popupContainer .modalityClose {
  padding-left: 10px;
  font-size: 14px;
}
.popupContainer .modality:hover,
.popupContainer .modalityClose:hover,
.popupContainer .state:hover {
  cursor: pointer;
  background-color: rgb(236, 236, 236);
}
.asrCheckbox {
  position: absolute;
  margin-top: 30px;
}
.generate {
  position: absolute;
  margin-top: 35px;
  right: 150px;
  width: 95px;
}
.methodContent {
  width: 320px;
  overflow: hidden;
  margin-top: 10px;
}
.dialogReplaceTextContainer {
  display: flex;
  flex-direction: column;
  margin-top: -30px;
}
.dialogReplaceTitle {
  width: 100px;
}
.dialogReplaceItem {
  display: flex;
  flex-direction: row;
  align-items: center;
  margin-top: 10px;
}
.modalityMethod {
  display: flex;
  flex-direction: row;
  align-items: center;
  line-height: 25px;
  user-select: none;
}
.modalityMethod span {
  margin-top: -1.5px;
  margin-left: 10px;
  font-size: 15px;
}
.methodState {
  margin-left: 26px;
  font-size: 14px;
  position: relative;
  color: #616161;
  line-height: 28px;
  display: flex;
  flex-direction: row;
  align-items: center;
  user-select: none;
}
.modalityMethod span:hover,
.methodState:hover {
  cursor: pointer;
}
.methodState span:hover {
  background-color: rgb(230, 230, 230);
}
.selectMethod {
  position: relative;
  top: 10px;
}
.textTimeInput,
.startTimeInput,
.endTimeInput {
  width: 90px;
}
.textTimeInput,
.startTimeInput {
  margin-right: 10px;
}
.startTimeInput,
.endTimeInput {
  margin-left: 10px;
}
</style>
  
  <style lang="scss">
.el-step.is-simple .el-step__arrow {
  flex-grow: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 50px !important;
  transform: scale(0.5);
}
.el-step.is-simple {
  display: flex;
  flex-direction: row;
  align-items: center;
  flex-basis: 0 !important;
}
.el-step.is-simple .el-step__icon {
  width: 16px;
  height: 20px;
  line-height: 20px;
}
.el-step.is-simple .el-step__main {
  position: relative;
  display: flex;
  align-items: stretch;
  flex-grow: 0;
  width: max-content;
}
.el-step.is-simple .el-step__title {
  font-size: 14px;
}
#app .stepTop .el-step__main .el-step__title {
  max-width: 100%;
}
.el-step:last-of-type.is-flex {
  width: 121px;
}
#modified-my-player {
  margin-top: 10px;
  width: 470px;
  height: 320px;
}
.stepTop .el-step__main .is-finish,
.stepTop .is-finish {
  color: #67c23a;
}
div.stepTop .is-process > div,
#app .stepTop .el-step__main .is-process {
  color: #409eff;
}
.el-upload-dragger .successUpload .el-icon--upload {
  color: #409eff;
}
.el-upload__tip {
  font-size: 13px;
  display: flex;
  flex-direction: row;
  align-items: center;
}
.el-upload__tip span {
  margin-left: 5px;
}
.el-card__body {
  padding: 10px;
}
.tooltip-base-box .box-item {
  width: 110px;
}
.el-dialog__body {
  height: 40px;
}
.el-dialog__footer {
  padding-top: 0;
}
#modifiedWaveform {
  height: 75px !important;
}
#modifiedWaveform > wave {
  height: 75px !important;
  &::-webkit-scrollbar {
    height: 10px;
  }
  &::-webkit-scrollbar-track {
    background-color: rgb(255, 255, 255);
    box-shadow: inset 0 0 2px rgba(0, 0, 0, 0.2);
  }
  &::-webkit-scrollbar-thumb {
    background-color: rgb(207, 207, 207);
    border-radius: 10px;
  }
}
.methodContent .el-slider__runway.show-input {
  margin-right: 10px;
}
.methodContent .el-slider__button {
  height: 14px;
  width: 14px;
}
.methodContent .el-slider__input {
  width: 60px;
}
.methodContent .el-input-number__decrease,
.methodContent .el-input-number__increase {
  width: 14px;
}
.modifiedCon .el-divider--horizontal {
  margin: 15px 0;
  margin-top: 80px;
}
</style>