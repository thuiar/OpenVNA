<template>
  <div class="appContainer">
    <StepProcess :activeStep="activeStep" />
    <div class="firstPageContainer" v-if="activeStep === 0">
      <UploadVideo @transmitMp4="currentFile = $event" />
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
          label="If selected, the constructed audio noise will interfere with the text."
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
const currentFile = ref("");
const McurrentFile = ref("");
const currentVideoUrl = ref("1.mp4");
const McurrentVideoUrl = ref("1.mp4");
const activeStep = ref(4);
const modifiedVideoTime = ref(0);
const modifiedWavesurfer = ref(null);
const duration = ref(2.1);
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
  defence: [],
  models: ["MMIN", "TFR_Net", "T2FN", "TPFN"],
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
// const viewResults = ref(null);
const viewResults = ref({
  code: 200,
  feature: {
    defended: {
      AU01_r: [
        0.40428571428571425, 0.09375, 0.47833333333333333, 0.1175, 0.02,
        0.051111111111111114, 0.0, 0.0, 0.0, 0.0,
      ],
      AU02_r: [
        0.26999999999999996, 0.20375, 0.5258333333333333, 0.0, 0.0, 0.0,
        0.037500000000000006, 0.0, 0.0, 0.0,
      ],
      AU04_r: [
        0.0071428571428571435, 0.01875, 0.3516666666666666, 0.5875, 0.505,
        0.8644444444444445, 1.135, 1.1375, 0.925, 0.6122222222222222,
      ],
      AU05_r: [0.19, 0.0025, 0.0025, 0.02375, 0.0075, 0.0, 0.0, 0.0, 0.0, 0.0],
      AU06_r: [
        0.1885714285714286, 0.53125, 0.9108333333333332, 0.82125, 0.81,
        0.8755555555555555, 1.40625, 1.4874999999999998, 1.4525000000000001,
        1.221111111111111,
      ],
      AU07_r: [
        0.11, 0.37875, 0.7433333333333333, 1.09125, 0.82, 0.9611111111111111,
        1.5025, 1.445, 1.2674999999999998, 0.94,
      ],
      AU09_r: [
        0.0, 0.0, 0.04666666666666667, 0.007500000000000001, 0.0, 0.0,
        0.026250000000000002, 0.065, 0.1275, 0.08222222222222224,
      ],
      AU10_r: [
        0.2914285714285714, 0.17124999999999999, 0.45083333333333336, 0.32875,
        0.47, 0.6944444444444444, 1.1600000000000001, 0.7625,
        0.7875000000000001, 0.6733333333333333,
      ],
      AU12_r: [
        0.43142857142857144, 0.6125, 0.5391666666666667, 0.60625, 0.8175,
        0.6566666666666667, 0.8075, 0.8925000000000001, 0.9625,
        0.8877777777777779,
      ],
      AU14_r: [
        0.23142857142857146, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      AU15_r: [
        0.63, 1.9175, 0.19166666666666665, 0.05875, 0.2475, 0.5444444444444445,
        0.17625000000000002, 0.0, 0.0, 0.0,
      ],
      AU17_r: [
        1.3228571428571427, 1.4975000000000003, 0.28833333333333333,
        0.8674999999999999, 0.765, 0.8177777777777777, 0.3675, 0.28, 0.2375,
        0.3088888888888889,
      ],
      AU20_r: [
        0.46714285714285714, 1.0174999999999998, 0.32916666666666666, 0.08, 0.0,
        0.1677777777777778, 0.3925, 0.0, 0.0, 0.0,
      ],
      AU23_r: [
        0.722857142857143, 0.24000000000000002, 0.0016666666666666668, 0.0025,
        0.14750000000000002, 0.5066666666666666, 0.46749999999999997, 0.01, 0.0,
        0.01,
      ],
      AU25_r: [
        0.41285714285714287, 0.14500000000000002, 0.7733333333333334, 0.29625,
        0.40750000000000003, 0.03888888888888889, 0.6375, 1.0025, 1.085,
        1.0722222222222222,
      ],
      AU26_r: [
        0.38, 0.255, 0.20916666666666664, 0.485, 0.0475, 0.30888888888888894,
        0.9337500000000001, 0.5125, 0.295, 0.1811111111111111,
      ],
      AU45_r: [
        0.23714285714285713, 0.0, 0.015, 0.06875, 0.0025, 0.044444444444444446,
        0.0925, 0.065, 0.0925, 0.1877777777777778,
      ],
      F1amplitude: [
        0.0, -27.1655330657959, -61.5722541809082, 0.8455129861831665,
        8.922711372375488, -4.426234245300293, -5.364791393280029,
        -17.095701217651367, -17.095701217651367, -128.54888916015625,
        -98.55758666992188, 3.887383460998535, 0.0,
      ],
      F1bandwidth: [
        0.0, 1513.2564697265625, 1449.9925537109375, 1231.851806640625,
        1394.2886962890625, 1495.9149169921875, 1461.988037109375,
        1401.8651123046875, 1401.8651123046875, 1199.6478271484375,
        1337.5255126953125, 1202.6016845703125, 0.0,
      ],
      F1frequency: [
        0.0, 715.3416748046875, 671.6199340820312, 610.2813720703125,
        868.8509521484375, 554.9208374023438, 459.06640625, 515.6585693359375,
        515.6585693359375, 594.905029296875, 507.5403747558594,
        415.2783508300781, 0.0,
      ],
      HNR: [
        0.0, 9.019758224487305, 5.337822914123535, 11.15988540649414,
        6.8120927810668945, 7.5113372802734375, 10.425657272338867,
        6.323482990264893, 6.323482990264893, 3.4131112098693848,
        4.620552062988281, 5.560962200164795, 0.0,
      ],
      alphaRatio: [
        0.0, -7.102252006530762, -12.196745872497559, -12.503073692321777,
        -25.205198287963867, -14.299078941345215, -22.683610916137695,
        -16.56375503540039, -16.56375503540039, -0.8143048882484436,
        -20.064998626708984, -15.807799339294434, 0.0,
      ],
      loudness: [
        0.0, 0.8367114067077637, 0.729333221912384, 1.0656462907791138,
        0.594571053981781, 0.2987436056137085, 0.23693883419036865,
        0.2944853603839874, 0.2944853603839874, 0.1286957859992981,
        0.11854644119739532, 0.31139951944351196, 0.0,
      ],
      mfcc1: [
        0.0, 39.01980972290039, 43.4235954284668, 38.17974853515625,
        45.73247528076172, 50.30038070678711, 50.51851272583008,
        33.13459396362305, 33.13459396362305, 2.1913557052612305,
        20.27118682861328, 36.053836822509766, 0.0,
      ],
      mfcc2: [
        0.0, -31.514854431152344, -23.256990432739258, -23.791393280029297,
        -0.36708977818489075, 0.4896640181541443, 12.084959983825684,
        6.9147725105285645, 6.9147725105285645, 19.535587310791016,
        32.914554595947266, 4.534545421600342, 0.0,
      ],
      mfcc3: [
        0.0, 0.13886915147304535, -29.468856811523438, -6.6251959800720215,
        -14.576168060302734, -10.422444343566895, 5.368560314178467,
        22.51289176940918, 22.51289176940918, 15.74084758758545,
        31.0614013671875, 4.247032165527344, 0.0,
      ],
      mfcc4: [
        0.0, -13.61067008972168, -25.679676055908203, -24.97237205505371,
        -50.578468322753906, -19.375743865966797, 8.13053035736084,
        -0.9415223598480225, -0.9415223598480225, -11.971861839294434,
        -1.4402899742126465, -8.094376564025879, 0.0,
      ],
      pitch: [
        0.0, 32.0971794128418, 28.999208450317383, 38.119049072265625,
        33.952083587646484, 33.557682037353516, 32.799530029296875,
        29.439943313598633, 29.439943313598633, 12.09283447265625,
        17.941875457763672, 30.323244094848633, 0.0,
      ],
    },
    modified: {
      AU01_r: [
        0.43714285714285717, 0.07, 0.40583333333333327, 0.12500000000000003,
        0.02, 0.07333333333333333, 0.0, 0.0, 0.0, 0.04428571428571428,
      ],
      AU02_r: [
        0.26, 0.125, 0.4116666666666667, 0.0, 0.0, 0.0, 0.005, 0.0, 0.02,
        0.037142857142857144,
      ],
      AU04_r: [
        0.02, 0.02625, 0.31750000000000006, 0.5587500000000001, 0.495,
        0.9222222222222223, 1.0216666666666665, 0.68625, 0.54,
        0.6528571428571429,
      ],
      AU05_r: [
        0.17714285714285713, 0.030000000000000002, 0.0, 0.04125000000000001,
        0.005, 0.0, 0.0, 0.0, 0.08249999999999999, 0.11571428571428573,
      ],
      AU06_r: [
        0.20000000000000004, 0.54875, 0.8875000000000001, 0.8912499999999999,
        1.0150000000000001, 0.9755555555555557, 1.39, 1.25625,
        0.9249999999999999, 0.5457142857142857,
      ],
      AU07_r: [
        0.19, 0.5375000000000001, 0.7908333333333332, 1.1675, 1.1425,
        1.1577777777777778, 1.3608333333333336, 1.2575, 0.9225000000000001,
        0.9942857142857143,
      ],
      AU09_r: [
        0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 0.10083333333333333, 0.2775, 0.09,
        0.012857142857142857,
      ],
      AU10_r: [
        0.27714285714285714, 0.14500000000000002, 0.39333333333333337, 0.405,
        0.6375, 0.897777777777778, 1.2116666666666667, 0.89375, 0.53,
        0.31285714285714283,
      ],
      AU12_r: [
        0.4285714285714285, 0.61625, 0.5258333333333333, 0.5912499999999999,
        0.9199999999999999, 0.7677777777777778, 0.9841666666666667, 0.955,
        0.7175, 0.5142857142857143,
      ],
      AU14_r: [
        0.08142857142857143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      AU15_r: [
        0.72, 1.8324999999999998, 0.15833333333333333, 0.00875, 0.095,
        0.4111111111111111, 0.09500000000000001, 0.02375, 0.025, 0.0,
      ],
      AU17_r: [
        1.327142857142857, 1.51875, 0.31666666666666665, 0.8787499999999999,
        1.15, 0.8966666666666667, 0.31249999999999994, 0.12875, 0.1175,
        0.18142857142857144,
      ],
      AU20_r: [
        0.45999999999999996, 1.0924999999999998, 0.3658333333333333, 0.04875,
        0.0, 0.026666666666666665, 0.20166666666666666, 0.0075, 0.0, 0.0,
      ],
      AU23_r: [
        0.7942857142857144, 0.28125, 0.0, 0.0, 0.1875, 0.5555555555555556,
        0.31083333333333335, 0.0, 0.0, 0.0,
      ],
      AU25_r: [
        0.3514285714285714, 0.14, 0.5025000000000001, 0.1625,
        0.052500000000000005, 0.09888888888888889, 0.5175000000000001,
        0.8412499999999998, 0.78, 0.33285714285714285,
      ],
      AU26_r: [
        0.37714285714285717, 0.22749999999999998, 0.2541666666666667, 0.53125,
        0.25, 0.4044444444444445, 0.7566666666666665, 0.03875000000000001, 0.23,
        0.18285714285714286,
      ],
      AU45_r: [
        0.22285714285714286, 0.0, 0.0025, 0.015, 0.0, 0.0, 0.06166666666666667,
        0.11625, 0.185, 0.09999999999999999,
      ],
      F1amplitude: [
        -25.9796142578125, -77.21607208251953, -28.13727378845215,
        5.029468059539795, -6.669066905975342, -4.352915287017822,
        -86.65570068359375, -62.79671096801758, -5.797416687011719,
        2.3495986461639404,
      ],
      F1bandwidth: [
        1493.850341796875, 1662.228271484375, 1253.1856689453125,
        1302.529541015625, 1533.3612060546875, 1323.8211669921875,
        1184.103271484375, 1269.5845947265625, 1126.3819580078125,
        1399.779541015625,
      ],
      F1frequency: [
        590.2025146484375, 732.8140258789062, 605.6240844726562,
        866.2485961914062, 600.577880859375, 529.20654296875, 617.7623291015625,
        477.667724609375, 406.58282470703125, 656.6705322265625,
      ],
      HNR: [
        8.421034812927246, 4.518342018127441, 10.030532836914062,
        6.373897552490234, 6.380834579467773, 9.05953311920166,
        3.915076732635498, 3.3538572788238525, 5.082917213439941,
        5.984951019287109,
      ],
      alphaRatio: [
        -7.124314308166504, -10.509737014770508, -9.835587501525879,
        -20.348922729492188, -9.10754108428955, -12.363197326660156,
        -3.557070255279541, -5.089970588684082, -8.761316299438477,
        -14.597612380981445,
      ],
      loudness: [
        0.9687491655349731, 0.8250975608825684, 1.2241657972335815,
        0.6880991458892822, 0.47458913922309875, 0.421176940202713,
        0.46832722425460815, 0.5411249995231628, 0.4605310261249542,
        0.685849666595459,
      ],
      mfcc1: [
        37.5585823059082, 38.703372955322266, 28.842103958129883,
        42.299171447753906, 35.571128845214844, 40.50459671020508,
        12.38611125946045, 8.281085014343262, 19.307554244995117,
        42.09113693237305,
      ],
      mfcc2: [
        -30.33211898803711, -22.270971298217773, -20.78191375732422,
        -4.938215255737305, -6.814371109008789, -7.941344261169434,
        0.9046648740768433, 11.190454483032227, 6.442173480987549,
        -10.511086463928223,
      ],
      mfcc3: [
        4.204305171966553, -19.085464477539062, -2.477163553237915,
        -5.887889385223389, 3.416363000869751, 16.615476608276367,
        18.084882736206055, 10.782327651977539, 11.763980865478516,
        1.0811346769332886,
      ],
      mfcc4: [
        -16.529600143432617, -32.14363098144531, -26.883827209472656,
        -56.02228546142578, -23.051963806152344, -1.0602284669876099,
        -2.275239944458008, -8.505615234375, -3.7007880210876465,
        -54.559452056884766,
      ],
      pitch: [
        31.991926193237305, 22.681396484375, 33.04902267456055,
        33.57320785522461, 33.50397491455078, 32.747581481933594,
        18.861509323120117, 23.020832061767578, 31.651390075683594,
        29.260404586791992,
      ],
    },
    original: {
      AU01_r: [
        0.43714285714285717, 0.07, 0.40583333333333327, 0.12500000000000003,
        0.02, 0.07333333333333333, 0.0, 0.0, 0.0, 0.04428571428571428,
      ],
      AU02_r: [
        0.26, 0.125, 0.4116666666666667, 0.0, 0.0, 0.0, 0.005, 0.0, 0.02,
        0.037142857142857144,
      ],
      AU04_r: [
        0.02, 0.02625, 0.31750000000000006, 0.5587500000000001, 0.495,
        0.9222222222222223, 1.0216666666666665, 0.68625, 0.54,
        0.6528571428571429,
      ],
      AU05_r: [
        0.17714285714285713, 0.030000000000000002, 0.0, 0.04125000000000001,
        0.005, 0.0, 0.0, 0.0, 0.08249999999999999, 0.11571428571428573,
      ],
      AU06_r: [
        0.20000000000000004, 0.54875, 0.8875000000000001, 0.8912499999999999,
        1.0150000000000001, 0.9755555555555557, 1.39, 1.25625,
        0.9249999999999999, 0.5457142857142857,
      ],
      AU07_r: [
        0.19, 0.5375000000000001, 0.7908333333333332, 1.1675, 1.1425,
        1.1577777777777778, 1.3608333333333336, 1.2575, 0.9225000000000001,
        0.9942857142857143,
      ],
      AU09_r: [
        0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 0.10083333333333333, 0.2775, 0.09,
        0.012857142857142857,
      ],
      AU10_r: [
        0.27714285714285714, 0.14500000000000002, 0.39333333333333337, 0.405,
        0.6375, 0.897777777777778, 1.2116666666666667, 0.89375, 0.53,
        0.31285714285714283,
      ],
      AU12_r: [
        0.4285714285714285, 0.61625, 0.5258333333333333, 0.5912499999999999,
        0.9199999999999999, 0.7677777777777778, 0.9841666666666667, 0.955,
        0.7175, 0.5142857142857143,
      ],
      AU14_r: [
        0.08142857142857143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      AU15_r: [
        0.72, 1.8324999999999998, 0.15833333333333333, 0.00875, 0.095,
        0.4111111111111111, 0.09500000000000001, 0.02375, 0.025, 0.0,
      ],
      AU17_r: [
        1.327142857142857, 1.51875, 0.31666666666666665, 0.8787499999999999,
        1.15, 0.8966666666666667, 0.31249999999999994, 0.12875, 0.1175,
        0.18142857142857144,
      ],
      AU20_r: [
        0.45999999999999996, 1.0924999999999998, 0.3658333333333333, 0.04875,
        0.0, 0.026666666666666665, 0.20166666666666666, 0.0075, 0.0, 0.0,
      ],
      AU23_r: [
        0.7942857142857144, 0.28125, 0.0, 0.0, 0.1875, 0.5555555555555556,
        0.31083333333333335, 0.0, 0.0, 0.0,
      ],
      AU25_r: [
        0.3514285714285714, 0.14, 0.5025000000000001, 0.1625,
        0.052500000000000005, 0.09888888888888889, 0.5175000000000001,
        0.8412499999999998, 0.78, 0.33285714285714285,
      ],
      AU26_r: [
        0.37714285714285717, 0.22749999999999998, 0.2541666666666667, 0.53125,
        0.25, 0.4044444444444445, 0.7566666666666665, 0.03875000000000001, 0.23,
        0.18285714285714286,
      ],
      AU45_r: [
        0.22285714285714286, 0.0, 0.0025, 0.015, 0.0, 0.0, 0.06166666666666667,
        0.11625, 0.185, 0.09999999999999999,
      ],
      F1amplitude: [
        -25.9796142578125, -77.21607208251953, -28.13727378845215,
        5.029468059539795, -6.669066905975342, -4.352915287017822,
        -86.65570068359375, -62.79671096801758, -5.797416687011719,
        2.3495986461639404,
      ],
      F1bandwidth: [
        1493.850341796875, 1662.228271484375, 1253.1856689453125,
        1302.529541015625, 1533.3612060546875, 1323.8211669921875,
        1184.103271484375, 1269.5845947265625, 1126.3819580078125,
        1399.779541015625,
      ],
      F1frequency: [
        590.2025146484375, 732.8140258789062, 605.6240844726562,
        866.2485961914062, 600.577880859375, 529.20654296875, 617.7623291015625,
        477.667724609375, 406.58282470703125, 656.6705322265625,
      ],
      HNR: [
        8.421034812927246, 4.518342018127441, 10.030532836914062,
        6.373897552490234, 6.380834579467773, 9.05953311920166,
        3.915076732635498, 3.3538572788238525, 5.082917213439941,
        5.984951019287109,
      ],
      alphaRatio: [
        -7.124314308166504, -10.509737014770508, -9.835587501525879,
        -20.348922729492188, -9.10754108428955, -12.363197326660156,
        -3.557070255279541, -5.089970588684082, -8.761316299438477,
        -14.597612380981445,
      ],
      loudness: [
        0.9687491655349731, 0.8250975608825684, 1.2241657972335815,
        0.6880991458892822, 0.47458913922309875, 0.421176940202713,
        0.46832722425460815, 0.5411249995231628, 0.4605310261249542,
        0.685849666595459,
      ],
      mfcc1: [
        37.5585823059082, 38.703372955322266, 28.842103958129883,
        42.299171447753906, 35.571128845214844, 40.50459671020508,
        12.38611125946045, 8.281085014343262, 19.307554244995117,
        42.09113693237305,
      ],
      mfcc2: [
        -30.33211898803711, -22.270971298217773, -20.78191375732422,
        -4.938215255737305, -6.814371109008789, -7.941344261169434,
        0.9046648740768433, 11.190454483032227, 6.442173480987549,
        -10.511086463928223,
      ],
      mfcc3: [
        4.204305171966553, -19.085464477539062, -2.477163553237915,
        -5.887889385223389, 3.416363000869751, 16.615476608276367,
        18.084882736206055, 10.782327651977539, 11.763980865478516,
        1.0811346769332886,
      ],
      mfcc4: [
        -16.529600143432617, -32.14363098144531, -26.883827209472656,
        -56.02228546142578, -23.051963806152344, -1.0602284669876099,
        -2.275239944458008, -8.505615234375, -3.7007880210876465,
        -54.559452056884766,
      ],
      pitch: [
        31.991926193237305, 22.681396484375, 33.04902267456055,
        33.57320785522461, 33.50397491455078, 32.747581481933594,
        18.861509323120117, 23.020832061767578, 31.651390075683594,
        29.260404586791992,
      ],
    },
  },
  msg: "success",
  result: {
    defended: {
      MMIN: -0.6156,
      T2FN: 0.3488,
      TFR_Net: -0.099,
      TPFN: -0.0053,
    },
    modified: {
      MMIN: -0.6157,
      T2FN: 0.3076,
      TFR_Net: -0.042,
      TPFN: 0.0097,
    },
    original: {
      MMIN: -0.6157,
      T2FN: 0.6326,
      TFR_Net: -0.0834,
      TPFN: 0.005,
    },
  },
});
const methodData = ref(null);
const modifiedTextList = ref([
  {
    conf: 0.9924983864489116,
    end: 0.25155598958333336,
    start: 0.05031119791666667,
    text: "now",
  },
  {
    conf: 0.7846563020477458,
    end: 0.4729252604166667,
    start: 0.25155598958333336,
    text: "the",
  },
  {
    conf: 0.6531528330428251,
    end: 0.8351658854166667,
    start: 0.4729252604166667,
    text: "title",
  },
  {
    conf: 0.47641091293124294,
    end: 1.0766596354166666,
    start: 0.8351658854166667,
    text: "of",
  },
  {
    conf: 0.24061251992673066,
    end: 1.1974065104166667,
    start: 1.0766596354166666,
    text: "the",
  },
  {
    conf: 0.5447095983243366,
    end: 1.4590247395833333,
    start: 1.1974065104166667,
    text: "movie",
  },
  {
    conf: 0.6084983060602999,
    end: 1.8212653645833332,
    start: 1.4590247395833333,
    text: "baseke",
  },
  {
    conf: 0.6754744243939316,
    end: 2.0426346354166665,
    start: 1.8212653645833332,
    text: "sets",
  },
  {
    conf: 0.47667267739500935,
    end: 2.14325703125,
    start: 2.0426346354166665,
    text: "it",
  },
  {
    conf: 0.6783241011170195,
    end: 2.3646263020833334,
    start: 2.14325703125,
    text: "all",
  },
]);
const editAligned = ref([]);
const asrNosie = ref(false);
const defenceVideoUrl = ref("1.mp4");
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
  time[0] = item[0].toFixed(4);
  time[1] = item[1].toFixed(4);
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
onMounted(() => {
  window.addEventListener("mousedown", mouseDown);
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
  padding-bottom: 150px;
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