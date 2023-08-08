<template>
  <div class="appContainer">
    <div style="margin-top: 20px; font-size: 23px">OpenVNA Platform</div>
    <el-divider />
    <div class="pageContainer" style="position: relative">
      <div style="font-size: 19px">Sample introduction</div>
      <el-form-item label="Sample select" style="margin-top: 20px">
        <el-select
          v-model="displayType"
          placeholder="please select your zone"
          @change="changeDisplayType"
          style="width: 200px"
        >
          <el-option label="no_re_asr" value="no_re_asr" />
          <el-option label="re_asr" value="re_asr" />
          <el-option label="video_gpt" value="video_gpt" />
        </el-select>
      </el-form-item>
      <div
        class="previewTop"
        style="margin-top: 10px"
        v-if="displayType != 'video_gpt'"
      >
        <div class="topTip">1. Model selection</div>
        <div class="topDescribe">
          Choose the sentiment analysis algorithm model, and choose the recovery
          method if there is noise data.
        </div>
      </div>
      <select-method
        v-if="displayType != 'video_gpt'"
        class="selectMethod"
        :disabled="true"
        @transmitMethods="methodList = $event"
      />
    </div>
    <div class="pageContainer" v-if="displayResult">
      <div class="previewTop">
        <div
          class="topTip"
          v-if="displayType == 'video_gpt'"
          style="margin-top: 15px"
        >
          1. Result presentation
        </div>
        <div class="topTip" v-else>2. Result presentation</div>
        <div class="topDescribe">
          We also present raw data versus noise data.
        </div>
      </div>
      <BigModal v-if="displayType == 'video_gpt'" />
      <view-result
        v-if="displayType != 'video_gpt'"
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
    <div class="nosieDetails" v-if="displayType != 'video_gpt'">
      <div class="topDescribe">
        <el-icon class="info"><InfoFilled /></el-icon>Display noise type
        details.
      </div>
      <div class="topDescribe" v-if="displayType == 'no_re_asr'">
        Occlusion: (x, y, w, h) of occlusion box. If mode is "percent", values
        are given as percentage of video size. Otherwise, they are given as
        pixels.
      </div>
      <div class="topDescribe" v-if="displayType == 're_asr'">
        Background: (filename, volume) of background noise. Filename is the name
        of background noise file. Supported Values are ["random", "metro",
        "office", "park", "restaurant", "traffic", "white", "music_soothing",
        "music_tense", "song_gentle", "song_rock"]. If filename is "random", a
        random background noise will be selected. Volume is a float number
        greater than 0. Background noise is applied to the specified time range.
        Currently, the background noise is not looped. Thus if the noise is
        shorter, it will not cover the whole time range. A random part of the
        noise will be selected if the noise file is longer. Most noise files are
        about 5 minutes long, except for the music and songs which are about 3
        minutes long.
      </div>
    </div>
    <el-divider />
    <div class="pageContainer" style="display: flex; flex-direction: row">
      <div class="previewTop">
        <div style="font-size: 19px">Get started</div>
        <div class="topDescribe">You can enjoy our platform here.</div>
      </div>
      <el-button
        type="success"
        @click="start"
        style="width: 200px; margin-top: 10px; margin-left: 80px"
        >Start</el-button
      >
    </div>
  </div>
</template>
  <script setup>
import { ref, onMounted } from "vue";
import SelectMethod from "@/components/SelectMethod.vue";
import BigModal from "@/components/BigModal.vue";
import ViewResult from "@/components/ViewResult.vue";
import { useRouter } from "vue-router";
const router = useRouter();
const noReAsrEditAligned = [
  [[0.675, 1.0982], "a", "mute", ""],
  [[1.0982, 1.6422], "v", "occlusion", ["10", "10", "100", "100"]],
  [[1.6422, 2.2467, 7], "t", "remove", ""],
];
const reAsrEditAligned = [
  [[1.6422, 2.2064], "a", "mute", ""],
  [[0.675, 1.0982], "a", "background", ["restaurant", "8"]],
  [[1.0982, 1.6422], "v", "color_inversion", ""],
];
const displayType = ref("no_re_asr");
const asrNosie = ref(false);
const displayResult = ref(false);
const start = () => {
  router.push({ path: "/viedoeditor" });
};
const currentVideoUrl = ref("");
const McurrentVideoUrl = ref("");
const duration = ref(2.466667);
const methodList = ref({
  defence: ["a_denoise", "v_reconstruct"],
  models: ["MMIN", "TFR_Net", "T2FN", "TPFN"],
});
const viewResults = ref(null);
const methodData = ref(null);
const modifiedTextList = ref([]);
const editAligned = ref([]);
const defenceVideoUrl = ref("");
const loadData = async (type_str) => {
  fetch(`samples/${type_str}/aligned_modified.json`)
    .then((response) => response.json())
    .then((data) => {
      modifiedTextList.value = data;
    })
    .catch((error) => console.log("请求失败。", error));

  fetch(`samples/${type_str}/test_result.json`)
    .then((response) => response.json())
    .then((data) => {
      viewResults.value = data;
      displayResult.value = true;
    })
    .catch((error) => console.log("请求失败。", error));

  currentVideoUrl.value = `samples/${type_str}/raw_video.mp4`;
  McurrentVideoUrl.value = `samples/${type_str}/modified_video.mp4`;
  defenceVideoUrl.value = `samples/${type_str}/defended_video.mp4`;
};

onMounted(() => {
  loadData(displayType.value);
  editAligned.value = noReAsrEditAligned;
});
const changeDisplayType = () => {
  displayResult.value = false;
  if (displayType.value == "re_asr") {
    loadData("re_asr");
    editAligned.value = reAsrEditAligned;
    asrNosie.value = true;
  } else if (displayType.value == "no_re_asr") {
    loadData("no_re_asr");
    editAligned.value = noReAsrEditAligned;
    asrNosie.value = false;
  } else {
    displayResult.value = true;
  }
};
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
  padding-bottom: 50px;
}
.pageContainer {
  width: 800px;
}
.selectMethod {
  position: relative;
  top: 10px;
}
.nosieDetails {
  width: 710px;
}
.topDescribe {
  font-size: 14px;
  color: #727272;
  margin-top: 5px;
  display: flex;
  flex-direction: row;
  align-items: center;
  & .info {
    margin-right: 7px;
  }
  & span {
    font-weight: bold;
  }
}
.previewTop {
  padding-bottom: 10px;
}
.topTip {
  font-size: 17px;
  color: #353535;
}
</style> 