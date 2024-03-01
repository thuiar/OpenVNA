<template>
  <div class="addNoiseContainer">
    <el-card shadow="always" class="methodContent fromMethod">
      <div class="methodContainer">
        <el-form
          v-if="selectModality == 'audio'"
          :model="a_form_item"
          label-width="80px"
        >
          <el-form-item label="Modality">
            <el-radio-group v-model="selectModality">
              <el-radio label="audio" />
              <el-radio label="video" />
            </el-radio-group>
          </el-form-item>
          <el-form-item label="Start time">
            <el-input
              v-model="a_form_item.start"
              placeholder="Please input start time"
              class="oneInput"
              type="text"
            />
          </el-form-item>
          <el-form-item label="End time">
            <el-input
              v-model="a_form_item.end"
              placeholder="Please input end time"
              class="oneInput"
              type="text"
            />
          </el-form-item>
          <el-form-item label="Type">
            <template #label>
              <span>Type</span>
              <el-tooltip
                class="box-item"
                effect="dark"
                placement="right-start"
              >
                <template #content>
                  <div class="boxItemTip">
                    Supported modes: "volume", "coloran", "background",
                    "sudden", "lowpass", "reverb"
                  </div>
                </template>
                <el-icon class="questionFilled">
                  <QuestionFilled />
                </el-icon>
              </el-tooltip>
            </template>

            <el-select
              v-model="a_form_item.type"
              class="oneInput"
              placeholder="please select your zone"
            >
              <el-option
                v-for="item in options_details['audio']['type']"
                :key="item"
                :label="item"
                :value="item"
              />
            </el-select>
          </el-form-item>
          <el-form-item
            v-if="
              a_form_item.type == 'coloran' ||
              a_form_item.type == 'background' ||
              a_form_item.type == 'sudden' ||
              a_form_item.type == 'reverb'
            "
          >
            <template #label>
              <span>Option</span>
              <el-tooltip
                class="box-item"
                effect="dark"
                placement="right-start"
              >
                <template #content>
                  <div class="boxItemTip">
                    {{ boxItemTip }}
                  </div>
                </template>
                <el-icon class="questionFilled">
                  <QuestionFilled />
                </el-icon>
              </el-tooltip>
            </template>
            <el-select
              v-model="a_form_item.option[a_form_item.type][0]"
              class="halfSelect"
              placeholder="please select"
            >
              <el-option
                v-for="item in options_details['audio'][a_form_item.type]"
                :key="item"
                :label="item"
                :value="item"
              />
            </el-select>
            --
            <el-input
              v-model="a_form_item.option[a_form_item.type][1]"
              placeholder="num"
              class="halfInput"
              type="text"
            />
          </el-form-item>
          <el-form-item v-if="a_form_item.type == 'volume'">
            <template #label>
              <span>Option</span>
              <el-tooltip
                class="box-item"
                effect="dark"
                placement="right-start"
              >
                <template #content>
                  <div class="boxItemTip">
                    {{ boxItemTip }}
                  </div>
                </template>
                <el-icon class="questionFilled">
                  <QuestionFilled />
                </el-icon>
              </el-tooltip>
            </template>
            <el-input
              v-model="a_form_item.option[a_form_item.type]"
              placeholder="0 to 1"
              class="oneInput"
              type="text"
            />
          </el-form-item>
          <el-form-item v-if="a_form_item.type == 'lowpass'" label="Lowpass">
            <template #label>
              <span>Option</span>
              <el-tooltip
                class="box-item"
                effect="dark"
                placement="right-start"
              >
                <template #content>
                  <div class="boxItemTip">
                    {{ boxItemTip }}
                  </div>
                </template>
                <el-icon class="questionFilled">
                  <QuestionFilled />
                </el-icon>
              </el-tooltip>
            </template>
            <el-input
              v-model="a_form_item.option[a_form_item.type]"
              placeholder="eg. 500"
              class="oneInput"
              type="text"
            />
          </el-form-item>
          <el-form-item>
            <el-button
              type="success"
              class="oneInput"
              @click="addNoiseItem"
              plain
              >Add</el-button
            >
          </el-form-item>
        </el-form>
        <el-form v-else :model="v_form_item" label-width="80px">
          <el-form-item label="Modality">
            <el-radio-group v-model="selectModality">
              <el-radio label="audio" />
              <el-radio label="video" />
            </el-radio-group>
          </el-form-item>
          <el-form-item label="Start time">
            <el-input
              v-model="v_form_item.start"
              class="oneInput"
              placeholder="Please input start time"
              type="text"
            />
          </el-form-item>
          <el-form-item label="End time">
            <el-input
              v-model="v_form_item.end"
              class="oneInput"
              placeholder="Please input end time"
              type="text"
            />
          </el-form-item>
          <el-form-item>
            <template #label>
              <span>Type</span>
              <el-tooltip
                class="box-item"
                effect="dark"
                placement="right-start"
              >
                <template #content>
                  <div class="boxItemTip">
                    Supported modes: "blank", "avgblur", "gblur",
                    "impulse_value", "occlusion", "color",
                    "color_channel_swapping", "color_inversion".
                  </div>
                </template>
                <el-icon class="questionFilled">
                  <QuestionFilled />
                </el-icon>
              </el-tooltip>
            </template>
            <el-select
              v-model="v_form_item.type"
              placeholder="please select your zone"
              class="oneInput"
            >
              <el-option
                v-for="item in options_details['video']['type']"
                :key="item"
                :label="item"
                :value="item"
              />
            </el-select>
          </el-form-item>
          <el-form-item
            v-if="
              v_form_item.type == 'impulse_value' || v_form_item.type == 'gblur'
            "
          >
            <template #label>
              <span>Option</span>
              <el-tooltip
                class="box-item"
                effect="dark"
                placement="right-start"
              >
                <template #content>
                  <div class="boxItemTip">
                    {{ boxItemTip }}
                  </div>
                </template>
                <el-icon class="questionFilled">
                  <QuestionFilled />
                </el-icon>
              </el-tooltip>
            </template>
            <el-input
              v-model="v_form_item.option.value"
              placeholder="> 0"
              type="text"
              class="oneInput"
            />
          </el-form-item>
          <el-form-item v-if="v_form_item.type == 'avgblur'">
            <template #label>
              <span>Option</span>
              <el-tooltip
                class="box-item"
                effect="dark"
                placement="right-start"
              >
                <template #content>
                  <div class="boxItemTip">
                    {{ boxItemTip }}
                  </div>
                </template>
                <el-icon class="questionFilled">
                  <QuestionFilled />
                </el-icon>
              </el-tooltip>
            </template>
            <el-input
              v-model="v_form_item.option.avgblur[0]"
              placeholder="sizeX"
              type="text"
              class="optionHalfInput"
            />-
            <el-input
              v-model="v_form_item.option.avgblur[1]"
              placeholder="sizeY"
              type="text"
              class="optionHalfInput"
            />
          </el-form-item>
          <el-form-item v-if="v_form_item.type == 'color'">
            <template #label>
              <span>Option</span>
              <el-tooltip
                class="box-item"
                effect="dark"
                placement="right-start"
              >
                <template #content>
                  <div class="boxItemTip">
                    {{ boxItemTip }}
                  </div>
                </template>
                <el-icon class="questionFilled">
                  <QuestionFilled />
                </el-icon>
              </el-tooltip>
            </template>
            <el-input
              v-model="v_form_item.option.color[0]"
              placeholder="con..."
              type="text"
              class="optionThirdInput"
            />-
            <el-input
              v-model="v_form_item.option.color[1]"
              placeholder="bri..."
              type="text"
              class="optionThirdInput"
            />
            -
            <el-input
              v-model="v_form_item.option.color[2]"
              placeholder="sat..."
              type="text"
              class="optionThirdInput"
            />
            <el-input
              v-model="v_form_item.option.color[3]"
              placeholder="gam_r"
              type="text"
              style="margin-top: 5px"
              class="optionThirdInput"
            />-
            <el-input
              v-model="v_form_item.option.color[4]"
              placeholder="gam_g"
              type="text"
              style="margin-top: 5px"
              class="optionThirdInput"
            />
            -
            <el-input
              v-model="v_form_item.option.color[5]"
              placeholder="gam_b"
              type="text"
              style="margin-top: 5px"
              class="optionThirdInput"
            />
          </el-form-item>
          <el-form-item v-if="v_form_item.type == 'occlusion'">
            <template #label>
              <span>Option</span>
              <el-tooltip
                class="box-item"
                effect="dark"
                placement="right-start"
              >
                <template #content>
                  <div class="boxItemTip">
                    {{ boxItemTip }}
                  </div>
                </template>
                <el-icon class="questionFilled">
                  <QuestionFilled />
                </el-icon>
              </el-tooltip>
            </template>
            <el-input
              v-model="v_form_item.option.occlusion[0]"
              placeholder="x"
              type="text"
              class="optionHalfInput"
            />-
            <el-input
              v-model="v_form_item.option.occlusion[1]"
              placeholder="y"
              type="text"
              class="optionHalfInput"
            />
            <el-input
              v-model="v_form_item.option.occlusion[2]"
              placeholder="w"
              type="text"
              style="margin-top: 5px"
              class="optionHalfInput"
            />-
            <el-input
              v-model="v_form_item.option.occlusion[3]"
              placeholder="h"
              type="text"
              style="margin-top: 5px"
              class="optionHalfInput"
            />
          </el-form-item>
          <el-form-item
            v-if="v_form_item.type == 'color_channel_swapping'"
          >
          <template #label>
              <span>Option</span>
              <el-tooltip
                class="box-item"
                effect="dark"
                placement="right-start"
              >
                <template #content>
                  <div class="boxItemTip">
                    {{ boxItemTip }}
                  </div>
                </template>
                <el-icon class="questionFilled">
                  <QuestionFilled />
                </el-icon>
              </el-tooltip>
            </template>
            <el-select
              v-model="v_form_item.option.color_channel_swapping[0]"
              class="optionHalfInput"
              placeholder="[r,g,b]"
            >
              <el-option
                v-for="item in options_details['video'][
                  'color_channel_swapping'
                ]"
                :key="item"
                :label="item"
                :value="item"
              /> </el-select
            >-
            <el-select
              v-model="v_form_item.option.color_channel_swapping[1]"
              class="optionHalfInput"
              placeholder="[r,g,b]"
            >
              <el-option
                v-for="item in options_details['video'][
                  'color_channel_swapping'
                ]"
                :key="item"
                :label="item"
                :value="item"
              />
            </el-select>
          </el-form-item>
          <el-form-item>
            <el-button
              type="success"
              class="oneInput"
              @click="addNoiseItem"
              plain
              >Add</el-button
            >
          </el-form-item>
        </el-form>
      </div>
    </el-card>
  </div>
</template>
  
<script setup>
import { ref, computed } from "vue";
import { ElMessage } from "element-plus";
import OperationMethod from "../utils/operation.js";
const selectModality = ref("audio");
const a_form_item = ref({
  type: "",
  start: "",
  end: "",
  option: {
    volume: "",
    lowpass: "",
    reverb: [,],
    sudden: [,],
    background: [,],
    coloran: [,],
    mute: "",
  },
});
const v_form_item = ref({
  type: "",
  start: "",
  end: "",
  option: {
    color_channel_swapping: [,],
    avgblur: [,],
    gblur: "",
    impulse_value: "",
    occlusion: [, , ,],
    color: [, , , , ,],
    blank: "",
    color_inversion: "",
  },
});
const options_details = ref({
  audio: {
    coloran: [
      "white",
      "pink",
      "brown",
      "blue",
      "violet",
      "grey",
      "brown",
      "violet",
    ],
    background: [
      "random",
      "metro",
      "office",
      "park",
      "restaurant",
      "traffic",
      "white",
      "music_soothing",
      "music_tense",
      "song_gentle",
      "song_rock",
    ],
    sudden: ["random", "beep", "glass", "thunder", "dog"],
    reverb: ["hall", "room"],
    others_required: ["volume"],
    no_required: ["mute"],
    type: [
      "mute",
      "volume",
      "lowpass",
      "reverb",
      "sudden",
      "background",
      "coloran",
    ],
  },
  video: {
    color_channel_swapping: ["r", "g", "b"],
    type: [
      "blank",
      "avgblur",
      "gblur",
      "impulse_value",
      "occlusion",
      "color",
      "color_channel_swapping",
      "color_inversion",
    ],
    no_required: ["blank", "color_inversion"],
  },
});
const emit = defineEmits(["addNoise"]);
const boxItemTip = computed(() => {
  let from_item_type =
    selectModality.value == "audio"
      ? a_form_item.value.type
      : v_form_item.value.type;
  return OperationMethod.noiseTip(from_item_type);
});
const addNoiseItem = () => {
  let nosieItem = [, , ,];
  if (selectModality.value == "audio") {
    nosieItem[0] = [a_form_item.value.start, a_form_item.value.end];
    nosieItem[1] = "a";
    nosieItem[2] = a_form_item.value.type;
    nosieItem[3] = a_form_item.value.option[a_form_item.value.type];

    if (
      a_form_item.value.type == "volume" ||
      a_form_item.value.type == "lowpass"
    ) {
      if (nosieItem[3] == "") {
        ElMessage({
          message: "The option cannot be empty.",
          type: "warning",
        });
        return;
      }
    } else if (
      a_form_item.value.type == "reverb" ||
      a_form_item.value.type == "sudden" ||
      a_form_item.value.type == "background" ||
      a_form_item.value.type == "coloran"
    ) {
      if (nosieItem[3][0] == "" || nosieItem[3][1] == "") {
        ElMessage({
          message: "The option cannot be empty.",
          type: "warning",
        });
        return;
      }
    }
  } else if (selectModality.value == "video") {
    nosieItem[0] = [v_form_item.value.start, v_form_item.value.end];
    nosieItem[1] = "v";
    nosieItem[2] = v_form_item.value.type;
    nosieItem[3] = v_form_item.value.option[v_form_item.value.type];

    if (
      v_form_item.value.type == "gblur" ||
      v_form_item.value.type == "impulse_value"
    ) {
      if (nosieItem[3] == "") {
        ElMessage({
          message: "The option cannot be empty.",
          type: "warning",
        });
        return;
      }
    } else if (
      v_form_item.value.type == "avgblur" ||
      v_form_item.value.type == "color_channel_swapping"
    ) {
      if (nosieItem[3][0] == "" || nosieItem[3][1] == "") {
        ElMessage({
          message: "The option cannot be empty.",
          type: "warning",
        });
        return;
      }
    } else if (v_form_item.value.type == "color") {
      if (
        nosieItem[3][0] == "" ||
        nosieItem[3][1] == "" ||
        nosieItem[3][2] == "" ||
        nosieItem[3][3] == "" ||
        nosieItem[3][4] == "" ||
        nosieItem[3][5] == ""
      ) {
        ElMessage({
          message: "The option cannot be empty.",
          type: "warning",
        });
        return;
      }
    } else if (v_form_item.value.type == "occlusion") {
      if (
        nosieItem[3][0] == "" ||
        nosieItem[3][1] == "" ||
        nosieItem[3][2] == "" ||
        nosieItem[3][3] == ""
      ) {
        ElMessage({
          message: "The option cannot be empty.",
          type: "warning",
        });
        return;
      }
    }
  }
  if (nosieItem[0][0] == "" || nosieItem[0][1] == "") {
    ElMessage({
      message: "Please fill in the time.",
      type: "warning",
    });
    return;
  } else if (nosieItem[2] == "") {
    ElMessage({
      message: "Please fill in the noise type.",
      type: "warning",
    });
    return;
  }
  a_form_item.value = {
    type: "",
    start: "",
    end: "",
    option: {
      volume: "",
      lowpass: "",
      reverb: [,],
      sudden: [,],
      background: [,],
      coloran: [,],
      mute: "",
    },
  };
  v_form_item.value = {
    type: "",
    start: "",
    end: "",
    option: {
      color_channel_swapping: [,],
      avgblur: [,],
      gblur: "",
      impulse_value: "",
      occlusion: [, , ,],
      color: [, , , , ,],
      blank: "",
      color_inversion: "",
    },
  };
  emit("addNoise", nosieItem);
};
</script>
  
<style scoped>
.methodContent {
  width: 320px;
  overflow: hidden;
  margin-top: 10px;
}

.fromMethod {
  height: auto;
}

.fromMethod .el-form-item {
  margin-bottom: 10px;
}

.halfSelect {
  width: 140px;
}
.halfInput {
  width: 57px;
}

.optionHalfInput {
  width: 102px;
}
.optionThirdInput {
  width: 65.3px;
}
.oneInput {
  width: 210px;
}
.boxItemTip {
  width: 200px;
}
.questionFilled {
  color: #bbbbbb;
  height: 33px;
  margin-left: 5px;
}
</style>