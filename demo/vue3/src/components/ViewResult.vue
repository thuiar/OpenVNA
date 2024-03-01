<template>
  <div class="viewResultContainer">
    <div class="topContainer">
      <div class="videoContainer">
        <div class="labelContainer" style="margin-top: 2px">
          <span class="labelTitle">Original:</span>
          <el-tooltip
            class="box-item"
            effect="dark"
            content="click to view scores from each model"
            placement="right"
          >
            <span
              class="labelContent"
              v-if="labelResult.original > 0.5"
              style="color: rgb(60, 190, 158)"
              @click="dialogScores = !dialogScores"
              >Positive</span
            >
            <span
              class="labelContent"
              v-else-if="labelResult.original >= -0.5"
              style="color: rgb(172, 175, 2)"
              @click="dialogScores = !dialogScores"
              >Neutral</span
            >
            <span
              class="labelContent"
              v-else
              style="color: rgb(241, 76, 76)"
              @click="dialogScores = !dialogScores"
              >Negative</span
            >
          </el-tooltip>
        </div>
        
        <div class="viewOriginal">
          <video
            id="viewOriginal"
            class="video-js vjs-big-play-centered"
            controls
            data-setup="{}"
          >
            <source :src="originalVideoUrl" type="video/mp4" />
          </video>
        </div>
        <div class="labelContainer" style="margin-top: 10px">
          <span
            class="labelTitle"
            v-if="viewResults.result.defended == undefined"
            >Modified:</span
          >
          <el-select
            v-else
            v-model="modifiedDefence"
            @change="defenceChange"
            class="modifiedDefence"
          >
            <el-option
              v-for="item in modifiedDefenceList"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-tooltip
            class="box-item"
            effect="dark"
            content="click to view scores from each model"
            placement="right"
          >
            <div v-if="modifiedDefence == 'Modified'">
              <span
                class="labelContent"
                v-if="labelResult.modified > 0.5"
                style="color: rgb(60, 190, 158)"
                @click="dialogScores = !dialogScores"
                >Positive</span
              >
              <span
                class="labelContent"
                v-else-if="labelResult.modified >= -0.5"
                style="color: rgb(172, 175, 2)"
                @click="dialogScores = !dialogScores"
                >Neutral</span
              >
              <span
                class="labelContent"
                v-else
                style="color: rgb(241, 76, 76)"
                @click="dialogScores = !dialogScores"
                >Negative</span
              >
            </div>
            <div v-else>
              <span
                class="labelContent"
                v-if="labelResult.defended > 0.5"
                style="color: rgb(60, 190, 158)"
                @click="dialogScores = !dialogScores"
                >Positive</span
              >
              <span
                class="labelContent"
                v-else-if="labelResult.defended >= -0.5"
                style="color: rgb(172, 175, 2)"
                @click="dialogScores = !dialogScores"
                >Neutral</span
              >
              <span
                class="labelContent"
                v-else
                style="color: rgb(241, 76, 76)"
                @click="dialogScores = !dialogScores"
                >Negative</span
              >
            </div>
          </el-tooltip>
        </div>
        <div class="viewModified" id="viewModifiedContainer">
          <video
            id="viewModified"
            class="video-js vjs-big-play-centered"
            controls
            data-setup="{}"
          >
            <source :src="videoUrl" type="video/mp4" />
          </video>
        </div>
        
      </div>
      <div class="featureContainer">
        <div class="selectContainer">
          <div class="selectTitle">Choose feature to view:</div>
          <el-select
            v-model="methodValue"
            @change="chooseClick('one')"
            class="methodOne"
          >
            <el-option
              v-for="item in methodList"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-select
            v-model="methodDetailValue"
            @change="chooseClick('two')"
            class="methodTwo"
          >
            <el-option
              v-for="item in methodDetailList"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-tooltip class="box-item" effect="dark" placement="right-start">
            <template #content>
              <div class="boxItemTip">{{ boxItemTip }}</div>
            </template>
            <el-icon class="questionFilled">
              <QuestionFilled />
            </el-icon>
          </el-tooltip>
        </div>
        <div style="position: relative; margin-left: -15px">
          <div
            id="main"
            style="width: 550px; height: 350px; margin-top: -20px"
          ></div>
        </div>
        <div class="topDescribe" style="margin-top: -40px; margin-left: 15px">
          Original audio visualization. unit of time: second.
        </div>
        <div class="waveform">
          <div id="originalWaveTimeline" ref="originalWaveTimeline"></div>
          <div
            id="originalWaveform"
            @click="originalSeekToTime(null)"
            ref="originalWaveform"
          ></div>
        </div>
        <div class="topDescribe" style="margin-top: 15px; margin-left: 15px">
          Modified/Defended audio visualization. unit of time: second.
        </div>
        <div class="waveform" v-if="modifiedWaveSwitch">
          <div id="modifiedTimeline" ref="modifiedTimeline"></div>
          <div
            id="modifiedWave"
            @click="modifiedSeekToTime(null)"
            ref="modifiedWave"
          ></div>
        </div>
        <el-checkbox
          v-model="asrNosie"
          label="If selected, text changes will only be affected by audio noise."
          class="asrCheckbox"
          disabled
          size="large"
        />
      </div>
    </div>
    <div class="bottomContainer">
      <TableModified
        :editAligned="editAligned"
        :onlyDisplay="true"
        style="margin-top: -5px"
      />
      <el-divider  border-style="dashed" />
      
      <div class="textContent">
        <span
          @click="modifiedSeekToTime(textitem.start)"
          v-for="(textitem, index) in textList"
          :key="textitem"
          :id="index"
          >{{ textitem.text }}</span
        >
      </div>
      <div class="audioContainer">
        <el-slider
          v-model="sliderValue"
          :format-tooltip="formatTooltip"
          @input="sliderChange"
        />
        <div class="control">
          <svg
            t="1662145678112"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            @click="playVideo('start')"
            p-id="2970"
            width="100"
            height="100"
          >
            <path
              d="M554.666667 512a53.44 53.44 0 0 0 17.4 39.413333l362.666666 330.6a52.833333 52.833333 0 0 0 35.713334 14 53.84 53.84 0 0 0 21.76-4.666666 52.666667 52.666667 0 0 0 31.793333-48.793334V181.406667A53.333333 53.333333 0 0 0 934.74 142l-362.666667 330.6A53.44 53.44 0 0 0 554.666667 512z m-451.933334-39.413333l362.666667-330.6A53.333333 53.333333 0 0 1 554.666667 181.406667v661.186666a52.666667 52.666667 0 0 1-31.793334 48.793334 53.84 53.84 0 0 1-21.76 4.666666 52.833333 52.833333 0 0 1-35.713333-14l-362.666667-330.6a53.333333 53.333333 0 0 1 0-78.826666zM0 874.666667V149.333333a21.333333 21.333333 0 0 1 42.666667 0v725.333334a21.333333 21.333333 0 0 1-42.666667 0z"
              fill="#5C5C66"
              p-id="2971"
            />
          </svg>
          <svg
            t="1662145745021"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            @click="playVideo('last')"
            p-id="4610"
            width="100"
            height="100"
          >
            <path
              d="M213.333333 512a52.92 52.92 0 0 0 25.78 45.666667l618.666667 373.28a53.333333 53.333333 0 0 0 80.886667-45.666667V138.72a53.333333 53.333333 0 0 0-80.886667-45.666667L239.133333 466.333333A52.92 52.92 0 0 0 213.333333 512z m-128 405.333333V106.666667a21.333333 21.333333 0 0 1 42.666667 0v810.666666a21.333333 21.333333 0 0 1-42.666667 0z"
              fill="#5C5C66"
              p-id="4611"
            />
          </svg>
          <svg
            t="1662146562563"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            v-if="mediaPaused"
            @click="playVideo('play')"
            p-id="7407"
            width="100"
            height="100"
          >
            <path
              d="M128 138.666667c0-47.232 33.322667-66.666667 74.176-43.562667l663.146667 374.954667c40.96 23.168 40.853333 60.8 0 83.882666L202.176 928.896C161.216 952.064 128 932.565333 128 885.333333v-746.666666z"
              fill="#3D3D3D"
              p-id="7408"
            />
          </svg>
          <svg
            v-else
            t="1662554869612"
            @click="playVideo('play')"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="2207"
            width="100"
            height="100"
          >
            <path
              d="M426.666667 138.666667v746.666666a53.393333 53.393333 0 0 1-53.333334 53.333334H266.666667a53.393333 53.393333 0 0 1-53.333334-53.333334V138.666667a53.393333 53.393333 0 0 1 53.333334-53.333334h106.666666a53.393333 53.393333 0 0 1 53.333334 53.333334z m330.666666-53.333334H650.666667a53.393333 53.393333 0 0 0-53.333334 53.333334v746.666666a53.393333 53.393333 0 0 0 53.333334 53.333334h106.666666a53.393333 53.393333 0 0 0 53.333334-53.333334V138.666667a53.393333 53.393333 0 0 0-53.333334-53.333334z"
              fill="#5C5C66"
              p-id="2208"
            />
          </svg>
          <svg
            t="1662145758367"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            @click="playVideo('next')"
            p-id="5545"
            width="100"
            height="100"
          >
            <path
              d="M810.666667 512a52.92 52.92 0 0 1-25.78 45.666667l-618.666667 373.28a53.333333 53.333333 0 0 1-80.886667-45.666667V138.72a53.333333 53.333333 0 0 1 80.886667-45.666667l618.666667 373.28A52.92 52.92 0 0 1 810.666667 512z m128 405.333333V106.666667a21.333333 21.333333 0 0 0-42.666667 0v810.666666a21.333333 21.333333 0 0 0 42.666667 0z"
              fill="#5C5C66"
              p-id="5546"
            />
          </svg>
          <svg
            t="1662145724176"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="3756"
            width="100"
            height="100"
            @click="playVideo('end')"
          >
            <path
              d="M469.335173 512a53.44 53.44 0 0 1-17.4 39.413333L89.268507 882a52.833333 52.833333 0 0 1-35.713334 14 53.84 53.84 0 0 1-21.76-4.666667A52.666667 52.666667 0 0 1 0.00184 842.593333V181.406667a52.666667 52.666667 0 0 1 31.793333-48.793334 52.666667 52.666667 0 0 1 57.466667 9.386667l362.666667 330.6A53.44 53.44 0 0 1 469.335173 512z m451.933334-39.413333L558.595173 142A53.333333 53.333333 0 0 0 469.335173 181.406667v661.186666a52.666667 52.666667 0 0 0 31.793334 48.793334 53.84 53.84 0 0 0 21.76 4.666666 52.833333 52.833333 0 0 0 35.713333-14l362.666667-330.6a53.333333 53.333333 0 0 0 0-78.826666zM1024.00184 874.666667V149.333333a21.333333 21.333333 0 0 0-42.666667 0v725.333334a21.333333 21.333333 0 0 0 42.666667 0z"
              fill="#5C5C66"
              p-id="3757"
            />
          </svg>
        </div>
      </div>
      
    </div>
    <el-dialog v-model="dialogScores" title="Model Scores" width="650px">
      <div class="dialogContainer">
        <div class="dialogScoresContainer">
          <div>Original</div>
          <div
            class="dialogItem"
            v-if="viewResults.result.original.T2FN != undefined"
          >
            <div class="dialogTitle">T2FN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.original.T2FN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.original.T2FN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.original.T2FN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.original.TPFN != undefined"
          >
            <div class="dialogTitle">TPFN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.original.TPFN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.original.TPFN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.original.TPFN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.original.CTFN != undefined"
          >
            <div class="dialogTitle">CTFN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.original.CTFN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.original.CTFN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.original.CTFN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.original.MMIN != undefined"
          >
            <div class="dialogTitle">MMIN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.original.MMIN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.original.MMIN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.original.MMIN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.original.TFRNet != undefined"
          >
            <div class="dialogTitle">TFR-Net:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.original.TFRNet < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.original.TFRNet < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.original.TFRNet"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.original.GCNET != undefined"
          >
            <div class="dialogTitle">GCNET:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.original.GCNET < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.original.GCNET < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.original.GCNET"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.original.NIAT != undefined"
          >
            <div class="dialogTitle">NIAT:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.original.NIAT < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.original.NIAT < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.original.NIAT"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.original.EMT_DLFR != undefined"
          >
            <div class="dialogTitle">EMT-DLFR:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.original.EMT_DLFR < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.original.EMT_DLFR < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.original.EMT_DLFR"
            />
          </div>
          <div class="dialogItem" v-if="labelResult.original != undefined">
            <div class="dialogTitle">Average:</div>
            <el-input
              :style="{
                backgroundColor:
                  labelResult.original < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : labelResult.original < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="labelResult.original"
            />
          </div>
        </div>
        <div class="dialogScoresContainer">
          <div>Modified</div>
          <div
            class="dialogItem"
            v-if="viewResults.result.modified.T2FN != undefined"
          >
            <div class="dialogTitle">T2FN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.modified.T2FN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.modified.T2FN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.modified.T2FN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.modified.TPFN != undefined"
          >
            <div class="dialogTitle">TPFN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.modified.TPFN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.modified.TPFN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.modified.TPFN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.modified.CTFN != undefined"
          >
            <div class="dialogTitle">CTFN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.modified.CTFN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.modified.CTFN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.modified.CTFN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.modified.MMIN != undefined"
          >
            <div class="dialogTitle">MMIN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.modified.MMIN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.modified.MMIN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.modified.MMIN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.modified.TFRNet != undefined"
          >
            <div class="dialogTitle">TFR-Net:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.modified.TFRNet < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.modified.TFRNet < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.modified.TFRNet"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.modified.GCNET != undefined"
          >
            <div class="dialogTitle">GCNET:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.modified.GCNET < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.modified.GCNET < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.modified.GCNET"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.modified.NIAT != undefined"
          >
            <div class="dialogTitle">NIAT:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.modified.NIAT < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.modified.NIAT < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.modified.NIAT"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.modified.EMT_DLFR != undefined"
          >
            <div class="dialogTitle">EMT-DLFR:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.modified.EMT_DLFR < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.modified.EMT_DLFR < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.modified.EMT_DLFR"
            />
          </div>

          <div class="dialogItem" v-if="labelResult.modified != undefined">
            <div class="dialogTitle">Average:</div>
            <el-input
              :style="{
                backgroundColor:
                  labelResult.modified < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : labelResult.modified < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="labelResult.modified"
            />
          </div>
        </div>
        <div
          class="dialogScoresContainer"
          v-if="viewResults.result.defended != undefined"
        >
          <div>Defended</div>
          <div
            class="dialogItem"
            v-if="viewResults.result.defended.T2FN != undefined"
          >
            <div class="dialogTitle">T2FN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.defended.T2FN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.defended.T2FN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.defended.T2FN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.defended.TPFN != undefined"
          >
            <div class="dialogTitle">TPFN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.defended.TPFN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.defended.TPFN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.defended.TPFN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.defended.CTFN != undefined"
          >
            <div class="dialogTitle">CTFN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.defended.CTFN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.defended.CTFN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.defended.CTFN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.defended.MMIN != undefined"
          >
            <div class="dialogTitle">MMIN:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.defended.MMIN < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.defended.MMIN < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.defended.MMIN"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.defended.TFRNet != undefined"
          >
            <div class="dialogTitle">TFR-Net:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.defended.TFRNet < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.defended.TFRNet < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.defended.TFRNet"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.defended.GCNET != undefined"
          >
            <div class="dialogTitle">GCNET:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.defended.GCNET < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.defended.GCNET < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.defended.GCNET"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.defended.NIAT != undefined"
          >
            <div class="dialogTitle">NIAT:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.defended.NIAT < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.defended.NIAT < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.defended.NIAT"
            />
          </div>
          <div
            class="dialogItem"
            v-if="viewResults.result.defended.EMT_DLFR != undefined"
          >
            <div class="dialogTitle">EMT-DLFR:</div>
            <el-input
              :style="{
                backgroundColor:
                  viewResults.result.defended.EMT_DLFR < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : viewResults.result.defended.EMT_DLFR < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="viewResults.result.defended.EMT_DLFR"
            />
          </div>
          <div class="dialogItem" v-if="labelResult.defended != undefined">
            <div class="dialogTitle">Average:</div>
            <el-input
              :style="{
                backgroundColor:
                  labelResult.defended < -0.5
                    ? 'rgba(241, 76, 76, 0.1)'
                    : labelResult.defended < 0.5
                    ? 'rgba(172, 175, 2, 0.1)'
                    : 'rgba(60, 190, 157, 0.1)',
              }"
              readonly
              v-model="labelResult.defended"
            />
          </div>
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button type="primary" @click="dialogScores = false"
            >Confirm</el-button
          >
        </span>
      </template>
    </el-dialog>
  </div>
</template>
  <script setup>
import { ref, computed, onMounted } from "vue";
import OperationMethod from "@/utils/operation.js";
import TableModified from "@/components/TableModified.vue";
import * as echarts from "echarts";
const modifiedWave = ref(null);
const originalWaveform = ref(null);
const originalWavesurfer = ref(null);
const modifiedWavesurfer = ref(null);
const originalPlayer = ref(null);
const modifiedPlayer = ref(null);
const sliderValue = ref(0);
const mediaPaused = ref(true);
const props = defineProps({
  textList: Array,
  duration: Number,
  videoUrl: String,
  originalVideoUrl: String,
  editAligned: Array,
  methodData: Object,
  viewResults: Object,
  defenceVideoUrl: String,
  asrNosie:Boolean
});
const asrNosie = computed(() => {
  return props.asrNosie;
});
const textList = computed(() => {
  return props.textList;
});
const duration = computed(() => {
  return props.duration;
});
const videoUrl = computed(() => {
  return props.videoUrl;
});
const regions = computed(() => {
  let dataList = [];
  textList.value.forEach((item) => {
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
const labelResult = computed(() => {
  var labelResult = {
    original: 0,
    modified: 0,
    defended: 0,
  };
  var account = 0;
  for (let item in props.viewResults.result.original) {
    labelResult.original += props.viewResults.result.original[item];
    labelResult.modified += props.viewResults.result.modified[item];
    if (props.viewResults.result.defended != undefined) {
      labelResult.defended += props.viewResults.result.defended[item];
    }
    account++;
  }
  labelResult.original = Number((labelResult.original / account).toFixed(3));
  labelResult.modified = Number((labelResult.modified / account).toFixed(3));
  labelResult.defended = Number((labelResult.defended / account).toFixed(3));
  return labelResult;
});
const viewResults = computed(() => {
  let middleResult = props.viewResults;
  for (let item in middleResult.result.original) {
    middleResult.result.original[item] = Number(
      middleResult.result.original[item].toFixed(3)
    );
    middleResult.result.modified[item] = Number(
      middleResult.result.modified[item].toFixed(3)
    );
    if (middleResult.result.defended != undefined) {
      middleResult.result.defended[item] = Number(
        middleResult.result.defended[item].toFixed(3)
      );
    }
  }
  return middleResult;
});
const editAligned = computed(() => {
  return props.editAligned;
});
const originalVideoUrl = computed(() => {
  return props.originalVideoUrl;
});
const methodList = ref([
  {
    value: "audio",
    label: "Audio",
  },
  {
    value: "video",
    label: "Video",
  },
]);
const methodValue = ref("audio");
const methodDetailList = computed(() => {
  var detailList = [];
  for (let item in props.viewResults.feature.original) {
    if (item.indexOf("AU") == -1 && methodValue.value == "audio") {
      detailList.push({
        value: item,
        label: item,
      });
    } else if (item.indexOf("AU") != -1 && methodValue.value == "video") {
      detailList.push({
        value: item,
        label: item,
      });
    }
  }
  return detailList;
});
const methodDetailValue = ref(methodDetailList.value[0].value);
const boxItemTip = computed(() => {
  return OperationMethod.echartsDataChange(methodDetailValue.value);
});
const modifiedWaveSwitch = ref(true);

const chooseClick = (t) => {
  if (t == "one") {
    methodDetailValue.value = methodDetailList.value[0].value;
  }
  var originalSeriesDate = [];
  var modifiedSeriesDate = [];
  var defendedSeriesDate = [];
  for (let i = 0; i < textList.value.length; i++) {
    originalSeriesDate.push(
      viewResults.value.feature.original[methodDetailValue.value][i]
    );
    modifiedSeriesDate.push(
      viewResults.value.feature.modified[methodDetailValue.value][i]
    );
    if (props.viewResults.feature.defended != undefined) {
      defendedSeriesDate.push(
        viewResults.value.feature.defended[methodDetailValue.value][i]
      );
    }
  }
  option.series[0].data = originalSeriesDate;
  option.series[1].data = modifiedSeriesDate;
  if (props.viewResults.feature.defended != undefined) {
    option.series[2].data = defendedSeriesDate;
  }

  myChart.setOption(option, false, false);
};
const originalEvent = () => {
  originalWavesurfer.value = OperationMethod.waveformCreate(
    "#originalWaveform",
    regions.value,
    "#originalWaveTimeline"
  );
  originalWavesurfer.value.load(originalVideoUrl.value);
  originalWavesurfer.value.on("ready", () => {});
};
const modifiedEvent = (url) => {
  modifiedWavesurfer.value = OperationMethod.waveformCreate(
    "#modifiedWave",
    regions.value,
    "#modifiedTimeline"
  );
  modifiedWavesurfer.value.load(url);
  modifiedWavesurfer.value.on("ready", () => {});
};
var myChart = null;
var option = null;
const chartCreate = () => {
  var chartDom = document.getElementById("main");
  myChart = echarts.init(chartDom);
  var xAxisData = [];
  var originalSeriesDate = [];
  var modifiedSeriesDate = [];
  var defenceSeriesDate = [];
  for (let i = 0; i < textList.value.length; i++) {
    xAxisData.push({ value: textList.value[i].text, id: i });
    originalSeriesDate.push(
      viewResults.value.feature.original[methodDetailValue.value][i]
    );
    modifiedSeriesDate.push(
      viewResults.value.feature.modified[methodDetailValue.value][i]
    );
    if (props.viewResults.feature.defended != undefined) {
      defenceSeriesDate.push(
        viewResults.value.feature.defended[methodDetailValue.value][i]
      );
    }
  }
  var optionData = ["Original", "Modified"];
  var optionSeries = [
    {
      data: originalSeriesDate,
      type: "line",
      name: "Original",
      color: ["#e69d87"],
      animation: false,
      markLine: {
        symbol: "none",
        label: {
          position: "start",
          textStyle: {
            fontSize: 16,
          },
          formatter: function (params) {
            return textList.value[params.value].text;
          },
        },
        data: [
          {
            silent: false,
            lineStyle: {
              type: "line",
              color: "#505050",
            },

            xAxis: 0,
          },
        ],
      },
    },
    {
      data: modifiedSeriesDate,
      type: "line",
      name: "Modified",
      color: ["#8dc1a9"],
    },
  ];
  if (props.viewResults.feature.defended != undefined) {
    optionData.push("Defended");
    optionSeries.push({
      data: defenceSeriesDate,
      type: "line",
      name: "Defended",
      color: ["#51abff"],
    });
  }
  option = {
    legend: {
      icon: "roundRect",
      orient: "horizontal",
      itemHeight: 2,
      data: optionData,
      right: 50,
      top: 30,
    },
    xAxis: {
      type: "category",
      show: false,
      data: xAxisData,
      axisLabel: { interval: "auto" },
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#000000",
      textStyle: {
        color: "#ffffff",
      },
      formatter: function (params) {
        return params[0].name;
      },
    },
    yAxis: {
      type: "value",
    },
    series: optionSeries,
  };
  option && myChart.setOption(option, false, false);
};
const modifiedVideoTime = ref(0);
const originalVideoTime = ref(0);
const dialogScores = ref(false);
const originalVideoTimeEvent = (that) => {
  let count = 10;
  let timer = setInterval(() => {
    let nowTime = that.currentTime();
    if (nowTime >= duration.value) {
      clearInterval(timer);
      return;
    }
    originalVideoTime.value = nowTime;
    originalWavesurfer.value.seekAndCenter(
      originalVideoTime.value / duration.value > 1
        ? 1
        : originalVideoTime.value / duration.value
    );
    if (
      originalVideoTime.value > textList.value[textList.value.length - 1].end
    ) {
      $(`#originalWaveform > wave > region`).css({
        background: "rgba(181, 198, 241, 0.2)",
      });
    } else {
      let watchSwitch = true;
      for (let index = 0; index < textList.value.length; index++) {
        if (
          textList.value[index].start < originalVideoTime.value &&
          originalVideoTime.value <= textList.value[index].end
        ) {
          $(`#originalWaveform > wave > region:nth-child(${index + 2})`).css({
            background: "rgba(181, 198, 241, 0.2)",
          });
          $(`#originalWaveform > wave > region:nth-child(${index + 3})`).css({
            background: "rgba(255, 228, 196, 0.5)",
          });
          watchSwitch = false;
          break;
        }
      }
      if (watchSwitch) {
        $(`#originalWaveform > wave > region`).css({
          background: "rgba(181, 198, 241, 0.2)",
        });
      }
    }
    count == 0 ? clearInterval(timer) : count--;
  }, 25);
};
const modifiedVideoTimeEvent = (that) => {
  let count = 10;
  let timer = setInterval(() => {
    let nowTime = that.currentTime();
    if (nowTime >= duration.value) {
      clearInterval(timer);
      watchVideo(nowTime);
      return;
    }
    modifiedVideoTime.value = nowTime;
    modifiedWavesurfer.value.seekAndCenter(
      modifiedVideoTime.value / duration.value > 1
        ? 1
        : modifiedVideoTime.value / duration.value
    );
    watchVideo(modifiedVideoTime.value);
    echartTimeOffset(modifiedVideoTime.value);
    sliderValue.value = (modifiedVideoTime.value / duration.value) * 100;
    count == 0 ? clearInterval(timer) : count--;
  }, 25);
};
var lastword = "";
const echartTimeOffset = (modifiedVideoTime) => {
  for (let i = props.textList.length - 1; i >= 0; i--) {
    if (
      modifiedVideoTime >= props.textList[i].start &&
      lastword != props.textList[i].text &&
      modifiedVideoTime <= props.textList[i].end
    ) {
      lastword = props.textList[i].text;
      option.series[0].markLine.data[0].xAxis = i;
      myChart.setOption(
        {
          series: option.series,
        },
        false,
        false
      );
      break;
    }
  }
};
const hightLightInit = () => {
  $(`#modifiedWave > wave > region`).css({
    background: "rgba(181, 198, 241, 0.2)",
  });
  $(`.pageContainer .viewResultContainer .textContent span`).css({
    background: "#fff",
  });
};
const highlightOver = (index, background, textBackground) => {
  $(`#modifiedWave > wave > region:nth-child(${index + 3})`).css({
    background,
  });
  $(
    `.pageContainer .viewResultContainer .textContent span:nth-child(${
      index + 1
    })`
  ).css({
    background: textBackground,
  });
};
const watchVideo = (videoTime) => {
  if (videoTime > textList.value[textList.value.length - 1].end) {
    hightLightInit();
    return;
  }
  let watchSwitch = true;
  for (let index = 0; index < textList.value.length; index++) {
    if (
      textList.value[index].start < videoTime &&
      videoTime <= textList.value[index].end
    ) {
      highlightOver(index - 1, "rgba(181, 198, 241, 0.2)", "#fff");
      highlightOver(
        index,
        "rgba(255, 228, 196, 0.5)",
        "rgba(255, 228, 196, 0.5)"
      );
      watchSwitch = false;
      break;
    }
  }
  if (watchSwitch) {
    hightLightInit();
  }
};
var originalLocked = true;
var modifiedLocked = true;
const originalVideo = () => {
  let player = videojs.getPlayer("viewOriginal");
  if (player) {
    player.dispose();
  }
  originalPlayer.value = videojs("viewOriginal", {}, function onPlayerReady() {
    this.on("play", function (e) {
      modifiedPlayer.value.play();
      modifiedVideoTimeEvent(this);
      originalVideoTimeEvent(this);
      this.volume(0);
      modifiedPlayer.value.volume(1);
      mediaPaused.value = false;
    });
    this.on("seeked", function () {
      originalLocked == false;
      if (modifiedLocked) {
        let nowTime = this.currentTime();
        modifiedPlayer.value.currentTime(nowTime);
      }
      modifiedLocked = true;
      $(`#originalWaveform > wave > region`).css({
        background: "rgba(181, 198, 241, 0.2)",
      });
      hightLightInit();
    });
    this.on("pause", function () {
      mediaPaused.value = true;
      modifiedPlayer.value.pause();
    });
  });
};
const modifiedVideo = () => {
  let player = videojs.getPlayer("viewModified");
  if (player) {
    player.dispose();
  }
  modifiedPlayer.value = videojs("viewModified", {}, function onPlayerReady() {
    this.on("play", function (e) {
      originalPlayer.value.play();
      modifiedVideoTimeEvent(this);
      originalVideoTimeEvent(this);
      mediaPaused.value = false;
      this.volume(0);
      originalPlayer.value.volume(1);
    });
    this.on("timeupdate", function (e) {
      originalVideoTimeEvent(this);
      modifiedVideoTimeEvent(this);
    });
    this.on("seeked", function () {
      modifiedLocked = false;
      if (originalLocked) {
        let nowTime = this.currentTime();
        originalPlayer.value.currentTime(nowTime);
      }
      originalLocked == true;
      $(`#originalWaveform > wave > region`).css({
        background: "rgba(181, 198, 241, 0.2)",
      });
      hightLightInit();
    });
    this.on("pause", function () {
      mediaPaused.value = true;
      originalPlayer.value.pause();
    });
  });
};

const modifiedDefenceList = ref([
  { label: "Modified", value: "Modified" },
  { label: "Defended", value: "Defended" },
]);

const modifiedDefence = ref("Modified");
const defenceChange = () => {
  modifiedPlayer.value.currentTime(0);
  originalPlayer.value.currentTime(0);
  option.series[0].markLine.data[0].xAxis = 0;
  myChart.setOption(option, false, false);
  modifiedPlayer.value.dispose();
  var url = videoUrl.value;
  if (modifiedDefence.value != "Modified") {
    url = props.defenceVideoUrl;
  }
  let html =
    '<video id="viewModified" class="video-js vjs-big-play-centered" controls data-setup="{}"><source src="' +
    url +
    '" type="video/mp4" /></video>';
  document.getElementById("viewModifiedContainer").innerHTML = html;
  modifiedVideo();
  modifiedWaveSwitch.value = false;
  setTimeout(() => {
    modifiedWaveSwitch.value = true;
    setTimeout(() => {
      modifiedEvent(url);
    }, 2);
  }, 2);
};
const originalSeekToTime = (textitem) => {
  $(`#originalWaveform > wave > region`).css({
    background: "rgba(181, 198, 241, 0.2)",
  });
  if (textitem == null) {
    setTimeout(() => {
      originalPlayer.value.currentTime(
        originalWavesurfer.value.getCurrentTime() + 0.000001
      );
    }, 5);
  } else {
    originalPlayer.value.currentTime(textitem + 0.000001);
  }
};
const modifiedSeekToTime = (textitem) => {
  hightLightInit();
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
onMounted(() => {
  originalEvent();
  modifiedEvent(videoUrl.value);
  chartCreate();
  originalVideo();
  modifiedVideo();
});

const playVideo = (type) => {
  if (type == "play") {
    if (sliderValue.value != 100) {
      mediaPaused.value
        ? modifiedPlayer.value.play()
        : modifiedPlayer.value.pause();
      mediaPaused.value = !mediaPaused.value;
    }
  } else if (type == "start") {
    modifiedPlayer.value.currentTime(0);
    option.series[0].markLine.data[0].xAxis = 0;
    myChart.setOption(
      {
        series: option.series,
      },
      false,
      false
    );
  } else if (type == "end") {
    modifiedPlayer.value.currentTime(duration.value - 0.000001);
    option.series[0].markLine.data[0].xAxis = props.textList.length - 1;
    myChart.setOption(
      {
        series: option.series,
      },
      false,
      false
    );
  } else if (type == "last") {
    if (
      modifiedVideoTime.value > textList.value[textList.value.length - 1].end
    ) {
      modifiedPlayer.value.currentTime(
        textList.value[textList.value.length - 1].start + 0.000001
      );
    } else if (
      modifiedVideoTime.value > textList.value[textList.value.length - 1].start
    ) {
      modifiedPlayer.value.currentTime(
        textList.value[textList.value.length - 2].start + 0.000001
      );
    } else if (
      modifiedVideoTime.value < textList.value[textList.value.length - 1].start
    ) {
      let videoTime = 0;
      for (let index = 0; index < textList.value.length; index++) {
        if (
          textList.value[index].start < modifiedVideoTime.value &&
          modifiedVideoTime.value <= textList.value[index + 1].start
        ) {
          if (index) {
            videoTime = textList.value[index - 1].start + 0.000001;
          }
          break;
        }
      }
      modifiedPlayer.value.currentTime(videoTime);
    }
  } else if (type == "next") {
    if (modifiedVideoTime.value < textList.value[0].start) {
      modifiedPlayer.value.currentTime(textList.value[0].start + 0.000001);
    } else if (
      modifiedVideoTime.value < textList.value[textList.value.length - 1].start
    ) {
      let videoTime = 0;
      for (let index = 0; index < textList.value.length; index++) {
        if (
          textList.value[index].start < modifiedVideoTime.value &&
          modifiedVideoTime.value <= textList.value[index + 1].start
        ) {
          videoTime = textList.value[index + 1].start + 0.000001;
          break;
        }
      }
      modifiedPlayer.value.currentTime(videoTime);
    }
  }
};
const sliderChange = (e) => {
  modifiedPlayer.value.currentTime(
    (sliderValue.value / 100) * duration.value + 0.000001
  );
};
const formatTooltip = (e) => {
  return Math.floor((e / 100) * duration.value * 100) / 100 + "S";
};
</script>
  <style lang="scss" scope>
.viewResultContainer {
  display: flex;
  flex-direction: column;
  margin-top: 2px;
}
.topContainer {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
}
.video-js {
  width: 320px;
  height: 238px;
}
.labelContainer {
  font-size: 15px;
  display: flex;
  flex-direction: row;
  align-items: center;
}
.labelTitle {
  display: inline-block;
  width: 80px;
}
.viewOriginal {
  margin-top: 10px;
}
.labelContent:hover {
  cursor: pointer;
}
.viewModified {
  margin-top: 5px;
}
.featureContainer {
  display: flex;
  flex-direction: column;
  width: 500px;
}
.selectContainer {
  display: flex;
  flex-direction: row;
  align-items: center;
  margin-left: 15px;
  z-index: 99;
}
.selectTitle {
  width: 180px;
  font-size: 15px;
}
.waveform {
  height: 75px;
  box-shadow: 0 0 1px 1px rgba(181, 198, 241, 0.2);
  margin-top: 7px;
  margin-left: 15px;
  width: 465px;
}
#modifiedWave,
#originalWaveform {
  height: 60px !important;
}
#modifiedWave > wave,
#originalWaveform > wave {
  height: 60px !important;
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
#originalWaveTimeline > timeline,
#originalWaveTimeline > timeline > canvas {
  height: 17px !important;
}
.bottomContainer {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  .textContent {
    margin-top: 5px;
    padding-bottom: 10px;
    font-size: 18px;
    line-height: 25px;
    span {
      display: inline-block;
      margin-top: 5px;
      margin-right: 10px;
      user-select: none;
      &:hover {
        background-color: rgba(255, 228, 196, 0.5);
        cursor: pointer;
      }
    }
  }
  .audioContainer {
    width: 500px;
    display: flex;
    flex-direction: column;
    align-items: center;
    .control {
      display: flex;
      flex-direction: row;
      justify-content: space-around;
      margin-top: 5px;
      width: 400px;
      .icon {
        height: 23px;
        width: 23px;
      }
    }
  }
}
.dialogScoresContainer {
  margin-top: -20px;
  width: 180px;
  .dialogItem {
    display: flex;
    flex-direction: row;
    align-items: center;
    margin-top: 10px;
    .dialogTitle {
      width: 155px;
    }
  }
}
.el-dialog__body {
  height: auto !important;
}
.dialogContainer {
  display: flex;
  flex-direction: row;
  justify-content: space-around;
}
.el-input__wrapper {
  background-color: transparent;
}
.methodOne {
  width: 80px;
}
.methodTwo {
  margin-left: 10px;
  width: 150px;
}
.questionFilled {
  color: #bbbbbb;
  height: 33px;
  margin-left: 10px;
}
.boxItemTip {
  width: 200px;
}
.modifiedDefence {
  width: 120px;
  margin-right: 20px;
}
.topDescribe {
  font-size: 14px;
  color: #727272;
}
.topDescribe span {
  font-weight: bold;
}
.asrCheckbox {
  margin-top: 10px;
  margin-left: 15px;
}
</style>
  <style>
.viewResultContainer .el-divider--horizontal {
  margin: 15px 0;
}
</style>