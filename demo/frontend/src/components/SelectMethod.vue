<template>
  <div>
    <div class="selectMethod" v-if="!disabled">
      <div class="methodTitle">Methods</div>
      <div class="description">Please select at least one model!</div>
    </div>
    <div class="checkboxContainer">
      <div class="checkboxTitle">Data Level Defence:</div>
      <el-checkbox-group class="checkboxGroup" v-model="dataDefense" :disabled="disabled" @change="selectMethod">
        <el-checkbox label="Audio Denoising" />
        <el-checkbox label="Video MCI (time-consuming)" />
      </el-checkbox-group>
    </div>
    <div class="checkboxContainer">
      <div class="checkboxTitle">Feature Level Defence:</div>
      <el-checkbox-group class="checkboxGroup" v-model="featureDefense" :disabled="disabled" @change="selectMethod">
        <el-checkbox label="Feature Interpolation" />
      </el-checkbox-group>
    </div>
    <div class="checkboxContainer">
      <div class="checkboxTitle">MSA Models:</div>
      <el-checkbox-group class="checkboxGroup" v-model="msaModel" :disabled="disabled" @change="selectMethod">
        <el-checkbox label="MMIN" />
        <el-checkbox label="TFR-Net" />
        <el-checkbox label="T2FN" />
        <el-checkbox label="TPFN" />
      </el-checkbox-group>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
const dataDefense = ref(['Video MCI (time-consuming)','Audio Denoising']);
const featureDefense = ref([]);
const msaModel = ref(["MMIN","TFR-Net","T2FN","TPFN"]);
const props = defineProps({
  disabled: Boolean
});
const emit = defineEmits(["transmitMethods"]);
const selectMethod = () => {
  let msaList = [];
  msaModel.value.forEach(item => {
    if (item == "MMIN") {
      msaList.push("MMIN");
    } else if (item == "TFR-Net") {
      msaList.push("TFR_Net");
    } else if (item == "T2FN") {
      msaList.push("T2FN");
    } else if (item == "TPFN") {
      msaList.push("TPFN");
    }
  });
  let defenseList = [];
  featureDefense.value.forEach(item => {
    if (item == "Feature Interpolation") {
      defenseList.push("f_interpol");
    }
  });
  dataDefense.value.forEach(item => {
    if (item == "Audio Denoising") {
      defenseList.push("a_denoise");
    } else if (item == "Video MCI (time-consuming)") {
      defenseList.push("v_reconstruct");
    }
  });
  let methods = { defence: defenseList, models: msaList };
  emit("transmitMethods", methods);
};
</script>
<style scoped>
.selectContainer {
  display: flex;
  flex-direction: row;
  align-items: center;
}
.selectTitle {
  width: 100px;
}
.selectMethod {
  display: flex;
  flex-direction: column;
  padding-bottom: 20px;
}
.methodTitle {
  font-size: 18px;
  color: #2c2c2c;
  line-height: 35px;
}
.description {
  font-size: 15px;
  color: #7a7a7a;
  line-height: 20px;
}
.checkboxContainer {
  display: flex;
  flex-direction: row;
  padding-bottom: 10px;
}
.checkboxTitle {
  width: 195px;
  font-size: 16px;
  color: #2c2c2c;
  line-height: 30px;
}
.checkboxGroup {
  width: 600px;
}
</style>
<style>
.selectMethod .checkboxContainer .el-checkbox__label {
  font-size: 16px;
}
</style>