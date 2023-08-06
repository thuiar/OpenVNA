<template>
  <div class="stepProcessContainer">
    <div class="stepTop">
      <div class="arrayTitle" :style="{ width: arrayTitle + 'px' }">
        Robust-MSA
      </div>
      <el-steps :active="activeStep" class="elSteps" simple>
        <el-step
          title="Upload Video"
          :icon="activeStep > 0 ? CircleCheckFilled : UploadFilled"
        />
        <el-step
          title="Revise Transcript"
          :icon="activeStep > 1 ? CircleCheckFilled : List"
        />
        <el-step
          title="Modify Video"
          :icon="activeStep > 2 ? CircleCheckFilled : EditPen"
        />
        <el-step
          title="Select Methods"
          :icon="activeStep > 3 ? CircleCheckFilled : Operation"
        />
        <el-step
          title="View Results"
          :icon="activeStep > 4 ? CircleCheckFilled : Histogram"
        />
      </el-steps>
    </div>
  </div>
</template>
    
  <script setup>
import {
  EditPen,
  List,
  UploadFilled,
  Operation,
  Histogram,
  CircleCheckFilled,
  Flag,
} from "@element-plus/icons-vue";
import { ref, computed, onMounted } from "vue";
const props = defineProps({
    activeStep: Number
});
const activeStep = computed(() => {
  return props.activeStep;
});
const arrayTitle = ref("");
onMounted(() => {
  arrayTitle.value = $(".elSteps").width() + 5;
});
</script>
    
  <style scoped>
.stepTop {
  position: absolute;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0;
  left: 0;
  width: 100%;
  background-color: #fff !important;
  z-index: 99;
}
.elSteps {
  background-color: #fff;
}
.arrayTitle {
  position: relative;
  padding-top: 10px;
  font-size: 21px;
  color: #464646;
}
</style>