<template>
  <div class="stepProcessContainer">
    <div class="stepTop">
      <div class="arrayTitle" :style="{ width: arrayTitle + 'px' }">
        OpenVNA
      </div>
      <el-steps :active="activeStep" class="elSteps" simple>
        <el-step
          title="Upload Video"
          @click="nowActivate(0)"
          :icon="activeStep > 0 ? CircleCheckFilled : UploadFilled"
        />
        <el-step
          title="Revise Transcript"
          @click="nowActivate(1)"
          :icon="activeStep > 1 ? CircleCheckFilled : List"
        />
        <el-step
          title="Modify Video"
          @click="nowActivate(2)"
          :icon="activeStep > 2 ? CircleCheckFilled : EditPen"
        />
        <el-step
          title="Select Methods"
          @click="nowActivate(3)"
          :icon="activeStep > 3 ? CircleCheckFilled : Operation"
        />
        <el-step
          title="View Results"
          @click="nowActivate(4)"
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
const emit = defineEmits(["trainsActivate"]);
const nowActivate = (e)=>{
  emit("trainsActivate", e);
}
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