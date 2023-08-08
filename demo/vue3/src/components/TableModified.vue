<template>
  <div class="tableModifiedContainer">
    <div class="previewTop">
      <!-- <div class="topTip">Noise display</div> -->
      <div class="topDescribe">Show details of the added noise.</div>
    </div>
    <el-table :data="tableData" border stripe class="table_details">
      <el-table-column prop="index" label="Index" width="70" />
      <el-table-column prop="modality" label="Modality" width="90" />
      <el-table-column prop="start" label="Start time" width="95" />
      <el-table-column prop="end" label="End time" width="95" />
      <el-table-column prop="type" label="Noise type" width="160" />
      <el-table-column prop="option" label="Noise details" width="200" />
      <el-table-column fixed="right" v-if="!onlyDisplay" label="Operate">
        <template #default="scope">
          <el-tag
            class="ml-2"
            @click="handleDelete(scope.$index)"
            type="danger"
            >Delete</el-tag
          >
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>
  
<script setup>
import { ref, computed } from "vue";

const props = defineProps({
  editAligned: Array,
  onlyDisplay:Boolean,
});
const tableData = computed(() => {
  let tempData = [];
  let index = 1;
  for (let item of props.editAligned) {
    let modality = "";
    if (item[1] == "t") {
      modality = "text";
      item[3] = item[3]?`'${item[3][0]}' to '${item[3][1]}'`:''
    } else if (item[1] == "a") {
      modality = "audio";
    } else if (item[1] == "v") {
      modality = "video";
    }
    tempData.push({
      index: index,
      modality: modality,
      start: item[0][0],
      end: item[0][1],
      type: item[2],
      option: item[3],
    });
    index++;
  }
  return tempData;
});
const emit = defineEmits(["deleteNoise"]);
const handleDelete = (e)=>{
    emit("deleteNoise", e);
}
</script>
  
<style scoped>
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
.table_details {
  margin-top: 10px;
}
.table_details .ml-2 {
  cursor: pointer;
}
</style>

<style>
.tableModifiedContainer .el-table .cell {
  line-height: 17px;
  font-size: 13px;
}
.tableModifiedContainer
  .el-table__inner-wrapper
  .el-table__body-wrapper
  .el-scrollbar__wrap.el-scrollbar__wrap--hidden-default
  table
  > tbody
  > tr
  .cell {
  font-size: 14px;
}
</style>