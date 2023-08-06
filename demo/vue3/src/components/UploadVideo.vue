<template>
  <div class="uploadVideoContainer">
    <div class="tipNote">
      Note: Uploaded file should be no more than 10MB. Currently only support
      English.
    </div>
    <el-upload
      class="upload-demo"
      ref="upload"
      action="#"
      :auto-upload="false"
      :limit="1"
      drag
      :on-change="uploadMp4"
      accept=".MP4, .AVI, .MOV, .MKV, .3GP, .M4V, .FLV, .MPG, .mp4, .avi, .mov, .mkv, .3gp, .m4v, .flv, .mpg"
      :show-file-list="false"
    >
      <div v-if="typeof currentFile == 'undefined' || currentFile == ''">
        <el-icon class="el-icon--upload">
          <upload-filled />
        </el-icon>
        <div class="el-upload__text">
          Drop file here or
          <em>click to upload</em>
        </div>
      </div>
      <div v-else class="successUpload">
        <el-icon class="el-icon--upload">
          <upload-filled />
        </el-icon>
        <div class="el-upload__text">
          Drop file here or
          <em>click to upload</em>
        </div>
      </div>
      <template #tip>
        <div
          class="el-upload__tip"
          v-if="!(typeof currentFile == 'undefined' || currentFile == '')"
        >
          <el-icon>
            <VideoCameraFilled />
          </el-icon>
          <span>Filename: {{ currentFile.name }}</span>
        </div>
      </template>
    </el-upload>
  </div>
</template>
    
  <script setup>
import { ref } from "vue";
const currentFile = ref("");
const upload = ref("");
const emit = defineEmits(["transmitMp4"]);
const uploadMp4 = async (file, fileList) => {
  if (file.status === "ready") {
    const is10M = file.size / 1024 / 1024 < 10;
    if (is10M) {
      currentFile.value = file.raw;
      emit("transmitMp4", currentFile.value);
    } else {
      ElMessage({
        message: "Uploaded file should be no more than 10MB.",
        type: "warning",
      });
    }
    upload.value.clearFiles();
  }
};
</script>
    
  <style scoped>
.tipNote {
  font-size: 14px;
  color: #5f5f5f;
  margin-top: 50px;
}
.upload-demo {
  margin-top: 15px;
  height: 260px;
}
</style>