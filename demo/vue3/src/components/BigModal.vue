<template>
  <div class="bigModalContainer">
    <div class="topContainer">
      <div class="videoContainer">
        <div class="topDescribe">
          Original video and results presentation.
        </div>
        <div class="viewOriginal">
          <video
            class="video-js vjs-big-play-centered"
            controls
            data-setup="{}"
          >
            <source :src="originalVideoUrl" type="video/mp4" />
          </video>
        </div>
        <div class="topDescribe" style="margin-top:20px">
          Modified video and results display.
        </div>
        <div class="viewModified" id="viewModifiedContainer">
          <video
            class="video-js vjs-big-play-centered"
            controls
            data-setup="{}"
          >
            <source :src="videoUrl" type="video/mp4" />
          </video>
        </div>
      </div>
      <div class="featureContainer">
        <el-image
          style="width: 500px; height: 270px;margin-top:25px"
          src="samples/video_gpt/big_modal_raw.png"
          fit="contain"
        />
        <el-image
          style="width: 500px; height: 270px;margin-top:30px"
          src="samples/video_gpt/big_modal_noise.png"
          fit="contain"
        />
      </div>
    </div>
  </div>
</template>
  <script setup>
import { ref } from "vue";
const originalVideoUrl = ref("samples/video_gpt/raw_video.mp4");
const videoUrl = ref("samples/video_gpt/modified_video.mp4");
</script>
  <style lang="scss" scope>
.bigModalContainer {
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
  width: 340px;
  height: 250px;
  object-fit: cover;
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
.featureContainer {
  display: flex;
  flex-direction: column;
  width: 500px;
}
.topDescribe {
  font-size: 14px;
  color: #727272;
  margin-top: 5px;
}
.topDescribe span {
  font-weight: bold;
}
</style>