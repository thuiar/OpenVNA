<template>
  <div class="introduceContainer">
    <el-button type="primary" @click="redirectToPage">Primary</el-button>
  </div>
</template>
    
<script setup>
import { ref } from "vue";

import {  useRouter } from "vue-router";

const router = useRouter();

const redirectToPage = ()=> {
  router.push({ path: "/viedoeditor" });
}
</script>
    
  <style scoped>
</style>