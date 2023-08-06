import { createRouter, createWebHashHistory } from "vue-router";
import VideoEditor from "../views/VideoEditor.vue";
import Introduce from "../views/Introduce.vue"

const routes = [
  {
    path: "/viedoeditor",
    name: "viedoeditor",
    component: VideoEditor
  },
  {
    path: "/",
    name: "introduce",
    component: Introduce
  }
];

const router = createRouter({
  history: createWebHashHistory(),
  routes
});

export default router;
