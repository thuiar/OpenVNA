import axios from "axios";
import { ElMessage } from "element-plus";
var Axios = axios.create();
// Axios.defaults.baseURL = window.base_url
Axios.interceptors.request.use(
  (config) => {
    // token
    return config;
  },
  (err) => {
    Promise.reject(err);
  }
);
Axios.interceptors.response.use(
  (config) => {
    return config;
  },
  (err) => {
    Promise.reject(err);
  }
);
// get
export function get({ url, params = {} }) {
  return new Promise((resolve, reject) => {
    Axios({
      url: window.base_url + url,
      params,
      method: "get",
    })
      .then((res) => {
        if (res.data.code == 400) {
          ElMessage.error(res.data.msg);
          reject(false);
        } else {
          resolve(res.data);
        }
      })
      .catch((err) => {
        ElMessage.error('Server failed to return data');
        reject(err);
      });
  });
}
// post
export function post({ url, data = {} }) {
  return new Promise((resolve, reject) => {
    axios({
      url: window.base_url + url,
      // params,
      data,
      method: "post",
    })
      .then((res) => {
        if (res.data.code == 400) {
          ElMessage.error(res.data.msg);
          reject(false);
        } else {
          resolve(res);
        }
      })
      .catch((err) => {
        ElMessage.error('Server failed to return data');
        reject(err);
      });
  });
}
// delete
export function del(url, params = {}, data = {}) {
  return new Promise((resolve, reject) => {
    axios({
      url: window.base_url + url,
      method: "delete",
      params,
      data,
    })
      .then((res) => {
        resolve(res);
      })
      .catch((err) => {
        reject(err);
      });
  });
}
//   Blob
export function getBlob({url, data}) {
  return new Promise((resolve, reject) => {
    axios
      .post(window.base_url + url, data, {
        "Content-type": "multipart/form-data"
      })
      .then(res => {
        if (res.data.code == 400) {
          ElMessage.error(res.data.msg);
          reject(false);
        } else {
          resolve(res);
        }
      }).catch((err) => {
        ElMessage.error('Server failed to return data');
        reject(err);
      });
  });
}

export default {
  get,
  post,
  del,
  getBlob
}