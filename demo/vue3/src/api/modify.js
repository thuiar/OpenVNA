import request from "@/utils/request"

export function videoEditAligned(data) {
    return request.post({
        url: 'videoEditAligned',
        data: data
    })
}

export function getFileFromUrl(url, fileName) {
    return new Promise((resolve, reject) => {
        var xhr = new XMLHttpRequest();
        xhr.open("GET", window.static_url+url, true);
        xhr.responseType = "blob";
        xhr.onload = () => {
            let file = new File([xhr.response], fileName, { type: "video/mp4" });
            resolve(file);
        };
        xhr.onerror = e => {
            resolve(e);
        };
        xhr.send();
    })
}