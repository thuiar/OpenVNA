import request from "@/utils/request"

export function getTest() {
    return request.get({
        url: 'test'
    })
}


export function uploadVideo(data) {
    return request.getBlob({
        url: 'uploadVideo',
        data:data
    })
}

export function uploadTranscript(data) {
    return request.post({
        url: 'uploadTranscript',
        data:data
    })
}


