import request from "@/utils/request"
export function callASR(data) {
    return request.post({
        url: 'callASR',
        data
    })
}
export function runMSAAligned(data) {
    return request.post({
        url: 'runMSAAligned',
        data
    })
}


