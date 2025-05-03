from fastapi import Response, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import threading
from ultralytics import YOLO
import torch
from utils.window import Window
from utils.reminder import reminder_bytes
from conf import window_size
from nets.net import Pnet
from conf import window_size, hidden_size, num_layers

stream_router = APIRouter()
streaming = False
cap_ip = None
device = torch.device("cuda:0")
lock = threading.Lock()
detector = YOLO('../../models/yolo11n-pose.pt')
pnet = Pnet(window_size, hidden_size, num_layers).to(device)


def get_net(net_name):
    if net_name == 'pnet':
        return pnet
    return None


def get_cap(ip=None):
    if ip is None:
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cv2.VideoCapture(ip)

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    res = detector(frame)[0]
    kp = res.keypoints.xyn
    if len(res.boxes.cls) > 1:
        # 如果监测出两个人及以上，取置信度最大的
        idx = res.boxes.conf.argmax(-1).item()
        kp = res.keypoints.xy[idx].unsqueeze(0)
    elif len(res.boxes.cls) == 0:
        kp = None
    return res.plot(), kp


def put_text(frame, occur):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)
    font_scale = 1
    if occur:
        color = (0, 0, 255)
        text = 'OCCURRING'
    else:
        color = (0, 255, 0)
        text = 'FINE'
    thickness = 2
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def gen_frames():
    occur = False  # 是否出现代偿行为
    window = Window(device, window_size, (17, 2))
    cap = get_cap(cap_ip)
    while True:
        with lock:
            if not streaming:
                cap.release()
                break

        success, frame = cap.read()
        if not success:
            break

        frame, kp = process_frame(frame)  # kp即关键点，形状[num_people, 17, 2]
        if kp is None:
            frame = reminder_bytes
        else:
            ready = window.add(kp)
            if ready:
                # 做后继模型的预测
                # print(window.data.shape)
                pred = get_net("pnet")(window.data.reshape(window_size, -1).unsqueeze(1))
                if pred.argmax(-1) == 0:
                    occur = True
                else:
                    occur = False
                window.clear()

            frame = put_text(frame, occur)
            success, frame = cv2.imencode('.jpg', frame)
            if not success:
                break
            frame = frame.tobytes()

        yield (b"--banana\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@stream_router.get("/video_feed")
def video_feed():
    with lock:
        if not streaming:
            return Response(content="Stream is closed", status_code=403)
        return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=banana")


@stream_router.post("/func/start_stream")
def start_stream():
    global streaming
    with lock:
        streaming = True
    return JSONResponse(content={"status": 1, "content": "stream started"})


@stream_router.post("/func/stop_stream")
def stop_stream():
    global streaming
    with lock:
        streaming = False
    return JSONResponse(content={"status": 1, "content": "stream stopped"})

@stream_router.post("/status")
def get_status():
    def get_cap_name():
        if cap_ip is None:
            return "默认摄像头"
        return f"远程摄像头{cap_ip}"
    return JSONResponse(content={
        "status": 1, 
        "cap": get_cap_name()
    })