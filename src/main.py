from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import threading
from ultralytics import YOLO
import torch

import logging
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.WARNING)

from utils.window import Window
from utils.reminder import reminder_bytes

app = FastAPI()
streaming = True
lock = threading.Lock()

model = YOLO('../models/yolo11n-pose.pt')
window_size = 24

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    res = model(frame)[0]
    kp = res.keypoints.xy
    if len(res.boxes.cls) > 1:
        # 如果监测出两个人及以上，取置信度最大的
        idx = res.boxes.conf.argmax(-1).item()
        kp = res.keypoints.xy[idx].unsqueeze(0)
    elif len(res.boxes.cls) == 0:
        kp = None
    return res.plot(), kp

def gen_frames():
    window = Window(torch.device("cuda:0"), window_size, (17, 2))
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        with lock:
            if not streaming:
                cap.release()
                break
        
        success, frame = cap.read()
        if not success:
            break
        
        frame, kp = process_frame(frame) # kp即关键点，形状[num_people, 17, 2]
        if kp is None:
            frame = reminder_bytes
        else:
            ready = window.add(kp)
            if ready:
                # 做后继模型的预测
                print(window.data.shape)
                window.clear()
            success, frame = cv2.imencode('.jpg', frame)
            if not success:
                break
            frame = frame.tobytes()

        yield (b"--banana\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        
@app.get("/video_feed")
def video_feed():
    with lock:
        if not streaming:
            return Response(content="Stream is closed", status_code=403)
        return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=banana")
    
@app.post("/func/start_stream")
def start_stream():
    global streaming
    with lock:
        streaming = True
    return JSONResponse(content={"status": 1, "content": "stream started"})

@app.post("/func/stop_stream")
def stop_stream():
    global streaming
    with lock:
        streaming = False
    return JSONResponse(content={"status": 1, "content": "stream stopped"})