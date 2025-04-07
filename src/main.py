from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import threading

app = FastAPI()
streaming = True
lock = threading.Lock()

def process_frame(frame):
    frame = cv2.flip(frame, 1)
    return frame

def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        with lock:
            if not streaming:
                cap.release()
                break
        
        success, frame = cap.read()
        if not success:
            break
        
        frame = process_frame(frame)
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