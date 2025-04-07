#### 前端推流

笔记本摄像头名称：USB2.0 HD UVC WebCam

ffmpeg.exe  -f dshow -video_size 640x480 -i video="USB2.0 HD UVC WebCam" -f rawvideo -pix_fmt bgr24 pipe:1



MJPEG+前端img轮询实现图片在前端显示



#### status码

1：正常返回

2：错误



#### 启动

uvicorn main:app --host 0.0.0.0 --port 8000
