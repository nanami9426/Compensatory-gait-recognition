#### FFmpeg使用

笔记本摄像头名称：USB2.0 HD UVC WebCam

ffmpeg.exe  -f dshow -video_size 640x480 -i video="USB2.0 HD UVC WebCam" -f rawvideo -pix_fmt bgr24 pipe:1





