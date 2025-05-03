# 代偿步态检测系统
这是我的毕业设计，可以识别被检测主体是否出现代偿步态。

前端操作界面在我另一个仓库中。





#### 前端推流

笔记本摄像头名称：USB2.0 HD UVC WebCam

ffmpeg.exe  -f dshow -video_size 640x480 -i video="USB2.0 HD UVC WebCam" -f rawvideo -pix_fmt bgr24 pipe:1



MJPEG+前端img轮询实现图片在前端显示



#### status码

1：正常返回

2：错误



#### 启动

8000端口
