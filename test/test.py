from ultralytics import YOLO
import logging
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.WARNING)

model = YOLO('../models/yolo11n-pose.pt')

import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    s = results[0].keypoints.xy.shape
    if s[0] > 1:
        print(results[0].boxes.conf.argmax(-1))
        idx = results[0].boxes.conf.argmax(-1).item()
        print(len(results[0].boxes.cls))
        print(results[0].boxes.conf)
        print(results[0].keypoints.xy[idx].unsqueeze(0).shape)
        break

    # cv2.imshow('YOLO11n-pose', frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
