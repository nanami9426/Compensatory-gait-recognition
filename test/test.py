from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')

import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)

    annotated_frame = results[0].plot()
    print(results[0].keypoints.xy.shape)

    cv2.imshow('YOLO11n-pose', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
