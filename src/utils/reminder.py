import cv2

img = cv2.imread("./utils/remind.jpg")
_, buffer = cv2.imencode('.jpg', img)

reminder_bytes = buffer.tobytes()
