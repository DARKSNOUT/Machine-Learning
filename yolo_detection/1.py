import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')  
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, show=False, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.conf < 0.3:
                continue
            x1, y1, x2, y2 = box.xyxy[0]
            x1 , y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if int(box.cls) == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 255, 150), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # print(float(box.conf))
            # print(model.names[int(box.cls)])
            cv2.putText(frame, f'{model.names[int(box.cls)]} {float(box.conf):.2f}', 
                        (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()