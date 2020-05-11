import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'OpenCV Tuts!',(10,500), font, 6, (200,255,155), 13, cv2.LINE_AA)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
