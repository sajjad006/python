import cv2
import zmq
import base64
import numpy as np

camera = cv2.VideoCapture(0)

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://*:5555')
footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

while True:
    ret, frame_self = camera.read()
    frame = footage_socket.recv_string()
    img = base64.b64decode(frame)
    npimg = np.fromstring(img, dtype=np.uint8)
    source = cv2.imdecode(npimg, 1)
    source[0:120, 0:180] = frame_self
    cv2.imshow("Stream", source)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("here")
        cv2.destroyAllWindows()
        break