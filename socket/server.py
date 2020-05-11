import base64
import cv2
import zmq

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://192.168.43.90:5555')

camera = cv2.VideoCapture(0)  # init the camera

while True:
    grabbed, frame = camera.read()  # grab the current frame
    frame = cv2.resize(frame, (640, 480))  # resize the frame

    cv2.imshow("winname", frame)
    
    encoded, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    footage_socket.send(jpg_as_text)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        break