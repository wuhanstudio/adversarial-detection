import socketio
import cv2
from time import sleep

# Data IO and Encoding-Decoding
from io import BytesIO
import base64

# Image Processing
from PIL import Image

import numpy as np

sio = socketio.Client()

# From image to base64 string
def img2base64(image):
    origin_img = Image.fromarray(np.uint8(image))
    origin_buff = BytesIO()
    origin_img.save(origin_buff, format="JPEG")

    return base64.b64encode(origin_buff.getvalue()).decode("utf-8")

@sio.event
def connect():
    print('connection established')

@sio.event
def disconnect():
    print('disconnected from server')

sio.connect('http://localhost:9090')

# Capture frame-by-frame (the latest one)
# cam = cv2.VideoCapture("http://192.168.199.240:4747/mjpegfeed?640x480")

# Capture frame-by-frame (the latest one)
# cam = cv2.VideoCapture(0)

while True:
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        continue

    # Crop the center 416 x 416
    # frame = cv2.resize(frame, (416, 416))
    frame = frame[int(frame.shape[0]/2 - 208):int(frame.shape[0]/2+208), int(frame.shape[1]/2 - 208):int(frame.shape[1]/2+208), :]
    
    frame = Image.fromarray(np.uint8(frame))
    b, g, r = frame.split()
    frame = np.array(Image.merge("RGB", (r, g, b)))

    sio.emit('frame', {'image': img2base64(frame)})

    # cv2.imshow("Test", np.uint8(frame))
    # cv2.waitKey(1)
