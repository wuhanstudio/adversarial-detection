import argparse

# Image Processing
import cv2
import numpy as np
from PIL import Image

# Data IO and Encoding-Decoding
import base64
import socketio
from io import BytesIO

sio = socketio.Client()

noise = None

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

# Parse arguments
parser = argparse.ArgumentParser(description='Adversarial Detection Camera')
parser.add_argument('--camera', help='camera index', type=int, default=0)
parser.add_argument('--noise', help='noises (0.0, 1.0) in numpy format (*.npy)', type=str, required=False)
args = parser.parse_args()

if args.noise:
    noise = np.load(args.noise)

sio.connect('http://localhost:9090')

# Capture frame-by-frame (the latest one)
cam = cv2.VideoCapture(args.camera)

while True:

    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        continue

    # Crop the center 416 x 416
    frame = frame[int(frame.shape[0]/2 - 208):int(frame.shape[0]/2+208), int(frame.shape[1]/2 - 208):int(frame.shape[1]/2+208), :]
    
    # Resize the image
    # frame = cv2.resize(frame, (416, 416))

    # BGR to RGB
    frame = Image.fromarray(np.uint8(frame))
    b, g, r = frame.split()
    frame = np.array(Image.merge("RGB", (r, g, b)))

    # Float to uint8
    frame = frame.astype(np.float32) / 255.0

    # Test perturbation
    if noise is not None:
        frame = frame + noise

    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

    # Convert image to base64 string and send to server
    sio.emit('frame', {'image': img2base64(frame)})

    # cv2.imshow("Camera", np.uint8(frame))
    # cv2.waitKey(1)
