import os
import time
import argparse
import threading

# Web framework
import socketio
import eventlet
import eventlet.wsgi

from flask import Flask, send_from_directory
from flask_cors import CORS

# Image Processing
import cv2
from PIL import Image

# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)

from adversarial_detection import AdversarialDetection

from scipy.special import expit
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from yolov3 import yolov3_anchors, yolov3_tiny_anchors
from yolov3 import yolo_process_output, draw_bounding_box, letterbox_resize, yolo_correct_boxes

classes = []
adv_detect = None

# Initialize the camera
camera = cv2.VideoCapture(0)

# Initialize the server
sio = socketio.Server(cors_allowed_origins='*')

# Initialize the flask (web) app
app = Flask(__name__)
CORS(app)

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# Static website

@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
    return send_from_directory(root, path)

@app.route('/', methods=['GET'])
def redirect_to_index():
    return send_from_directory(root, 'index.html')

# Websocket server

@sio.on('connect')
def connect(sid, environ):
    print("Connected ", sid)

@sio.on('fix_patch')
def fix_patch(sid, data):
    if(data > 0):
        # Stop iterating if we choose to fix the patch
        adv_detect.fixed = True
    else:
        adv_detect.fixed = False
    print("Fix Patch: ", data)

@sio.on('clear_patch')
def clear_patch(sid, data):
    if(data > 0):
        if adv_detect.monochrome:
            adv_detect.noise = np.zeros((416, 416))
        else:
            adv_detect.noise = np.zeros((416, 416, 3))
        adv_detect.iter = 0
    print('Clear Patch')

@sio.on('save_patch')
def save_patch(sid):
    if adv_detect:
        np.save('noise/noise.npy', adv_detect.noise)
        img = Image.fromarray(adv_detect.noise, 'RGB')
        img.save('noise/noise.png')
    print('Saved Filter')

# Detetion thread
def adversarial_detection_thread():  
    global adv_detect
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    while(True): 
        # Capture the video frame
        success, origin_cv_image = camera.read()  # read the camera frame
        if not success:
            break

        input_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_BGR2RGB)

        # For YOLO, the input pixel values are normalized to [0, 1]
        if args.letter_box:
            input_cv_image = letterbox_resize(Image.fromarray(input_cv_image), (416, 416))
        else:
            input_cv_image = cv2.resize(input_cv_image, (416, 416))

        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        start_time = int(time.time() * 1000)

        # Yolo inference
        input_cv_image, outs = adv_detect.attack(input_cv_image)
        boxes, class_ids, confidences = yolo_process_output(outs, yolov3_tiny_anchors, len(classes))

        # Calculate FPS
        elapsed_time = int(time.time()*1000) - start_time
        fps = 1000 / elapsed_time
        print ("fps: ", str(round(fps, 2)))

        # Draw bounding boxes
        out_img = cv2.cvtColor(input_cv_image, cv2.COLOR_RGB2BGR)
        out_img = draw_bounding_box(out_img, boxes, confidences, class_ids, classes, colors)
        cv2.imshow("result", out_img)
        cv2.waitKey(1)

# Websocket thread
def websocket_server_thread():
    global app
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 9090)), app)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Adversarial Detection')
    parser.add_argument('--model', help='deep learning model', type=str, required=True)
    parser.add_argument('--class_name', help='class names', type=str, required=True)
    parser.add_argument('--attack', help='adversarial attacks type', choices=['one_targeted', 'multi_targeted', 'multi_untargeted'], type=str, required=False, default="multi_untargeted")
    parser.add_argument('--monochrome', action='store_true', help='monochrome patch')
    parser.add_argument('--letter_box', action='store_true', help='use letter box resize')
    args = parser.parse_args()

    # Read class names
    with open(args.class_name) as f:
        content = f.readlines()
    classes = [x.strip() for x in content] 

    adv_detect = AdversarialDetection(args.model, args.attack, args.monochrome, classes)

    t1 = threading.Thread(target=adversarial_detection_thread, daemon=True)
    t1.start()

    t2 = threading.Thread(target=websocket_server_thread, daemon=True)
    t2.start()

    t1.join()
    t2.join()
