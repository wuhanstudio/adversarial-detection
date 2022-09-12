import os
import argparse
from adversarial_detection import AdversarialDetection
import time

import cv2

# Web framework
import socketio
import eventlet
import eventlet.wsgi

import threading

from flask import Flask, send_from_directory
from flask_cors import CORS

# Image Processing
from PIL import Image

# Data IO and Encoding-Decoding
from io import BytesIO
import base64

# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from yolov3 import yolov3_anchors, yolov3_tiny_anchors
from yolov3 import yolo_process_output, draw_bounding_box

classes = []
adv_detect = None

# Initialize the server
sio = socketio.Server(cors_allowed_origins='*', async_mode='threading')

# Initialize the flask (web) app
app = Flask(__name__)
CORS(app)

# From image to base64 string
def img2base64(image):
    img = Image.fromarray(np.uint8(image))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# Static website
@app.route('/<path:path>', methods=['GET'])
def static_proxy(path):
    return send_from_directory(root, path)

@app.route('/', methods=['GET'])
def redirect_to_index():
    return send_from_directory(root, 'index.html')

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

@sio.on('fix_patch')
def fix_patch(sid, data):
    if(data > 0):
        # Stop iterating if we choose to fix the patch
        adv_detect.fixed = True

        # Save each patch
        patch_cv_image = np.zeros((416, 416, 3))
        for box in adv_detect.adv_patch_boxes:
            if adv_detect.monochrome:
                # For black and white images R==G==B
                patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 0] = adv_detect.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 1] = adv_detect.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
                patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), 2] = adv_detect.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2])]
            else:
                patch_cv_image[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :] = adv_detect.noise[box[1]:(box[1]+box[3]), box[0]:(box[0] + box[2]), :]
        # Publish the patch image
        sio.emit('patch', {'data': img2base64(patch_cv_image*255.0), 'boxes': adv_detect.adv_patch_boxes})
    else:
        adv_detect.fixed = False

@sio.on('clear_patch')
def clear_patch(sid, data):
    if(data > 0):
        adv_detect.adv_patch_boxes = []
        if adv_detect.monochrome:
            adv_detect.noise = np.zeros((416, 416))
        else:
            adv_detect.noise = np.zeros((416, 416, 3))
        adv_detect.iter = 0
        adv_detect.fixed = False

@sio.on('add_patch')
def add_patch(sid, data):
    box = data[1:]
    if(data[0] < 0):
        adv_detect.adv_patch_boxes.append(box)
        adv_detect.iter = 0
    else:
        adv_detect.adv_patch_boxes[data[0]] = box

# Detetion thread
def adversarial_detection_thread():  
    global sio, adv_detect
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Capture the video frame
    origin_cv_image = cv2.imread("notebooks/demo.jpg")  # read the camera frame
    if origin_cv_image is None:
        print("Failed to open the camera")
        exit()

    while(True): 
        # Read the input image and pulish to the browser
        input_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_BGR2RGB)
        sio.emit('input', {'data': img2base64(input_cv_image)})

        # For YOLO, the input pixel values are normalized to [0, 1]
        input_cv_image = cv2.resize(input_cv_image, (416, 416))
        input_cv_image = input_cv_image.astype(np.float32) / 255.0
    
        start_time = int(time.time() * 1000)
        outs = adv_detect.attack(input_cv_image)

        # Yolo inference
        input_cv_image, outs = adv_detect.attack(input_cv_image)
        boxes, class_ids, confidences = yolo_process_output(outs, yolov3_tiny_anchors, len(classes))

        # Draw bounding boxes
        out_img = draw_bounding_box(input_cv_image, boxes, confidences, class_ids, classes, colors)

        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("output", out_img)
        cv2.waitKey(1)

        elapsed_time = int(time.time()*1000) - start_time
        fps = 1000 / elapsed_time
        print ("fps: ", str(round(fps, 2)))

        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

        # Send the output image to the browser
        sio.emit('adv', {'data': img2base64(out_img*255.0)})

        eventlet.sleep()

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
    args = parser.parse_args()

    with open(args.class_name) as f:
        content = f.readlines()
    classes = [x.strip() for x in content] 

    adv_detect = AdversarialDetection(args.model, args.attack, args.monochrome, classes)

    t1 = threading.Thread(target=websocket_server_thread, daemon=True)
    t1.start()

    t2 = threading.Thread(target=adversarial_detection_thread, daemon=True)
    t2.start()

    t1.join()
    t2.join()
