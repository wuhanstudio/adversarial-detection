import argparse
import time

import cv2
import atexit

# Image Processing
from PIL import Image


# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model
from scipy.special import expit, softmax
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keras.backend as K

classes = []
noise = None

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Adversarial Detection')
    parser.add_argument('--model', help='deep learning model', type=str, required=True)
    parser.add_argument('--class_name', help='class names', type=str, required=True)
    parser.add_argument('--noise', help='noise', type=str, required=False)
    args = parser.parse_args()

    model = load_model(args.model)
    model.summary()

    sess = tf.compat.v1.keras.backend.get_session()

    with open(args.class_name) as f:
        content = f.readlines()
    classes = [x.strip() for x in content] 

    if args.noise:
        noise = np.load(args.noise)

    # define a video capture object
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  
    while(True):
          
        # Capture the video frame
        ret, input_cv_image = vid.read()

        input_cv_image = input_cv_image.astype(np.float32) / 255.0

        # Add noise
        if noise is not None:
            height, width, channel = input_cv_image.shape
            input_cv_image = cv2.resize(input_cv_image, (noise.shape[0], noise.shape[1]), interpolation = cv2.INTER_AREA)
            input_cv_image = input_cv_image + noise
            input_cv_image = cv2.resize(input_cv_image, (height, width), interpolation = cv2.INTER_AREA)

        input_cv_image = np.clip(input_cv_image, 0, 1)

        input_cv_image = cv2.resize(input_cv_image, (416, 416), interpolation = cv2.INTER_AREA)
        # input_cv_image = input_cv_image[int(input_cv_image.shape[0]/2 - 208):int(input_cv_image.shape[0]/2+208), int(input_cv_image.shape[1]/2 - 208):int(input_cv_image.shape[1]/2+208), :]

        # Read the input image and pulish to the browser
        input_cv_image = np.array(input_cv_image)

        # For YOLO, the input pixel values are normalized to [0, 1]
        height, width, channels = input_cv_image.shape
     
        start_time = int(time.time() * 1000)
        input_cv_image = Image.fromarray(np.uint8(input_cv_image * 255.0))
        b, g, r = input_cv_image.split()
        input_cv_image = np.array(Image.merge("RGB", (r, g, b)))

        input_cv_image = input_cv_image.astype(np.float32) / 255.0
        input_cv_image = np.clip(input_cv_image, 0, 1)

        outs = sess.run(model.output, feed_dict={model.input:np.array([input_cv_image])})

        # Showing informations on the screen (YOLO)
        # The output of YOLO consists of three scales, each scale has three anchor boxes
        anchors = [ 
                    [[116,90],  [156,198],  [373,326]], 
                    [[30,61],  [62,45],  [59,119]],  
                    [[10,13],  [16,30],  [33,23]] 
                  ]
        # For different scales, we need to calculate the actual size of three potential anchor boxes
        for anchor_i, out in enumerate(outs):
            class_ids = []
            confidences = []
            boxes = []
            scores = []

            anchor = anchors[anchor_i]
            num_anchors = len(anchor)

            grid_size = np.shape(out)[1:3]
            out = out.reshape((-1, 5+len(classes)))

            # The output of each bounding box is relative to the grid
            # Thus we need to add an offset to retrieve absolute coordinates
            grid_y = np.arange(grid_size[0])
            grid_x = np.arange(grid_size[1])
            x_offset, y_offset = np.meshgrid(grid_x, grid_y)

            x_offset = np.reshape(x_offset, (-1, 1))
            y_offset = np.reshape(y_offset, (-1, 1))

            x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
            x_y_offset = np.tile(x_y_offset, (1, num_anchors))
            x_y_offset = np.reshape(x_y_offset, (-1, 2))

            anchor = np.tile(anchor, (grid_size[0] * grid_size[1], 1))

            # The output format of each bounding box is (x, y, w, h)
            box_xy = (expit(out[..., :2]) + x_y_offset) / np.array(grid_size)[::-1]
            box_wh = (np.exp(out[..., 2:4]) * anchor) / np.array((height, width))[::-1]

            # Calculate the confidence value of each bounding box
            score = expit(out[:, 5:])
            class_id = np.argmax(score, axis=1)
            score = score[class_id][:, 0]
            confidence = score * expit(out[:, 4])

            # We are only interested with the box with high confidence
            confidence_threshold = 0.01
            box_xy = box_xy[confidence > confidence_threshold]
            box_wh = box_wh[confidence > confidence_threshold]
            class_id = class_id[confidence > confidence_threshold]
            score = score[confidence > confidence_threshold]
            confidence = confidence[confidence > confidence_threshold]

            if(len(confidence) > 0):
                box_tmp = list(np.concatenate((box_xy, box_wh), axis=1))
                for b in box_tmp:
                    boxes.append(b)
                for c in confidence:
                    confidences.append(float(c))
                for s in score:
                    scores.append(float(s))
                for c in class_id:
                    class_ids.append(c)

            # Eliminate the boxes with low confidence and overlaped boxes
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    x = x - w / 2
                    y = y - h / 2
                    x = int(x * width) 
                    y = int(y * height)
                    w = int(w * width) 
                    h = int(h * height) 
                    label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                    print(label)
                    
                    # Draw the bounding box on the image with label
                    cv2.rectangle(input_cv_image, (x, y), (x + w, y + h), (255,0,0), 2)
                    cv2.putText(input_cv_image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        elapsed_time = int(time.time()*1000) - start_time
        fps = 1000 / elapsed_time
        print ("fps: ", str(round(fps, 2)))

        input_cv_image = Image.fromarray(np.uint8(input_cv_image*255.0))
        b, g, r = input_cv_image.split()
        input_cv_image = np.array(Image.merge("RGB", (r, g, b)))

        cv2.imshow("result", input_cv_image)
        cv2.waitKey(1)
