import time
import argparse

# Deep Learning Libraries
import cv2
import numpy as np
from keras.models import load_model

from yolov3 import yolov3_anchors, yolov3_tiny_anchors
from yolov3 import yolo_process_output, draw_bounding_box

classes = []
noise = None

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Adversarial Detection')
    parser.add_argument('--model', help='deep learning model', type=str, required=True)
    parser.add_argument('--class_name', help='class names', type=str, required=True)
    parser.add_argument('--noise', help='noise', type=str, required=False)
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)
    model.summary()

    # Read class names
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
        success, input_cv_image = vid.read()
        if not success:
            break

        input_cv_image = cv2.cvtColor(input_cv_image, cv2.COLOR_BGR2RGB)

        # Add noise
        if noise is not None:
            input_cv_image = input_cv_image.astype(np.float32) / 255.0
            height, width, _ = input_cv_image.shape
            input_cv_image = cv2.resize(input_cv_image, (noise.shape[0], noise.shape[1]), interpolation = cv2.INTER_AREA)
            input_cv_image = input_cv_image + noise
            input_cv_image = np.clip(input_cv_image, 0, 1)
            input_cv_image = cv2.resize(input_cv_image, (height, width), interpolation = cv2.INTER_AREA)
            input_cv_image = input_cv_image.astype(np.float32) * 255.0

        # For YOLO, the input pixel values are normalized to [0, 1]
        input_cv_image = cv2.resize(input_cv_image, (416, 416), interpolation = cv2.INTER_AREA)
        input_cv_image = input_cv_image.astype(np.float32) / 255.0

        start_time = int(time.time() * 1000)

        # Yolo inference
        outs = model.predict(np.array([input_cv_image]))
        boxes, class_ids, confidences = yolo_process_output(outs, yolov3_anchors, len(classes))

        # Calculate FPS
        elapsed_time = int(time.time()*1000) - start_time
        fps = 1000 / elapsed_time
        print ("fps: ", str(round(fps, 2)))

        # Draw bounding boxes
        out_img = cv2.cvtColor(input_cv_image, cv2.COLOR_RGB2BGR)
        out_img = draw_bounding_box(out_img, boxes, confidences, class_ids, classes)
        cv2.imshow("result", out_img)
        cv2.waitKey(1)
