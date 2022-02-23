import time
import argparse

# Deep Learning Libraries
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model

from yolov3 import yolov3_anchors, yolov3_tiny_anchors
from yolov3 import yolo_process_output, letterbox_resize, yolo_correct_boxes, draw_bounding_box

classes = []
noise = None

def bilinear_resize_vectorized(image, height, width):
  """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
  img_height, img_width = image.shape

  image = image.ravel()

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  y, x = np.divmod(np.arange(height * width), width)

  x_l = np.floor(x_ratio * x).astype('int32')
  y_l = np.floor(y_ratio * y).astype('int32')

  x_h = np.ceil(x_ratio * x).astype('int32')
  y_h = np.ceil(y_ratio * y).astype('int32')

  x_weight = (x_ratio * x) - x_l
  y_weight = (y_ratio * y) - y_l

  a = image[y_l * img_width + x_l]
  b = image[y_l * img_width + x_h]
  c = image[y_h * img_width + x_l]
  d = image[y_h * img_width + x_h]

  resized = a * (1 - x_weight) * (1 - y_weight) + \
            b * x_weight * (1 - y_weight) + \
            c * y_weight * (1 - x_weight) + \
            d * x_weight * y_weight

  return resized.reshape(height, width)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Adversarial Detection')
    parser.add_argument('--model', help='deep learning model', type=str, required=True)
    parser.add_argument('--class_name', help='class names', type=str, required=True)
    parser.add_argument('--noise', help='noise', type=str, required=False)
    parser.add_argument('--letter_box', action='store_true', help='use letter box resize')
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)
    model.summary()

    # Read class names
    with open(args.class_name) as f:
        content = f.readlines()
    classes = [x.strip() for x in content] 

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if args.noise:
        noise = np.load(args.noise)

    # define a video capture object
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  
    while(True):
        # Capture the video frame
        success, origin_cv_image = vid.read()
        if not success:
            break

        input_cv_image = cv2.cvtColor(origin_cv_image, cv2.COLOR_BGR2RGB)

        # Add noise
        if noise is not None:
            input_cv_image = input_cv_image.astype(np.float32) / 255.0
            height, width, _ = input_cv_image.shape
            # input_cv_image = cv2.resize(input_cv_image, (noise.shape[0], noise.shape[1]), interpolation = cv2.INTER_AREA)
            noise_r = bilinear_resize_vectorized(noise[:, :, 0], height, width)
            noise_g = bilinear_resize_vectorized(noise[:, :, 1], height, width)
            noise_b = bilinear_resize_vectorized(noise[:, :, 2], height, width)
            noise = np.dstack((noise_r, noise_g, noise_b))

            input_cv_image = input_cv_image + noise
            input_cv_image = np.clip(input_cv_image, 0, 1)

            # input_cv_image = cv2.resize(input_cv_image, (width, height), interpolation = cv2.INTER_AREA)
            input_cv_image = (input_cv_image * 255.0).astype(np.uint8)

        # For YOLO, the input pixel values are normalized to [0, 1]
        if args.letter_box:
            input_cv_image = letterbox_resize(Image.fromarray(input_cv_image), (416, 416))
        else:
            input_cv_image = cv2.resize(input_cv_image, (416, 416))

        input_cv_image = np.array(input_cv_image).astype(np.float32) / 255.0

        start_time = int(time.time() * 1000)

        # Yolo inference
        outs = model.predict(np.array([input_cv_image]))
        boxes, class_ids, confidences = yolo_process_output(outs, yolov3_anchors, len(classes))

        if(len(boxes) > 0 and args.letter_box):
            boxes = yolo_correct_boxes(boxes, origin_cv_image.shape[:2], (416, 416))

        # Calculate FPS
        elapsed_time = int(time.time()*1000) - start_time
        fps = 1000 / elapsed_time
        print ("fps: ", str(round(fps, 2)))

        # Draw bounding boxes
        out_img = draw_bounding_box(origin_cv_image, boxes, confidences, class_ids, classes, colors)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", out_img)
        cv2.waitKey(1)
