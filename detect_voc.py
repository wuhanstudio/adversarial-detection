import datetime
import argparse
from tqdm import tqdm

# Deep Learning Libraries
import cv2
import numpy as np
np.set_printoptions(suppress=True)

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from adversarial_detection import AdversarialDetection

import fiftyone.zoo as foz
from logger import TensorBoardLogger

from yolov3 import yolov3_anchors, yolov3_tiny_anchors
from yolov3 import yolo_process_output, draw_bounding_box

n_iteration = 10

classes = []
adv_detect = None

# Tensorboard
log_dir = 'logs/xi/8/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoardLogger(log_dir)

# Load Training Dataset from FiftyOne
train_dataset = foz.load_zoo_dataset("voc-2012", split="validation")
img_paths = train_dataset.values("filepath")

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

    adv_detect = AdversarialDetection(args.model, args.attack, args.monochrome, classes, xi=8/255.0, lr=1/255.0)

    adv_detect.adv_patch_boxes = [[168, 168, 80, 80]]
    # adv_detect.adv_patch_boxes = [[188, 188, 40, 40]]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    voc_success_rate = []
    voc_boxes_increase = []

    for i in tqdm(range(len(img_paths))):
        if args.monochrome:
            adv_detect.noise = np.zeros((416, 416))
        else:
            adv_detect.noise = np.zeros((416, 416, 3))
 
        img = cv2.imread(str(img_paths[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # For YOLO, the input pixel values are normalized to [0, 1]
        original_cv_image = cv2.resize(img, (416, 416))
        original_cv_image = np.array(original_cv_image).astype(np.float32) / 255.0

        original_outs = adv_detect.sess.run(adv_detect.model.output, feed_dict={adv_detect.model.input:np.array([original_cv_image])})
        original_boxes, _, _ = yolo_process_output(original_outs, yolov3_tiny_anchors, len(classes))

        box_increase = []
        for t in range(n_iteration):
            # Yolo inference
            output_cv_image, outs = adv_detect.attack(np.copy(original_cv_image))
            boxes, class_ids, confidences = yolo_process_output(outs, yolov3_tiny_anchors, len(classes))

            # Draw bounding boxes
            # out_img = draw_bounding_box(output_cv_image, boxes, confidences, class_ids, classes, colors)

            # out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            # cv2.imshow("output", out_img)
            # cv2.waitKey(1)

            box_increase.append(len(boxes) - len(original_boxes))

        voc_boxes_increase.append(box_increase)

    for t in range(n_iteration):
        tb.log_scalar('success rate', np.mean(np.array(voc_boxes_increase)[:, t]>0), t)
        tb.log_scalar('box increase', np.mean(np.array(voc_boxes_increase)[:, t]), t)
