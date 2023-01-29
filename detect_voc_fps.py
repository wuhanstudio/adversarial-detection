import gc
import time
import datetime
import argparse
from tqdm import tqdm

# Deep Learning Libraries
import cv2
import numpy as np
np.set_printoptions(suppress=True)

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

from adversarial_detection import AdversarialDetection

import fiftyone as fo
fo.config.default_ml_backend = "tensorflow"
import fiftyone.zoo as foz

from yolov3 import yolov3_anchors, yolov3_tiny_anchors
from yolov3 import yolo_process_output, draw_bounding_box

n_iteration = 20

classes = []

# Load Training Dataset from FiftyOne
train_dataset = foz.load_zoo_dataset("voc-2012", split="validation")
img_paths = train_dataset.values("filepath")

img_paths = img_paths[:int(len(img_paths))]

del train_dataset
gc.collect()

def voc_benchmark(model, attack, monochrome, classes, xi, lr, boxes):
    adv_detect = AdversarialDetection(model, attack, monochrome, classes, xi = xi / 255.0, lr = lr / 255.0)

    adv_detect.adv_patch_boxes = [boxes]

    voc_boxes_increase = []
    voc_attack_time = []

    for i in tqdm(range(len(img_paths))):
        adv_detect.noise = np.zeros((416, 416, 3))

        img = cv2.imread(str(img_paths[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # For YOLO, the input pixel values are normalized to [0, 1]
        original_cv_image = cv2.resize(img, (416, 416))
        original_cv_image = np.array(original_cv_image).astype(np.float32) / 255.0

        original_outs = adv_detect.sess.run(adv_detect.model.output, feed_dict={adv_detect.model.input:np.array([original_cv_image])})
        original_boxes, _, _ = yolo_process_output(original_outs, yolov3_tiny_anchors, len(classes))

        box_increase = []
        attack_time = []

        for t in range(n_iteration):
            # Yolo inference
            start_time = time.time() # start time of the loop
            output_cv_image, outs = adv_detect.attack(np.copy(original_cv_image))
            attack_time.append(time.time() - start_time)

            boxes, class_ids, confidences = yolo_process_output(outs, yolov3_tiny_anchors, len(classes))

            # Draw bounding boxes
            # out_img = draw_bounding_box(output_cv_image, boxes, confidences, class_ids, classes, colors)

            # out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            # cv2.imshow("output", out_img)
            # cv2.waitKey(1)

            box_increase.append(len(boxes) - len(original_boxes))

        voc_boxes_increase.append(box_increase)
        voc_attack_time.append(attack_time)

    print(adv_detect.adv_patch_boxes)
    print(np.array(voc_attack_time)[:, 1:2].sum(axis=1).mean())
    print(np.array(voc_attack_time)[:, 1:11].sum(axis=1).mean())
    print(np.array(voc_attack_time)[:, 1:21].sum(axis=1).mean())

    it_1 = []
    it_3 = []
    it_5 = []
    for vbi in voc_boxes_increase:
        if 1 in vbi:
            it_1.append(vbi.index(1))
        if 3 in vbi:
            it_3.append(vbi.index(3))
        if 5 in vbi:
            it_5.append(vbi.index(5))
    print(np.mean(it_1), np.mean(it_3), np.mean(it_5))

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Adversarial Detection')
    parser.add_argument('--model', help='deep learning model', type=str, required=True)
    parser.add_argument('--class_name', help='class names', type=str, required=True)

    args = parser.parse_args()

    with open(args.class_name) as f:
        content = f.readlines()
    classes = [x.strip() for x in content] 

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for attack in ["multi_untargeted"]:
        xi = 8
        lr = 2
        # box
        for box in [[0, 0, 64, 64], 
                    [0, 0, 128, 128],]:

            voc_benchmark(args.model, attack, False, classes, xi, lr, box)
