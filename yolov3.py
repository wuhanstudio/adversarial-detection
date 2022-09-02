# Image Processing
import cv2

# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)

from scipy.special import expit
from PIL import Image

YOLOV3_INPUT_SIZE = 416

yolov3_anchors = [ 
            [[116,90],  [156,198],  [373,326]], 
            [[30,61],  [62,45],  [59,119]],  
            [[10,13],  [16,30],  [33,23]] 
            ]

yolov3_tiny_anchors = [ 
            [[81,82],  [135,169],  [344,319]],
            [[10,14],  [23,27],  [37,58]]
            ]

yolov4_anchors = [
            [[142,110], [192,243], [459,401]],
            [[36,75], [76,55], [72,146]],
            [[12,16], [19,36], [40,28]]
            ]

yolov4_tiny_anchors = [ 
            [[81,82],  [135,169],  [344,319]],
            [[10,14],  [23,27],  [37,58]]
            ]

def letterbox_resize(image, target_size, return_padding_info=False):
    """
    Resize image with unchanged aspect ratio using padding
    # Arguments
        image: origin image to be resize
            PIL Image object containing image data
        target_size: target image size,
            tuple of format (width, height).
        return_padding_info: whether to return padding size & offset info
            Boolean flag to control return value
    # Returns
        new_image: resized PIL Image object.
        padding_size: padding image size (keep aspect ratio).
            will be used to reshape the ground truth bounding box
        offset: top-left offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    src_w, src_h = image.size
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale = min(target_w/src_w, target_h/src_h)
    padding_w = int(src_w * scale)
    padding_h = int(src_h * scale)
    padding_size = (padding_w, padding_h)

    dx = (target_w - padding_w)//2
    dy = (target_h - padding_h)//2
    offset = (dx, dy)

    # create letterbox resized image
    image = image.resize(padding_size, Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128,128,128))
    new_image.paste(image, offset)

    if return_padding_info:
        return new_image, padding_size, offset
    else:
        return new_image

def yolo_correct_boxes(boxes, img_shape, model_input_shape):
    '''rescale predicition boxes back to original image shape'''
    boxes = np.array(boxes)
    box_xy = boxes[..., :2]
    box_wh = boxes[..., 2:4]

    # model_input_shape & image_shape should be (height, width) format
    model_input_shape = np.array(model_input_shape, dtype='float32')
    image_shape = np.array(img_shape, dtype='float32')

    new_shape = np.round(image_shape * np.min(model_input_shape/image_shape))
    offset = (model_input_shape-new_shape)/2./model_input_shape

    scale = model_input_shape/new_shape

    # reverse offset/scale to match (w,h) order
    offset = offset[..., ::-1]
    scale = scale[..., ::-1]

    box_xy = (box_xy - offset) * scale
    box_wh *= scale

    # Convert centoids to top left coordinates
    # box_xy -= box_wh / 2

    return np.concatenate([box_xy, box_wh], axis=1)

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, boxes, confidences, class_ids, class_names, colors):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        x = x - w / 2
        y = y - h / 2
        x = int(x * img.shape[1])
        y = int(y * img.shape[0])
        w = int(w * img.shape[1])
        h = int(h * img.shape[0])
        label = str(class_names[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
        # print(label)
        
        # Draw the bounding box on the image with label
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_ids[i]], 2)
    
    return img

def yolo_process_output(outs, anchors, num_classes):
    # The output of YOLO consists of three scales, each scale has three anchor boxes

    # For different scales, we need to calculate the actual size of three potential anchor boxes
    class_ids = []
    confidences = []
    boxes = []

    for anchor_i, out in enumerate(outs):

        anchor = anchors[anchor_i]
        num_anchors = len(anchor)

        grid_size = np.shape(out)[1:3]
        out = out.reshape((-1, 5 + num_classes))

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
        box_wh = (np.exp(out[..., 2:4]) * anchor) / np.array((YOLOV3_INPUT_SIZE, YOLOV3_INPUT_SIZE))[::-1]

        # Calculate the confidence value of each bounding box
        score = expit(out[:, 5:]) # class_scores
        class_id = np.argmax(score, axis=1)
        confidence = np.expand_dims(expit(out[:, 4]), -1) * score # class_score * objectness
        confidence = np.max(confidence, axis=-1)

        # We are only interested with the box with high confidence
        confidence_threshold = 0.1
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
            for c in class_id:
                class_ids.append(c)

    # Eliminate the boxes with low confidence and overlaped boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        indexes = indexes.flatten()

    boxes = [boxes[i] for i in indexes]
    confidences = [confidences[i] for i in indexes]
    class_ids = [class_ids[i] for i in indexes]

    return boxes, class_ids, confidences
