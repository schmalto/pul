import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image
import numpy as np


def convert_center_xywh_to_xyxy(bounding_box, img_width, img_height):
    x = bounding_box[0]
    y = bounding_box[1]
    w = bounding_box[2]
    h = bounding_box[3]

    x = x - (w / 2)
    y = y - (h / 2)

    x1 = int(x * img_width)
    y1 = int(y * img_height)
    x2 = int(x1 + (w * img_width))
    y2 = int(h + (h * img_height))
    return [x1,y1,x2,y2]

'''
@returns: image in PIL format
'''
def load_image(image_path):
    image = Image.open(image_path)
    return image

def save_image_arr(image, save_path):
    image = Image.fromarray(image)
    image.save(save_path)

def load_bounding_boxes(bounding_boxes_path, image):
    img_w, img_h, _ = image.shape
    bounding_boxes = []
    with open(bounding_boxes_path) as f:
        lines = f.readlines()
        for line in lines:
            box = line.split(" ")
            box = [float(x) for x in box]
            box = convert_center_xywh_to_xyxy(box, img_w, img_h)
            bounding_boxes.append(BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
    bbs = BoundingBoxesOnImage(bounding_boxes, shape=image.shape)
    return bbs

if __name__ == '__main__':
    print("Hello World")