import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import optimizers
import keras_cv
import numpy as np
from keras_cv import bounding_box
import os
import resource
from keras_cv import visualization
from tqdm import tqdm
from pathlib import Path
from os import listdir
from os.path import isfile, join
from termcolor import colored



class_ids = [
"Truck",
"Car",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))



BATCH_SIZE = 4


def load_images(root, train_or_val):
    image_dir = os.path.join(root, "images", train_or_val)
    image_paths = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
    image_arr = []
    for image_file in tqdm(image_paths):
        image = tf.keras.utils.load_img(os.path.join(image_dir,image_file))
        input_arr = tf.keras.utils.img_to_array(image)
        image_arr.append(tf.cast(input_arr, tf.float32))
    images = tf.convert_to_tensor(np.array(image_arr))
    return images

def load_bounding_boxes(root, train_or_val):
    label_dir = os.path.join(root, "labels", train_or_val)
    label_paths = [f for f in listdir(label_dir) if isfile(join(label_dir, f))]
    bounding_boxes = {
            "classes": [],
            "boxes": []
            }

    for label_file in tqdm(label_paths):
        classes = []
        boxes = []
        with open(os.path.join(label_dir,label_file)) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(" ")
                class_id = int(line[0])
                x = float(line[1])
                y = float(line[2])
                w = float(line[3])
                h = float(line[4])
                x = x - (w/2)
                y = y - (h/2)
                bounding_box = [x,y,w,h]
                classes.append(class_id)
                boxes.append(bounding_box)
        bounding_boxes["boxes"].append(boxes)
        bounding_boxes["classes"].append(classes)
    bounding_boxes["boxes"] = tf.ragged.constant(bounding_boxes["boxes"])
    bounding_boxes["classes"] = tf.ragged.constant(bounding_boxes["classes"])
    return bounding_boxes

def convert_center_rel_xywh_to_rel_xywh(boxes):
    for box in tqdm(boxes):
        box[0] = box[0] - (box[2] / 2)
        box[1] = box[1] - (box[3] / 2)
    return boxes



def load_dataset(yaml_path="data.yaml"):
    yaml_path="/home/tobias/git_ws/pul/datasets/truck_labeled/truck_labeled.yaml"
    root = os.path.dirname(os.path.realpath(yaml_path))
    val_labels = load_bounding_boxes(root, "val")
    val_images = load_images(root, "val")
    val_ds = {
        "images": val_images,
        "bounding_boxes": val_labels,
    }
    return val_ds

    



def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images=images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
        path="eval.png"
    )

def main():
    train_ds = load_dataset()
    train_ds = tf.data.Dataset.from_tensor_slices(train_ds)


    train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.shuffle(BATCH_SIZE)


    visualize_dataset(
        train_ds, bounding_box_format="rel_xywh", value_range=(0, 255), rows=1, cols=1
    )


if __name__ == "__main__":
    main()