import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
#from tensorflow.keras import optimizers
import keras_cv
import numpy as np
import keras
from keras_cv import bounding_box
from keras.preprocessing.image import ImageDataGenerator
import os
import resource
from keras_cv import visualization
from tqdm import tqdm
from pathlib import Path
from os import listdir
from os.path import isfile, join
from termcolor import colored
from pathlib import Path, PurePath
import pickle
import mgzip
import gzip



class_ids = [
"Truck",
"Car",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))



BATCH_SIZE = 8


# def load_images(root, train_or_val):
#     image_dir = os.path.join(root, "images", train_or_val)
#     image_paths = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
#     image_arr = []
#     for image_file in tqdm(image_paths):
#         image = tf.keras.utils.load_pathlib.PurePathimg(os.path.join(image_dir,image_file))
#         input_arr = tf.keras.utils.img_to_array(image)
#         image_arr.append(tf.cast(input_arr, tf.float32))
#     images = tf.convert_to_tensor(np.array(image_arr))
#     return images

def load_images(root, train_or_val, size):
    image_dir = os.path.join(root, "images", train_or_val)
    image_paths = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
    file_list = []
    image_arr = []
    for image_file in tqdm(image_paths):
        image = tf.keras.utils.load_img(os.path.join(image_dir, image_file))
        input_arr = tf.keras.utils.img_to_array(image)
        if input_arr.shape[0] == size and input_arr.shape[1] == size:
            image_arr.append(np.array(input_arr, dtype=np.float32))
        else:
            file_list.append(image_file.split(".")[0])
    images = tf.stack(image_arr)
    #images = np.stack(image_arr)
    images = tf.cast(images, tf.float32)
    return images, file_list


def load_bounding_boxes(root, train_or_val, except_list):
    label_dir = os.path.join(root, "labels", train_or_val)
    label_paths = [f for f in listdir(label_dir) if isfile(join(label_dir, f))]
    bounding_boxes = {
            "classes": [],
            "boxes": []
            }
    for label_file in tqdm(label_paths):
        if not any(label_file in forbidden_file for forbidden_file in except_list):
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
    bounding_boxes["boxes"] = tf.ragged.constant(bounding_boxes["boxes"], ragged_rank=1)
    bounding_boxes["boxes"] = bounding_boxes["boxes"].to_tensor()
    bounding_boxes["classes"] = tf.ragged.constant(bounding_boxes["classes"]).to_tensor()
    bounding_boxes["boxes"] = tf.cast(bounding_boxes["boxes"], tf.float32)
    bounding_boxes["classes"] = tf.cast(bounding_boxes["classes"], tf.float32)
    return bounding_boxes

def convert_center_rel_xywh_to_rel_xywh(boxes):
    for box in tqdm(boxes):
        box[0] = box[0] - (box[2] / 2)
        box[1] = box[1] - (box[3] / 2)
    return boxes



def load_data( size,yaml_path="data.yaml"):
    #yaml_path="/home/tobias/git_ws/pul/datasets/truck_labeled/truck_labeled.yaml"
    root = os.path.dirname(os.path.realpath(yaml_path))
    #train_images, train_except= load_images(root, "train", size)
    val_images, val_except = load_images(root, "val", size)
    val_labels = load_bounding_boxes(root, "val", val_except)
    #train_labels = load_bounding_boxes(root, "train", train_except)
  
    val_ds = {
        "images": val_images,
        "bounding_boxes": val_labels,
    }
    # train_ds = {
    #     "images": train_images,inputs = next(iter(inputs.take(1)))
    # images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    #     "bounding_boxes": train_labels,
    # }
    return val_ds

def make_dataset(data_dict):
    dataset = tf.data.Dataset.from_tensor_slices(data_dict)
    return dataset

def save_dataset(dataset, path):
    dataset.save(path)
    return 1
    

def print_debug(str):
    print(colored(str, "red"))


def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    dataset = next(iter(inputs.take(1)))
    bounding_boxes = dataset["bounding_boxes"]
    images = dataset["images"]

    # Labels
    # 1 Bruchsal20200426_14.txt
    # 2 Bruchsal20200426_15.txt
    # 3 Bruchsal20200426_19.txt
    # 4 Bruchsal20200426_20.txt
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

def loading():
    datasets = [
        "datasets/truck_labeled_many_960_960/truck_labeled_many_960_960.yaml",
    ]
    for dataset in tqdm(datasets):
        data_yaml_path = Path(dataset).resolve()
        ds = load_data(960,yaml_path=str(data_yaml_path))
        save_dict(ds)


def training():
    eval_ds = load_dataset('data.dat.gz')
    # eval_ds = eval_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    #eval_ds = eval_ds.batch(BATCH_SIZE)
    #visualize_dataset( eval_ds, bounding_box_format="rel_xywh", value_range=(0, 255), rows=2, cols=4)
    train_model(eval_ds)

    

def main():
    
   #loading()
   training()
            
    # folder_name = PurePath(data_yaml_path)
    # data_save_path = os.path.join(os.getcwd(), "datasets/", "tf/", folder_name.parent.name)

    # #train_ds = make_dataset(ds[0])
    # #train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    # #train_ds = train_ds.batch(BATCH_SIZE)
        

def save_dict(dict):
    #pickle_obj = pickle.dumps(dict)
    with mgzip.open('data.dat.gz', 'wb') as file:
        pickle.dump(dict, file)

def load_dict(pickle_file):
    file = None
    with gzip.open(pickle_file, 'rb') as f:
        file = f.read()
    b = pickle.loads(file)
    return b

def format_dataset(inputs):
    dataset = next(iter(inputs.take(1)))
    bounding_boxes = dataset["bounding_boxes"]
    images = dataset["images"]
    bounding_boxes["boxes"] = bounding_boxes["boxes"].to_tensor()
    #bounding_boxes["classes"] = bounding_boxes["classes"].to_tensor()
    print(colored((bounding_boxes["boxes"]), "green"))
    #bounding_boxes["boxes"] = tf.reshape(bounding_boxes["boxes"], [bounding_boxes["boxes"].shape[0] * bounding_boxes["boxes"].shape[1], bounding_boxes["boxes"].shape[2], bounding_boxes["boxes"].shape[3]])
    #images = tf.reshape(images, [images.shape[0] * images.shape[1], images.shape[2], images.shape[3], images.shape[4]])
    #bounding_boxes["classes"]= tf.reshape(bounding_boxes["classes"], [bounding_boxes["classes"].shape[0] * bounding_boxes["classes"].shape[1], bounding_boxes["classes"].shape[2]])
    
    dataset["images"] = images
    dataset["bounding_boxes"] = bounding_boxes

    dataset = make_dataset(dataset)

    return dataset
 
def load_dataset(path):
    dataset = load_dict(path)
    #ds = make_dataset(dataset)
    #train_model(ds)
    #ds = format_dataset(ds)
    return dataset

def train_model(ds):
    #ds =  next(iter(ds.take(1)))
    #print_debug(ds)
    base_lr = 0.005
    # including a global_clipnorm is extremely important in object detection tasks
    model = keras_cv.models.RetinaNet(
        backbone=keras_cv.models.ResNet50Backbone.from_preset("resnet50_imagenet"),
        num_classes=2,
        # num_classes=len(class_mapping),
        # For more info on supported bounding box formats, visit
        # https://keras.io/api/keras_cv/bounding_box/
        bounding_box_format="rel_xywh",)
    
    model.compile(
        classification_loss="focal",
        #loss= keras_cv.losses.GIoULoss("rel_xywh"),
        box_loss="smoothl1",
        optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
        # We will use our custom callback to evaluate COCO metrics
        jit_compile=False)
    
    model.fit(
        ds["images"], ds["bounding_boxes"],
        #validation_data=eval_ds.take(20),
        # Run for 10-35~ epochs to achieve good scores.
        epochs=1,)
    print(colored("finished despite the error","red"))





if __name__ == "__main__":
    main()