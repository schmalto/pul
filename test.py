import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import tensorflow_datasets
import pathlib
from termcolor import colored
from tqdm import tqdm
import glob





def works():
    img = Image.open('301.jpg')
    img_to_tensor = tf.convert_to_tensor(img)
    arr_ = np.squeeze(img_to_tensor.numpy())
    plt.imshow(arr_)
    plt.show()

def labels():
    # train_labels = {
    #     "boxes": [
    #         [
    #             ['x_min', 'y_min', 'x_max', 'y_max'],
    #             [0, 0, 0, 0],
    #             [0, 0, 0, 0]
    #         ],
    #     ],
    #     "classes": [
    #         [1,1,1],

    #     ]
    # }
    train_labels = {
        "boxes": [
        ],
        "classes": [
            
        ]
    }
    for name in glob.glob('/home/tobias/git_ws/pul/train/buffalo/*.txt'):
        with open(name) as f:
            lines = f.readlines()
            classes_in_file =[]
            boxes_in_file = []
            for line in lines:
                line = line.split()
                class_name = line[0]
                x_min = line[1]
                x_max = line[2]
                y_min = line[3]
                y_max = line[4]
                classes_in_file.append(class_name)
                boxes_in_file.append([x_min, y_min, x_max, y_max])
            train_labels["classes"].append(classes_in_file)
            train_labels["boxes"].append(boxes_in_file)
    print(train_labels)


def to_test():
    img_height = 180
    img_width = 180
    batch_size = None
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
    data_dir = pathlib.Path(archive).with_suffix('')

    train_labels = {}
    valid_labels = {}

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,

    )
    
    train_ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,)
    
    val_ds.map(lambda x, y: (tf.cast(x, tf.float32), y))

    train_images = np.zeros((1, 180, 180, 3))
    valid_images = np.zeros((1, 180, 180, 3))
   
    for image_batch, labels_batch in tqdm(train_ds):
        image_batch = tf.expand_dims(image_batch, axis=0)
        image_batch = image_batch.numpy()
        train_images = np.append(train_images, image_batch, axis=0)

    for image_batch, labels_batch in tqdm(val_ds):
        image_batch = tf.expand_dims(image_batch, axis=0)
        image_batch = image_batch.numpy()
        valid_images = np.append(valid_images, image_batch, axis=0)

    train_images = tf.convert_to_tensor(train_images, dtype=tf.int32)
    valid_images = tf.convert_to_tensor(valid_images, dtype=tf.int32)

    return train_images, train_labels, valid_images, valid_labels

if __name__ == '__main__':
    labels()
