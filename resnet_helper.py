import tensorflow as tf
import numpy as np
from termcolor import colored
from tqdm import tqdm
import glob


def get_labels(data_dir):
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
    for name in tqdm(glob.glob(data_dir + '*.txt')):
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
    return train_labels


def get_data_set(data_dir):
    train_images = get_image_data(data_dir)
    train_labels = get_labels(data_dir)
    #print(colored(train_labels, "red"))
    train_dataset = {
        "images": train_images,
        "boundary_boxes": {
            'boxes': train_labels["boxes"],
            'classes': train_labels["classes"]
        }
    }
    train_labels = train_labels["classes"]
    return train_dataset, train_labels


def get_image_data(data_dir):
    img_height = 180
    img_width = 180
    batch_size = None


    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels=None
    )
    
    train_ds.map(lambda x: (tf.cast(x, tf.float32)))
    train_images = np.zeros((1, img_height, img_width, 3))

    #print(colored(train_ds, "green"))
   
    for image_batch in tqdm(train_ds):
        image_batch = tf.expand_dims(image_batch, axis=0)
        image_batch = image_batch.numpy()
        train_images = np.append(train_images, image_batch, axis=0)

    train_images = tf.convert_to_tensor(train_images, dtype=tf.int32)

    return train_images


if __name__ == '__main__':
    print(colored(get_data_set('/home/tobias/git_ws/pul/train/buffalo/'), 'green'))
