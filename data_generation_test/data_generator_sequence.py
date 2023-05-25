import os
import xml.etree.ElementTree as ET
import numpy as np
from keras.utils import Sequence
# from keras.preprocessing.image import load_img, img_to_array
from keras.utils import load_img, img_to_array
from keras.utils import to_categorical
import glob
import tensorflow as tf


class CustomDataGeneratorSequence(Sequence):
    '''
    @param image_dir: Directory containing the images
    @param annotation_dir: Directory containing the annotations
    @param batch_size: Batch size
    @param input_shape: Input shape of the model (height, width) tuple
    @param num_classes: Number of object classes
    @param image_list: List of image filenames
    '''

    def __init__(self, image_dir, annotation_dir, batch_size, target_size, num_classes, image_list=" "):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_list = []
        self.batch_size = batch_size
        self.input_shape = target_size
        self.num_classes = num_classes
        for filename in glob.glob(os.path.join(image_dir, '*.png')):
            self.image_list.append(os.path.basename(filename))

    def __len__(self):
        return int(np.ceil(len(self.image_list) / float(self.batch_size)))

    def __getitem__(self, idx):

        label_map = {"trafficlight": 0,
                     "crosswalk": 1,
                     "stop": 2,
                     "speedlimit": 3
                    }

        batch_image_list = self.image_list[idx *
                                           self.batch_size:(idx+1)*self.batch_size]

        batch_images = []
        batch_labels = []
        batch_boxes = []

        for image_filename in batch_image_list:
            # Load the image
            image = load_img(os.path.join(
                self.image_dir, image_filename), target_size=self.input_shape)
            image = img_to_array(image)

            # Load the annotations
            annotation_filename = os.path.splitext(image_filename)[0] + '.xml'
            tree = ET.parse(os.path.join(
                self.annotation_dir, annotation_filename))
            root = tree.getroot()

            # Extract the bounding boxes and class labels
            boxes = []
            labels = []
            all_obj = root.findall('object')
            for obj in root.findall('object'):
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                label = obj.find('name').text
                label = label_map[label]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

            # Convert the labels to one-hot encoded vectors
            labels = to_categorical(labels, num_classes=self.num_classes)

            batch_images.append(image)
            batch_labels.append(labels)
            batch_boxes.append(boxes)
            batch_images_np = np.array(batch_images, dtype='float32') / 255.0
            batch_boxes_np = np.array(batch_boxes, dtype='int32')
            batch_labels_np = np.array(batch_labels, dtype='int32')

            batch_images_tf = tf.convert_to_tensor(batch_images_np, dtype=tf.float32)
            batch_boxes_tf = tf.convert_to_tensor(batch_boxes_np, dtype=tf.int32)
            batch_labels_tf = tf.convert_to_tensor(batch_labels_np, dtype=tf.int32)

        return batch_images_tf, {
            "boxes": batch_boxes_tf,
            "classes": batch_labels_tf}
