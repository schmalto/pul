import os
import xml.etree.ElementTree as ET
import numpy as np
from keras.utils import Sequence
# from keras.preprocessing.image import load_img, img_to_array
from keras.utils import load_img, img_to_array
from keras.utils import to_categorical
import glob
import tensorflow as tf
from tqdm import tqdm

class CustomDataGenerator(Sequence):
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

    def getitem__(self):
        label_map = {"trafficlight": 0,
                     "crosswalk": 1,
                     "stop": 2,
                     "speedlimit": 3
                    }

        batch_image_list = self.image_list
        batch_images_np = np.zeros((1,4,4,3))
        batch_boxes = []

        for image_filename in tqdm(batch_image_list[:1]):
            # Load the image
            image = load_img(os.path.join(
                self.image_dir, image_filename), target_size=self.input_shape)
            image = img_to_array(image) / np.float32(255.0)

            # Load the annotations
            annotation_filename = os.path.splitext(image_filename)[0] + '.xml'
            tree = ET.parse(os.path.join(
                self.annotation_dir, annotation_filename))
            root = tree.getroot()

            # Extract the bounding boxes and class labels
            boxes = []
            labels = []
            for obj in root.findall('object'):
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                label = obj.find('name').text
                label = int(label_map[label])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
                print("")

            # Convert the labels to one-hot encoded vectors
            #labels = to_categorical(labels, num_classes=self.num_classes, dtype='int32').tolist()


            image = tf.expand_dims(image, axis=0)
            batch_images_np = np.append(batch_images_np, image, axis=0)
            batch_boxes.append(boxes)
        
        batch_images = batch_images_np[1:]
        batch_images_tf = tf.convert_to_tensor(batch_images, dtype=tf.float32)


        return batch_images_tf, {
            "boxes": tf.convert_to_tensor(batch_boxes),
            "classes": tf.convert_to_tensor(labels)}
