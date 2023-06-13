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
import tqdm


# image = [height, width, 3]
# bounding_boxes = {
#   "classes": [0], # 0 is an arbitrary class ID representing "cat"
#   "boxes": [[0.25, 0.4, .15, .1]]
#    # bounding box is in "rel_xywh" format
#    # so 0.25 represents the start of the bounding box 25% of
#    # the way across the image.
#    # The .15 represents that the width is 15% of the image width.
# }



filepath = tf.keras.utils.get_file(origin="https://i.imgur.com/gCNcJJI.jpg")
image = keras.utils.load_img(filepath)
image = np.array(image)

visualization.plot_image_gallery(
    [image],
    value_range=(0, 255),
    rows=1,
    cols=1,
    scale=5,
)

