import tensorflow as tf
from keras_cv.visualization import plot_image_gallery
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from data_generator import CustomDataGenerator
from termcolor import colored
import keras_cv

image_dir = "example_set/images"
annotation_dir = "example_set/annotations"
train_image_list = ""
batch_size = 8
target_size = (4, 4)
num_classes = 4



train_generator = CustomDataGenerator(image_dir, annotation_dir, batch_size, target_size, num_classes, train_image_list)
train_data = train_generator.getitem__()


labels = train_data[1]
images = train_data[0]

images = tf.ones(shape=(1, 4, 4, 3))
labels = {
    "boxes": [
        [
            [0, 0, 100, 100],
            [100, 100, 200, 200],
            [300, 300, 100, 100],
        ]
    ],
    "classes": [[1, 1, 1]],
}

# print(colored(labels, "red"))
# print(colored(images, "blue"))



model = keras_cv.models.YOLOV8Detector(
    num_classes=20,
    bounding_box_format="xyxy",
    backbone=keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_s_backbone"
    ),
    fpn_depth=2
)

# Evaluate model
model(images)

# Get predictions using the model
model.predict(images)

# Train model
model.compile(
    classification_loss='binary_crossentropy',
    box_loss='iou',
    optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
    jit_compile=False,
)
model.fit(images, labels)
