import tensorflow as tf
import keras_cv.models
from termcolor import colored

images = tf.ones(shape=(1, 512, 512, 3))
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
print(colored(type(labels), "red"))
print(colored(type(labels["boxes"]), "red"))
print(colored(type(labels["boxes"][0]), "red"))
print(colored(type(labels["boxes"][0][0]), "red"))
print(colored(type(labels["boxes"][0][0][0]), "red"))
print(colored(type(labels["classes"]), "red"))
print(colored(type(labels["classes"][0]), "red"))
print(colored(type(labels["classes"][0][0]), "red"))
#print(colored(type(labels["classes"][0][0][0]), "red"))
print(colored(type(images), "red"))

model = keras_cv.models.YOLOV8Detector(
    num_classes=20,
    bounding_box_format="xywh",
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
