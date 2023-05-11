import tensorflow as tf
from tensorflow import keras
from keras_cv.visualization import plot_image_gallery
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras_cv.models import YOLOV8Detector, YOLOV8Backbone
from resnet_helper import get_data_set
import keras_cv


train_path = '/home/tobias/git_ws/pul/val/buffalo/'
valid_path = '/home/tobias/git_ws/pul/val/buffalo/'


train_dataset, train_labels = get_data_set(train_path)
valid_dataset, valid_labels = get_data_set(valid_path)

plot_image_gallery(train_dataset, value_range=(0, 255))


resized = keras_cv.layers.Resizing(180, 180, bounding_box_format="xywh", pad_to_aspect_ratio=True)(train_dataset)



# Set up the model
model = YOLOV8Detector(
    num_classes=20, # Number of object classes
    bounding_box_format='xywh', # Format of bounding boxes: 'corners' or 'minmax'
    backbone=YOLOV8Backbone.from_preset(
        #"yolo_v8_m_backbone"
        "yolo_v8_m_backbone_coco"
    ),
    fpn_depth=2,
)

# Compile the model
model.compile(
    #optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
    #box_loss='mse',
    box_loss='iou',
    # box_loss='giou',
    classification_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    jit_compile=False,
)

# Set up training callbacks
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True),
    EarlyStopping(patience=10, restore_best_weights=True),
    TensorBoard(log_dir='./logs'),
]

# Train the model

model.fit(
    (train_dataset,
    train_labels),
    # epochs=100,
    # callbacks=callbacks,
    # validation_data=(valid_dataset, valid_labels),
)


