import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras_cv.models import YOLOV8Detector, YOLOV8Backbone
from resnet_helper import create_dataset

train_path = 'data/train'
valid_path = 'data/valid'

dataset_path = 'data'



train_dataset, train_labels, valid_dataset, valid_labels = create_dataset(dataset_path)


# Set up the model
model = YOLOV8Detector(
    num_classes=5, # Number of object classes
    bounding_box_format='corners', # Format of bounding boxes: 'corners' or 'minmax'
    backbone=YOLOV8Backbone.from_preset(
        "yolo_v8_m_backbone"
    ),
    fpn_depth=2,
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    #box_loss='mse',
    box_loss='iou',
    # box_loss='giou',
    classification_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# Set up training callbacks
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True),
    EarlyStopping(patience=10, restore_best_weights=True),
    TensorBoard(log_dir='./logs'),
]

# Train the model

model.fit(
    train_dataset,
    train_labels,
    epochs=100,
    callbacks=callbacks,
    validation_data=(valid_dataset, valid_labels),

)


