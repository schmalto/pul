import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from loader import load_csv_data_regression
from termcolor import colored
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow import keras

from datetime import datetime, timedelta



def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.show()

def plot_pred(x, y):
  #plt.scatter(train_features, train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('time')
  plt.ylabel('free')
  plt.legend()
  plt.show()

def generate_time_stamp():
    dates = []
    for i in range(0, 1000):
        dt = datetime.now() + timedelta(hours=9)
        timestamp = dt.timestamp() * 0.1
        dates.append(timestamp)
    return pd.Series(dates)


# Laden der Daten
dataset = load_csv_data_regression()

# Aufteilung in Trainings- und Testdaten
train_dataset = dataset.sample(frac=0.8, random_state=42)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('free')
test_labels = test_features.pop('free')

train_features = train_features['time']
test_features = test_features['time']



# sns.pairplot(train_dataset[['time', 'free']], diag_kind='kde')
# plt.show()

normalizer = keras.layers.Normalization(input_shape=[1,], axis=None)
normalizer.adapt(np.array(train_features))

model = keras.Sequential([
    normalizer,
    keras.layers.Dense(64, activation='selu'),
    keras.layers.Dense(64, activation='selu'),
    keras.layers.Dense(1)
])

model.summary()

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = model.fit(
    train_features,
    train_labels,
    epochs=10,
    # Suppress logging.
    verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

#plot_loss(history)days

test_results = {}

test_results['model'] = model.evaluate(
    test_features,
    test_labels, verbose=1)




x = generate_time_stamp()
y = model.predict(x)



plot_pred(x,y)
