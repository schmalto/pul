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


time_steps = 2000

def normalize(v):
    normalized_v = v/np.linalg.norm(v)
    return normalized_v


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
  #plt.show()
  plt.savefig('test_pred.png')

def generate_time_stamp():
    dates = []
    for i in range(0, time_steps):
        dt = datetime.now() + timedelta(minutes=1)
        timestamp = dt.timestamp() * 0.1
        dates.append(timestamp)
    return pd.Series(dates)

def create_sequences(features, labels, time_steps):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:i+time_steps])
        y.append(labels[i+time_steps])
    return np.array(X), np.array(y)


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

train_features = normalize(train_features)
test_features = normalize(test_features)
train_labels = normalize(train_labels)
test_labels = normalize(test_labels)


# Erstellen der Sequenzen
train_features, train_labels = create_sequences(train_features, train_labels, time_steps)
test_features, test_labels = create_sequences(test_features, test_labels, time_steps)


# sns.pairplot(train_dataset[['time', 'free']], diag_kind='kde')
# plt.show()



model_lstm = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(time_steps, 1)),  # Use LSTM layer for time series modeling
    keras.layers.LSTM(64, input_shape=(time_steps, 1)),  # Use LSTM layer for time series modeling
    keras.layers.LSTM(64, input_shape=(time_steps, 1)),  # Use LSTM layer for time series modeling
    keras.layers.LSTM(64, input_shape=(time_steps, 1)),  # Use LSTM layer for time series modeling
    keras.layers.Dense(1)
])


model_gru = keras.Sequential([
    keras.layers.GRU(64, input_shape=(time_steps, 1)),  # Use LSTM layer for time series modeling
    keras.layers.Dense(1)
])

model_gru_deeper = keras.Sequential([
    keras.layers.GRU(64, input_shape=(time_steps, 1)),  # Use LSTM layer for time series modeling
    keras.layers.GRU(64, input_shape=(time_steps, 1)),  # Use LSTM layer for time series modeling
    keras.layers.GRU(64, input_shape=(time_steps, 1)),  # Use LSTM layer for time series modeling
    keras.layer.Dense(1)
])

model_lstm_gru = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(time_steps, 1), return_sequences=True),  # Use LSTM layer for time series modeling
    keras.layers.GRU(64, input_shape=(time_steps, 1)),  # Use LSTM layer for time series modeling
    keras.layers.Dense(1)
])

model = model_lstm_gru

early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

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
    callbacks=[early_stopping],
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

#plot_loss(history)days

#model = keras.models.load_model('lstm.krs')

model.save('deeper_model.krs')


test_results = {}

test_results['model'] = model.evaluate(
    test_features,
    test_labels, verbose=1)




x = generate_time_stamp()
x = normalize(x)
y = model.predict(x)

with open('test_results.log', 'w') as f:
    print(test_results, file=f)

plot_pred(x,y)
