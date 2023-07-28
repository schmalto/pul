import csv
from datetime import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

def datetime_from_csv_string(csv_date_string):
    return datetime.strptime(csv_date_string, "%d.%m.%Y %H:%M:%S")

def datetime_to_float_ms(dt):
    # Konvertiere das datetime-Objekt in einen Float mit Millisekunden
    timestamp_ms = dt.timestamp() * 1000.0
    return timestamp_ms

# Laden der CSV-Datei und Umwandlung der Datumsangaben und lorry_free-Werte
daten = []
lorry_free_values = []

with open("bw_data_timeseries.csv", "r") as csvfile:
    csvreader = csv.DictReader(csvfile, delimiter=";")
    for row in csvreader:
        datum_str = row["Uhrzeit"]
        datum = datetime_from_csv_string(datum_str)
        float_value = datetime_to_float_ms(datum)
        daten.append(float_value)

        lorry_free_value = float(row["lorry_free"])
        lorry_free_values.append(lorry_free_value)

# Umwandlung der Daten in numpy-Arrays
x_werte = np.array(daten)
y_werte = np.array(lorry_free_values)

# Normalisierung der Eingabedaten
x_min = np.min(x_werte)
x_max = np.max(x_werte)
x_werte = (x_werte - x_min) / (x_max - x_min)

# Aufteilung der Daten in Trainings-, Validierungs- und Testdaten
x_train, x_temp, y_train, y_temp = train_test_split(x_werte, y_werte, test_size=0.3, random_state=42)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# Erstellen des Keras-Modells mit nur einer Dense-Schicht mit 16 Neuronen und L2-Regularisierung
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), input_shape=(1,)),
    keras.layers.Dense(1)  # Ausgabeneuron ohne Aktivierungsfunktion für Regression
])
print(model.summary())

# Kompilieren des Modells
optimizer = keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# EarlyStopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Visualisieren der Architektur und Speichern als PNG-Datei
#plot_model(model, to_file='modell_architektur.png', show_shapes=True)

# Training des Modells
history = model.fit(x_train, y_train, epochs=10000, batch_size=16, validation_data=(x_valid, y_valid), callbacks=[early_stopping])

# Evaluierung des Modells auf den Testdaten
test_loss = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)

# Plotten des Verlaufs von Trainings- und Validierungsverlust (Loss)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochen')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Speichern des trainierten Modells
model.save('regression.keras')

# Vorhersage für neue Daten (Beispiel: Zeitstempel für 10:31 Uhr)
#neuer_zeitstempel = datetime_to_float_ms(datetime(2023, 7, 26, 10, 31))
#neuer_zeitstempel_normalized = (neuer_zeitstempel - np.min(x_werte)) / (np.max(x_werte) - np.min(x_werte))
#vorhersage = model.predict(np.array([neuer_zeitstempel_normalized]))

#print("Vorhersage für lorry_free um 10:31 Uhr:", vorhersage[0][0])