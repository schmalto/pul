from datetime import datetime, timedelta
import numpy as np
import csv
import matplotlib.pyplot as plt
from tensorflow import keras

# Funktion zur Umwandlung des Datums und der Uhrzeit in einen Float-Wert
def datetime_to_float_ms(dt):
    timestamp_ms = dt.timestamp() * 1000.0
    return timestamp_ms

# Datum und Uhrzeit für den Start der Vorhersagen am 01.08.2023 um 09:32 Uhr
start_datum = datetime(2023, 8, 1, 9, 32)

def datetime_from_csv_string(csv_date_string):
    return datetime.strptime(csv_date_string, "%d.%m.%Y %H:%M")

def datetime_to_float_ms(dt):
    # Konvertiere das datetime-Objekt in einen Float mit Millisekunden
    timestamp_ms = dt.timestamp() * 1000.0
    return timestamp_ms

# Laden der CSV-Datei und Umwandlung der Datumsangaben und lorry_free-Werte
daten = []
lorry_free_values = []

with open("x_gfild.CSV", "r") as csvfile:
    csvreader = csv.DictReader(csvfile, delimiter=";")
    for row in csvreader:
        datum_str = row["Uhrzeit"]
        datum = datetime_from_csv_string(datum_str)
        float_value = datetime_to_float_ms(datum)
        daten.append(float_value)

        lorry_free_value = float(row["lorry_free"])
        lorry_free_values.append(lorry_free_value)

# Normalisierung des Zeitstempels basierend auf dem Trainingsdatensatz
# Umwandlung der Daten in numpy-Arrays
#x_werte = np.array(daten)
#y_werte = np.array(lorry_free_values)

from loader import load_csv_data_regression

dataset = load_csv_data_regression()

x_werte = np.array(dataset["time"])
y_werte = np.array(dataset["free"])


# Normalisierung der Eingabedaten
x_min = np.min(x_werte)
x_max = np.max(x_werte)
x_werte_normalized = (x_werte - x_min) / (x_max - x_min)

# Umwandlung in Sequenzen für das RNN
sequence_length = 10  # Anzahl der vergangenen Zeitpunkte, die das RNN betrachtet
sequences = []
labels = []

for i in range(len(x_werte_normalized) - sequence_length):
    sequence = x_werte_normalized[i:i+sequence_length]
    label = y_werte[i+sequence_length]
    sequences.append(sequence)
    labels.append(label)

sequences = np.array(sequences)
labels = np.array(labels)

# Umwandlung in die richtige Form (None, 10, 1)
sequences = np.expand_dims(sequences, axis=2)

print("Sequenzen:", sequences.shape)
print("Labels:", labels.shape)

# Verbesserte RNN-Architektur mit LSTM und tanh-Aktivierungsfunktion
model = keras.Sequential([
    keras.layers.LSTM(64, activation='tanh', input_shape=(sequence_length, 1), return_sequences=True),
    keras.layers.LSTM(32, activation='tanh'),
    keras.layers.Dense(1)  # Ausgabeneuron ohne Aktivierungsfunktion für Regression
])

model.summary()
# Kompilieren des Modells
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)


# Training des Modells
#model.fit(sequences, labels, epochs=50, batch_size=32, callbacks=[early_stopping])

model.fit(sequences, labels, epochs=50, batch_size=4, callbacks=[early_stopping])

model.save('lstm.krs')
# Vorhersagen für die Woche
vorhersage_lorry_free_values = []

for i in range(7):
    vorhersage_zeit = start_datum + timedelta(days=i)
    vorhersage_zeitstempel = datetime_to_float_ms(vorhersage_zeit)
    vorhersage_zeitstempel_normalized = (vorhersage_zeitstempel - x_min) / (x_max - x_min)

    # Vorhersage mit dem trainierten Modell
    sequence = x_werte_normalized[-sequence_length:]  # Verwende die letzten Werte als Eingabe
    sequence = np.append(sequence, vorhersage_zeitstempel_normalized)  # Füge den aktuellen Vorhersagezeitpunkt hinzu
    sequence = sequence[-sequence_length:]  # Begrenze die Sequenzlänge auf sequence_length
    sequence = np.expand_dims(sequence, axis=0)  # Füge eine zusätzliche Dimension hinzu, um die Batch-Dimension zu simulieren
    vorhersage = model.predict(sequence)

    # Die Vorhersage ist ein 2D-Array, wir nehmen den Wert des ersten Elements
    vorhersage_lorry_free = vorhersage[0][0]

    vorhersage_lorry_free_values.append(vorhersage_lorry_free)
    print(f"Vorhersage für lorry_free am {vorhersage_zeit.strftime('%d.%m.%Y %H:%M')} Uhr:", vorhersage_lorry_free)

# Plotten der Vorhersageergebnisse für die Woche
vorhersage_zeiten = [start_datum + timedelta(days=i) for i in range(7)]
plt.figure(figsize=(12, 6))
plt.plot(vorhersage_zeiten, vorhersage_lorry_free_values, label="Vorhersage lorry_free")
plt.xlabel("Datum")
plt.ylabel("lorry_free")
plt.title("Vorhersage für lorry_free ab 01.08.2023 um 09:32 Uhr für eine Woche")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

# plot speichern
plt.savefig('vorhersage_lorry_free_week.png', dpi=300)

plt.show()
