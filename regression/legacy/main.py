from datetime import datetime
import numpy as np
import csv

# Angenommen, wir haben das Modell bereits gespeichert und möchten es jetzt laden
from tensorflow import keras
geladenes_modell = keras.models.load_model('regression.keras')

# Funktion zur Umwandlung des Datums und der Uhrzeit in einen Float-Wert
def datetime_to_float_ms(dt):
    timestamp_ms = dt.timestamp() * 1000.0
    return timestamp_ms

# Datum und Uhrzeit für die Vorhersage am 26.08.2023 um 9:15 Uhr
vorhersage_datum = datetime(2022, 8, 26, 9, 15)
vorhersage_zeitstempel = datetime_to_float_ms(vorhersage_datum)

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
x_werte = np.array(daten)
y_werte = np.array(lorry_free_values)

# Normalisierung der Eingabedaten
x_min = np.min(x_werte)
x_max = np.max(x_werte)

vorhersage_zeitstempel_normalized = (vorhersage_zeitstempel - x_min) / (x_max - x_min)

# Vorhersage mit dem geladenen Modell
vorhersage = geladenes_modell.predict(np.array([vorhersage_zeitstempel_normalized]))

# Die Vorhersage ist ein 2D-Array, wir nehmen den Wert des ersten Elements
vorhersage_lorry_free = vorhersage[0][0]

print("Vorhersage für lorry_free am 26.08.2023 um 9:15 Uhr:", vorhersage_lorry_free)