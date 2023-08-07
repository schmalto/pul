from datetime import datetime
import numpy as np
import csv
import pandas as pd
import random



random.seed(42)
# Datum und Uhrzeit für den Start der Vorhersagen am 01.08.2023 um 09:32 Uhr
start_datum = datetime(2023, 8, 1, 9, 32)

def datetime_from_csv_string(csv_date_string):
    return datetime.strptime(csv_date_string, "%d.%m.%Y %H:%M")

def datetime_to_float_ms(dt):
    # Konvertiere das datetime-Objekt in einen Float mit Millisekunden
    timestamp_ms = dt.timestamp() * 0.1
    return timestamp_ms

# Laden der CSV-Datei und Umwandlung der Datumsangaben und lorry_free-Werte
'''
def load_csv_data_lstm(filepath="x_gfild.CSV"):
    daten = []
    lorry_free_values = []

    with open(filepath, "r") as csvfile:
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

    return sequences, labels, x_min, x_max
'''
def load_csv_data_regression(filepath="x_gfild.CSV"):
    daten = []
    lorry_free_values = []

    with open(filepath, "r") as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=";")
        for row in csvreader:
            datum_str = row["Uhrzeit"]
            datum = datetime_from_csv_string(datum_str)
            float_value = datetime_to_float_ms(datum)
            daten.append(float_value)
        
            lorry_free_value = float(row["lorry_free"])
                
            lorry_free_values.append(lorry_free_value)

    indices = [i for i, e in enumerate(lorry_free_values) if e == -1]

    lorry_free_values = np.array(lorry_free_values)
    daten = np.array(daten)

    
  
    # Umwandlung der Daten in numpy-Arrays
    x_werte = daten
    y_werte = lorry_free_values

    # Normalisierung der Eingabedaten
    #x_min = np.min(x_werte)
    #x_max = np.max(x_werte)
    #x_werte = (x_werte - x_min) / (x_max - x_min)

    df = pd.DataFrame(list(zip(x_werte, y_werte)), columns =['time', 'free'])
    df.drop(df[df.free > 28].index, inplace=True)
    return df
