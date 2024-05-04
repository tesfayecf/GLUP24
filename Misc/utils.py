import json
import numpy as np
import pandas as pd
from enum import Enum
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_parameters(file_path):
    with open(file_path, 'r') as file:
        parameters = json.load(file)
    return parameters

################ DATASET ################

def preprocess_data(df, create_timestamp=True, create_fatigue=True, scale_data=True, perform_pca=True, pca_components=8):
    """
    Processa el DataFrame d'entrada:
    
    Args:
        df (DataFrame): DataFrame d'entrada.
        create_timestamp (bool): Si cal crear un índex de temps (per defecte, True).
        create_fatigue (bool): Si cal crear la fatiga (per defecte, True).
        scale_data (bool): Si cal escalar les dades (per defecte, True).
        perform_pca (bool): Si cal realitzar PCA (per defecte, True).
        pca_components (int): Nombre de components per PCA (per defecte, 8).

    Returns:
        DataFrame: DataFrame processat.
    """
    # Remplacem les commes per punts
    df = df.replace(',', '.', regex=True)
    # Convertim tots els valors a float
    df = df.astype(float)
    
    # Creem la columna timestamp
    time_columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
    if create_timestamp:
        # Comprovem que les columnes necessaries existeixin
        if not all(col in df.columns for col in time_columns):
            raise ValueError(f"Missing required columns: {', '.join(set(time_columns) - set(df.columns))}")
        # Convertim les columnes temporals a format datetime
        df['timestamp'] = pd.to_datetime(
            {
                'year': df['year'],
                'month': df['month'],
                'day': df['day'],
                'hour': df['hour'],
                'minute': df['minute'],
                'second': df['second']
            }
        )
        df['timestamp'] = df['timestamp'].astype('int64').div(10**9)
        df.drop(columns=time_columns, inplace=True)

    # Creem la columna fatigue
    fatige_columns = ['work', 'exercise', 'sleep', 'stressors', 'illness']
    if create_fatigue:
        # Comprovem que les columnes necessaries existeixin
        if not all(col in df.columns for col in fatige_columns):
            raise ValueError(f"Missing required columns: {', '.join(set(fatige_columns) - set(df.columns))}")
        # Creem la nova variable
        df["fatigue"] = df["work"] + df["exercise"] - df["sleep"] + df["illness"] + df["stressors"]
        df.drop(columns=fatige_columns, inplace=True)
    
    # Escalem les dades
    if scale_data:
        target = df['glucose_level']
        features = df.drop(columns=['glucose_level'])
        features_scaled = StandardScaler().fit_transform(features)
        df = pd.concat([pd.DataFrame(features_scaled), target], axis=1)
    
    if perform_pca:
        target = df['glucose_level']
        features = df.drop(columns=['glucose_level'])
        pca = PCA(n_components=pca_components)
        features_pca = pca.fit_transform(features)
        df = pd.concat([pd.DataFrame(features_pca), target], axis=1)
        
    return df

def to_sequences(dataset, sequence_size: int, prediction_time: int, target_column: str):
    """
    Aquesta funció crea lots de seqüències i objectius per entrenar un model.

    Args:
        data (DataFrame): El conjunt de dades (sèries temporals).
        sequence_size (int): La mida de la finestra de seqüències.
        prediction_time (int): El nombre de passos a predir endavant.
        target_column (str): El nom de la columna considerada com a valor objectiu.

    Returns:
        tuple: Una tupla que conté dues matrius NumPy:
          - x: Les seqüències (dades d'entrenament).
          - y: Els objectius (prediccions).
    """
    # Comprova si les columnes proporcionades existeixen al DataFrame
    if not target_column in dataset.columns:
        raise ValueError("Target column not found in DataFrame")
    # Comprova si la mida de la seqüència i el temps de prediccio són vàlids
    if sequence_size < 1:
        raise ValueError("Sequence size must be greater than zero")
    if prediction_time < 1:
        raise ValueError("Prediction time must be greater than zero")

    x, y = [], []
    # Recorre les observacions, assegurant prou dades per la finestra i l'objectiu
    for i in range(sequence_size, len(dataset) - prediction_time - 1):
        # Obté la finestra de seqüències mitjançant l'ús de l'indexació
        window = dataset.iloc[i - sequence_size:i]
        # Extreu el valor objectiu
        after_window = dataset.iloc[i + prediction_time - 1][target_column]
         # Afegeix la seqüència i l'objectiu
        x.append(window)
        y.append(after_window)
    return np.array(x), np.expand_dims(np.array(y), axis=1)

################# MODEL #################

class Optimizer(str, Enum):
    Adam = 'Adam'
    SGD = 'SGD'
    RMSprop = 'RMSprop'
    Adagrad = 'Adagrad'
    Adadelta = 'Adadelta'

def get_optimizer(optimizer: Optimizer, learning_rate: float):
    if optimizer == Optimizer.Adam:
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == Optimizer.SGD:
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == Optimizer.RMSprop:
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == Optimizer.Adagrad:
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == Optimizer.Adadelta:
        return tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")

class Loss(str, Enum):
    mse = 'mse'
    mae = 'mae'
    mape = 'mape'
    msle = 'msle'
    hinge = 'hinge'

def get_loss(loss: Loss):
    if loss == Loss.mse:
        return tf.keras.losses.MeanSquaredError()
    elif loss == Loss.mae:
        return tf.keras.losses.MeanAbsoluteError()
    elif loss == Loss.mape:
        return tf.keras.losses.MeanAbsolutePercentageError()
    elif loss == Loss.msle:
        return tf.keras.losses.MeanSquaredLogarithmicError()
    elif loss == Loss.hinge:
        return tf.keras.losses.Hinge()
    else:
        raise ValueError(f"Invalid loss function: {loss}")
