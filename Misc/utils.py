import numpy as np
import pandas as pd
from enum import Enum
import tensorflow as tf
from typing import List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression

################ DATASET ################

def create_time_index(dataset):
    """
    Combines separate date and time columns into a single datetime column in a pandas DataFrame.
    
    Args:
      dataset (pandas.DataFrame): The DataFrame containing the date and time columns.
    
    Returns:
      pandas.DataFrame: The modified DataFrame with the combined datetime column and optional removal of the original columns.
    
    Raises:
      ValueError: If any of the date/time columns are missing or have incorrect data types.
    """
    
    # Check for presence and data type of all required columns
    required_columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
    if not all(col in dataset.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {', '.join(set(required_columns) - set(dataset.columns))}")
    
    # Combine date and time columns into a single datetime column
    dataset['timestamp'] = pd.to_datetime(
        {
            'year': dataset['year'],
            'month': dataset['month'],
            'day': dataset['day'],
            'hour': dataset['hour'],
            'minute': dataset['minute'],
            'second': dataset['second']
        }
    )

    # convert the datetime column to an integer
    dataset['timestamp'] = dataset['timestamp'].astype('int64')

    # divide the resulting integer by the number of nanoseconds in a second
    dataset['timestamp'] = dataset['timestamp'].div(10**9)

    return dataset

def process_data(df, time_index=True, scale_data=False, select_features=False, use_differences=False, prediction_time=None):
    """
    Preprocesses the input DataFrame:
    1. Replaces commas with periods in all values.
    2. Converts all values to float.
    3. Optionally scales the data.
    4. Optionally selects the most relevant features.
    5. Optionally creates a time index.
    
    Args:
        df (DataFrame): Input DataFrame.
        time_index (bool): Whether to create a time index (default False).
        impute_strategy (str): Strategy to use for imputing missing values ('mean', 'median', 'most_frequent').
        scale_data (bool): Whether to scale the data (default False).
        select_features (bool): Whether to select the most relevant features (default False).

    Returns:
        DataFrame: Preprocessed DataFrame.
    """
    # Replace commas with periods
    df = df.replace(',', '.', regex=True)
    
    # Convert all values to float
    df = df.astype(float)

    # Create time index if specified
    if time_index:
        df = create_time_index(df)

    # Remove time columns
    df = df.drop(['year', 'month', 'day', 'hour', 'minute', 'second'], axis=1)
    
    # Compute the difference of the first column respect to the prediction_time ahead
    if use_differences:
        # Assuming the first column is the target column and the rest are features
        target = df.iloc[:, 0]
        # Calculate differencies
        target = target.diff(periods=prediction_time)
        # Remove the first values 
        target = target[prediction_time:]
        # Remove the target column from the DataFrame
        df = df.iloc[:, 1:]
        # Remove the first values from the DataFrame
        df = df[prediction_time:]
        # Re-add the target column to the DataFrame
        df = pd.concat([target, df], axis=1)
        # Remove rows with NaN values
        df = df.dropna()
        # Reset index
        df = df.reset_index(drop=True)
            
    # Scale the data 
    if scale_data:
        scaler = StandardScaler()
        # Assuming the first column is the target column and the rest are features
        target = df.iloc[:, 0]
        # Remove the target column from the DataFrame
        df = df.iloc[:, 1:]
        # Scale the features
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        # Re-add the target column to the DataFrame
        df = pd.concat([target, df], axis=1)
    
    # Select the most relevant features
    if select_features:
        # Assuming the first column is the target column and the rest are features
        features = df.iloc[:, 1:]
        target = df.iloc[:, 0]
        # Compute most significant columns
        selector = SelectKBest(score_func=f_regression, k=min(10, len(features.columns)))
        features_selected = selector.fit_transform(features, target)
        selected_columns = features.columns[selector.get_support()]
        # Create new dataset with selected columns
        df = pd.DataFrame(features_selected, columns=selected_columns)
        df = pd.concat([target, df], axis=1)
        
    return df

def to_sequences(obs, seq_size, prediction_time):
    """
    This function creates batches of sequences and targets for training a model.
    
    Args:
        obs 
        (list): The list of observations (time series data).
        seq_size (int): The size of the sequence window.
        prediction_time (int): The number of steps ahead to predict.
    
    Returns:
        tuple: A tuple containing two NumPy arrays:
          - x: The sequences (training data).
          - y: The targets (predictions).
    """
    x = []
    y = []

    # Loop through observations, ensuring enough data for window and target
    for i in range(seq_size, len(obs) - prediction_time - 1):
        # Get the sequence window using slicing
        window = obs[(i - seq_size):i]
        window = [[x] for x in window]
        
        # Check if window extraction is within bounds
        if len(window) != seq_size:
            raise ValueError("Sequence size exceeds available data at index", i)
        
        # Extract the target value
        after_window = obs[i + prediction_time - 1]       
        
        # Append sequence and target
        x.append(window)
        y.append(after_window)
        
    return np.array(x), np.array(y)

def to_sequences_multi(obs: pd.DataFrame, seq_size: int, prediction_time: int, target_columns: List[str]):
    """
    This function creates batches of sequences and targets for training a model.

    Args:
        obs (list): The list of observations (time series data).
        seq_size (int): The size of the sequence window.
        prediction_time (int): The number of steps ahead to predict.
        target_columns (list): The list of column indices to be considered as prediction values.

    Returns:
        tuple: A tuple containing two NumPy arrays:
          - x: The sequences (training data).
          - y: The targets (predictions).
    """
    # Get feature columns
    feature_columns = list(obs.columns)
    for target_column in target_columns:
        feature_columns.remove(target_column)

    # Check target columns are valid
    if not all(col in obs.columns for col in target_columns):
        raise ValueError("One or more target columns not found in DataFrame")

    # Check if sequence size and prediction time are valid
    if seq_size < 1:
        raise ValueError("Sequence size must be greater than zero")
    if prediction_time < 1:
        raise ValueError("Prediction time must be greater than zero")
        
    x, y = [], []
    # Loop through observations, ensuring enough data for window and target
    for i in range(seq_size, len(obs) - prediction_time - 1):
        # Get the sequence window using slicing
        window = obs.iloc[i - seq_size:i][feature_columns].values

        # Extract the target value
        after_window = obs.iloc[i + prediction_time - 1][target_columns].values

        # # Append sequence and target
        x.append(window)
        y.append(after_window)

    return np.array(x), np.array(y)

def to_target(obs, sequence_columns, target_columns):
    """
    This function creates batches of sequences and targets for training a model.

    Args:
        obs (list): The list of observations (time series data).
        seq_size (int): The size of the sequence window.
        prediction_time (int): The number of steps ahead to predict.
        sequence_columns (list): The list of column indices to be considered as sequence values.
        target_columns (list): The list of column indices to be considered as prediction values.

    Returns:
        tuple: A tuple containing two NumPy arrays:
          - x: The sequences (training data).
          - y: The targets (predictions).
    """
    # Check if provided columns exist in the DataFrame
    if not all(col in obs.columns for col in sequence_columns):
        raise ValueError("One or more sequence columns not found in DataFrame")
    if not all(col in obs.columns for col in target_columns):
        raise ValueError("One or more target columns not found in DataFrame")
     
    x = []
    y = []

    # Loop through observations, ensuring enough data for window and target
    for i in range(len(obs)):
        # Get the sequence window using slicing
        features = obs.iloc[i][sequence_columns].values

        # Extract the target value
        target = obs.iloc[i][target_columns].values

        # # Append sequence and target
        x.append(features)
        y.append(target)

    return np.array(x), np.array(y)

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
