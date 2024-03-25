import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression

################ DATASET ################

def create_time_index(dataset, time_index):
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
    dataset['time'] = pd.to_datetime(
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
    dataset['timestamp'] = dataset['time'].astype('int64')

    # divide the resulting integer by the number of nanoseconds in a second
    dataset['timestamp'] = dataset['timestamp'].div(10**9)

    # Set the combined datetime column as the index
    if time_index:
        dataset.set_index('timestamp', inplace=True)
    
    # Optionally remove the original columns if desired
    dataset.drop(columns=required_columns + ['time'], inplace=True)
    
    return dataset

def process_data(df, time_index=True, impute_strategy=False, scale_data=False, encode_categorical=False, select_features=False):
    """
    Preprocesses the input DataFrame:
    1. Replaces commas with periods in all values.
    2. Converts all values to float.
    3. Optionally imputes missing values.
    4. Optionally scales the data.
    5. Optionally encodes categorical variables.
    6. Optionally selects the most relevant features.
    7. Optionally creates a time index.
    
    Args:
        df (DataFrame): Input DataFrame.
        time_index (bool): Whether to create a time index (default True).
        impute_strategy (str): Strategy to use for imputing missing values ('mean', 'median', 'most_frequent').
        scale_data (bool): Whether to scale the data (default True).
        encode_categorical (bool): Whether to encode categorical variables (default True).
        select_features (bool): Whether to select the most relevant features (default True).

    Returns:
        DataFrame: Preprocessed DataFrame.
    """
    # Replace commas with periods
    df = df.replace(',', '.', regex=True)
    
    # Convert all values to float
    df = df.astype(float)

    # Create time index if specified
    df = create_time_index(df, time_index)  # Assuming create_time_index function is defined
    
    # Impute missing values
    if impute_strategy:
        imputer = SimpleImputer(strategy=impute_strategy)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Scale the data
    if scale_data:
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])
    
    # Encode categorical variables
    if encode_categorical:
        # Assuming the last column is the target column and the rest are features
        features = df.iloc[:, :-1]
        target = df.iloc[:, -1]
        encoder = LabelEncoder()
        features = features.apply(encoder.fit_transform)
        df = pd.concat([features, target], axis=1)
    
    # Select the most relevant features
    if select_features:
        # Assuming the last column is the target column and the rest are features
        features = df.iloc[:, :-1]
        target = df.iloc[:, -1]
        selector = SelectKBest(score_func=f_regression, k=min(10, len(features.columns)))
        features_selected = selector.fit_transform(features, target)
        selected_columns = features.columns[selector.get_support()]
        df = pd.DataFrame(features_selected, columns=selected_columns)
        df = pd.concat([df, target], axis=1)
        
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
        # Get the sequence window using slicing (clarified using obs[(i-seq_size):i])
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

def to_sequences_multi(obs, seq_size, prediction_time, sequence_columns, target_columns):
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

    # Check if sequence size and prediction time are valid
    if seq_size < 1:
        raise ValueError("Sequence size must be greater than zero")
    if prediction_time < 1:
        raise ValueError("Prediction time must be greater than zero")
        
    x = []
    y = []

    # Loop through observations, ensuring enough data for window and target
    for i in range(seq_size, len(obs) - prediction_time - 1):
        # Get the sequence window using slicing
        window = obs.iloc[i - seq_size:i][sequence_columns].values

        # Extract the target value
        after_window = obs.iloc[i + prediction_time - 1][target_columns].values

        # # Append sequence and target
        x.append(window)
        y.append(after_window)

    return np.array(x), np.array(y)

################# MODEL #################

def get_optimizer(optimizer, learning_rate):
    if optimizer == 'Adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'Adagrad':
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == 'Adadelta':
        return tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")

def get_loss(loss):
    if loss == 'mse':
        return tf.keras.losses.MeanSquaredError()
    elif loss == 'mae':
        return tf.keras.losses.MeanAbsoluteError()
    elif loss == 'rmse':
        return tf.keras.losses.RootMeanSquaredError()
    elif loss == 'mape':
        return tf.keras.losses.MeanAbsolutePercentageError()
    elif loss == 'msle':
        return tf.keras.losses.MeanSquaredLogarithmicError()
    elif loss == 'hinge':
        return tf.keras.losses.Hinge()
    else:
        raise ValueError(f"Invalid loss function: {loss}")