import tensorflow as tf
from typing import Tuple

def LSTM_1(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, embedding_size: int = 32) -> tf.keras.Model:
    """
    Implements a time series prediction model using stacked LSTMs with dropout regularization.

    This function constructs an LSTM model with two stacked LSTM layers followed by dense layers.

    Args:
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        hidden_units (int, optional): The number of hidden units in each LSTM layer. Defaults to 64.
        embedding_size (int, optional): The size of the embedding layer. Defaults to 32.
        output_shape (int): The number of output units in the final dense layer.

    Returns:
        tf.keras.Model: A compiled time series prediction model.
    """

    # Feature input layer
    input_features = tf.keras.layers.Input(shape=input_shape)

    # LSTM layers for input features
    lstm_output = tf.keras.layers.LSTM(hidden_units, return_sequences=True)(input_features)
    lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)

    # Additional LSTM layer
    lstm_output = tf.keras.layers.LSTM(hidden_units)(lstm_output)
    lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)

    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='relu')(lstm_output)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)

    # Create the model
    model = tf.keras.models.Model(inputs=input_features, outputs=output)

    return model
