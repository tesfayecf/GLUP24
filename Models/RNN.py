import tensorflow as tf
from typing import Tuple

def RNN_1(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, dropout: float = 0.25) -> tf.keras.Model:
    """
    Implements an advanced RNN model for time series prediction with stacked layers, 
    dropout, BatchNormalization, and optional return sequences.

    This function constructs an RNN model with two stacked RNN layers with dropout 
    regularization, BatchNormalization, and the option to return sequences.

    Args:
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        output_shape (int, optional): The number of output units in the final dense layer. Defaults to 1.
        hidden_units (int, optional): The number of hidden units in the RNN layers. Defaults to 32.
        return_sequences (bool, optional): Whether to return the full sequence from the RNN layer (True) 
                                            or just the final output (False). Defaults to False.

    Returns:
        tf.keras.Model: A compiled time series prediction model.
    """
    # Feature input layer
    input_features = tf.keras.layers.Input(shape=input_shape)

    # Stacked RNN layers with dropout and BatchNormalization
    rnn_output = tf.keras.layers.SimpleRNN(hidden_units, return_sequences=True)(input_features)
    rnn_output = tf.keras.layers.Dropout(dropout)(rnn_output)
    rnn_output = tf.keras.layers.BatchNormalization()(rnn_output)

    rnn_output = tf.keras.layers.SimpleRNN(hidden_units)(rnn_output)
    rnn_output = tf.keras.layers.Dropout(dropout)(rnn_output)
    rnn_output = tf.keras.layers.BatchNormalization()(rnn_output)

    # Dense layer
    output = tf.keras.layers.Dense(output_shape, activation='linear')(rnn_output)

    # Create the model
    model = tf.keras.models.Model(inputs=input_features, outputs=output)

    return model
