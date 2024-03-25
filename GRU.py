import tensorflow as tf
from typing import Tuple

MODEL_NAME = "GRU"
MODEL_VERSION = 1

################# PARAMETERS #################
hidden_units: int = 32
embedding_size: int = 32

def GRU_1(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, embedding_size: int = 32) -> tf.keras.Model:
    """
    Implements a time series prediction model using stacked GRUs with dropout regularization.

    This function constructs an GRU model with two stacked GRU layers followed by dense layers.

    Args:
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        hidden_units (int, optional): The number of hidden units in each GRU layer. Defaults to 64.
        embedding_size (int, optional): The size of the embedding layer. Defaults to 32.
        output_shape (int): The number of output units in the final dense layer.

    Returns:
        tf.keras.Model: A compiled time series prediction model.
    """

    # Feature input layer
    input_features = tf.keras.layers.Input(shape=input_shape)

    # GRU layers for input features
    gru_output = tf.keras.layers.GRU(hidden_units, return_sequences=True)(input_features)
    gru_output = tf.keras.layers.Dropout(0.3)(gru_output)
    
    # Additional GRU layer
    gru_output = tf.keras.layers.GRU(hidden_units)(gru_output)
    gru_output = tf.keras.layers.Dropout(0.3)(gru_output)
    
    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='relu')(gru_output)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)
    
    # Create the model
    model = tf.keras.models.Model(inputs=input_features, outputs=output)
    
    return model
