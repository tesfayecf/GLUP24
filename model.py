import tensorflow as tf
from typing import Tuple

from LSTM import LSTM_1
from GRU import GRU_1
from CNN import CNN_1
from DR import DR_1

def get_model(model_id: str, input_shape: Tuple[int, ...], output_shape: int, **model_parameters) -> tf.keras.Model:
    """
    Constructs a time series prediction model based on the specified architecture.

    This function serves as a factory, dispatching model creation based on the provided name.

    Args:
        model_id (str): The name of the model to build.
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        output_shape (int): The number of output units in the final dense layer.
        kwargs (dict, optional): Additional keyword arguments specific to the chosen model architecture.

    Returns:
        tf.keras.Model: A compiled time series prediction model.

    Raises:
        ValueError: If an invalid model name is provided.
    """

    if model_id == "LSTM_1":
        return LSTM_1(input_shape, output_shape, **model_parameters)
    elif model_id == "GRU_1":
        return GRU_1(input_shape, output_shape, **model_parameters)
    elif model_id == "CNN_1":
        return CNN_1(input_shape, output_shape, **model_parameters)
    elif model_id == "DR_1":
        return DR_1(input_shape, output_shape, **model_parameters)
    else:
        raise ValueError(f"Invalid model id: {model_id}")

# def time2vec(timesteps: int, embedding_dim: int) -> tf.Tensor:
#   """
#   Implements Time2Vec positional encoding for a given sequence length.

#   This function generates positional encodings for a sequence based on the Time2Vec approach.

#   Args:
#       timesteps (int): The number of timesteps in the sequence.
#       embedding_dim (int): The dimensionality of the positional encoding.

#   Returns:
#       tf.Tensor: A tensor of shape (timesteps, embedding_dim) containing the positional encodings.
#   """
#   position = tf.range(timesteps, dtype=tf.float32)[:, None]
#   div_term = tf.math.exp(tf.range(0, embedding_dim, 2, dtype=tf.float32) * (-math.log(10000.0) / embedding_dim))
#   angle_rates = position * div_term
#   embeddings = tf.concat([tf.sin(angle_rates), tf.cos(angle_rates)], axis=-1)
#   return embeddings

# def Transformer_1(input_shape: Tuple[int, ...], output_shape: int = 1, d_model: int = 512, num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6) -> tf.keras.Model:
#   """
#   Constructs a basic Transformer model with Time2Vec positional encoding for time series forecasting.

#   This is a simplified implementation focusing on core concepts.

#   Args:
#       input_shape (Tuple[int, ...]): The expected input shape for the model.
#       d_model (int): The dimensionality of the model's embedding space.
#       num_heads (int): The number of attention heads in the multi-head attention layer.
#       num_encoder_layers (int): The number of encoder layers in the Transformer.
#       num_decoder_layers (int): The number of decoder layers in the Transformer.
#       output_shape (int): The number of output units in the final layer.

#   Returns:
#       tf.keras.Model: A compiled Transformer model for time series forecasting.
#   """

#   # Define Transformer layers
#   encoder_input = tf.keras.layers.Input(shape=input_shape)
#   decoder_input = tf.keras.layers.Input(shape=(1, d_model))  # Single value for decoder start

#   # Time2Vec positional encoding for encoder input
#   time_embeddings = time2vec(input_shape[0], d_model)  # Assuming time is the first dimension
#   encoded_input = encoder_input * time_embeddings

#   encoder_layers = [
#       tf.keras.layers.TransformerEncoder(d_model=d_model, num_heads=num_heads)
#       for _ in range(num_encoder_layers)
#   ]
#   for layer in encoder_layers:
#     encoder_output = layer(encoded_input)

#   decoder_layers = [
#       tf.keras.layers.TransformerDecoder(d_model=d_model, num_heads=num_heads)
#       for _ in range(num_decoder_layers)
#   ]
#   for layer in decoder_layers:
#     decoder_output = layer(decoder_input, encoder_output)

#   # Final dense layer for prediction
#   output = tf.keras.layers.Dense(output_shape)(decoder_output)

#   # Create and compile the model (adjust optimizer and loss as needed)
#   model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=output)

#   return model

