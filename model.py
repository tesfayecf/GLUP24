import math
import tensorflow as tf
from typing import Tuple


def get_model(model_version: str, input_shape: Tuple[int, ...], output_shape: int, *model_parameters) -> tf.keras.Model:
    """
    Constructs a time series prediction model based on the specified architecture.

    This function serves as a factory, dispatching model creation based on the provided name.

    Args:
        model_version (str): The name of the model to build.
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        output_shape (int): The number of output units in the final dense layer.
        kwargs (dict, optional): Additional keyword arguments specific to the chosen model architecture.

    Returns:
        tf.keras.Model: A compiled time series prediction model.

    Raises:
        ValueError: If an invalid model name is provided.
    """

    if model_version == "LSTM_1":
        return LSTM_1(input_shape, output_shape, *model_parameters)
    elif model_version == "CNN_1":
        return CNN_1(input_shape, output_shape, *model_parameters)
    elif model_version == "Transformer_1":
        return Transformer_1(input_shape, output_shape, *model_parameters)
    else:
        raise ValueError(f"Invalid model name: {model_version}")

def LSTM_1(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, embedding_size: int = 32, ) -> tf.keras.Model:
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

def CNN_1(input_shape: Tuple[int, ...], output_shape: int = 1, filters: list[int] = [32, 64, 128], kernel_size: int = 3, hidden_units: int = 128) -> tf.keras.Model:
  """
  Constructs a 1D CNN model for time series forecasting.

  This model utilizes convolutional layers to extract local features from the time series data.

  Args:
      input_shape (Tuple[int, ...]): The expected input shape for the model.
      filters (list[int]): A list specifying the number of filters in each convolutional layer.
      kernel_size (int): The size of the convolutional kernel.
      hidden_units (int): The number of units in the hidden dense layer.
      output_shape (int): The number of output units in the final layer.

  Returns:
      tf.keras.Model: A compiled CNN model for time series forecasting.
  """

  # Feature input layer
  input_features = tf.keras.layers.Input(shape=input_shape)

  # Stacked convolutional layers with ReLU activation and batch normalization
  for num_filters in filters:
    conv_output = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')(input_features)
    conv_output = tf.keras.layers.BatchNormalization()(conv_output)
    input_features = conv_output

  # Flatten the output of convolutional layers
  flat_features = tf.keras.layers.Flatten()(conv_output)

  # Hidden dense layer with dropout
  dense_output = tf.keras.layers.Dense(hidden_units, activation='relu')(flat_features)
  dense_output = tf.keras.layers.Dropout(0.2)(dense_output)

  # Output layer
  output = tf.keras.layers.Dense(output_shape)(dense_output)

  # Create and compile the model (adjust optimizer and loss as needed)
  model = tf.keras.models.Model(inputs=input_features, outputs=output)

  return model

def time2vec(timesteps: int, embedding_dim: int) -> tf.Tensor:
  """
  Implements Time2Vec positional encoding for a given sequence length.

  This function generates positional encodings for a sequence based on the Time2Vec approach.

  Args:
      timesteps (int): The number of timesteps in the sequence.
      embedding_dim (int): The dimensionality of the positional encoding.

  Returns:
      tf.Tensor: A tensor of shape (timesteps, embedding_dim) containing the positional encodings.
  """
  position = tf.range(timesteps, dtype=tf.float32)[:, None]
  div_term = tf.math.exp(tf.range(0, embedding_dim, 2, dtype=tf.float32) * (-math.log(10000.0) / embedding_dim))
  angle_rates = position * div_term
  embeddings = tf.concat([tf.sin(angle_rates), tf.cos(angle_rates)], axis=-1)
  return embeddings

def Transformer_1(input_shape: Tuple[int, ...], output_shape: int = 1, d_model: int = 512, num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6) -> tf.keras.Model:
  """
  Constructs a basic Transformer model with Time2Vec positional encoding for time series forecasting.

  This is a simplified implementation focusing on core concepts.

  Args:
      input_shape (Tuple[int, ...]): The expected input shape for the model.
      d_model (int): The dimensionality of the model's embedding space.
      num_heads (int): The number of attention heads in the multi-head attention layer.
      num_encoder_layers (int): The number of encoder layers in the Transformer.
      num_decoder_layers (int): The number of decoder layers in the Transformer.
      output_shape (int): The number of output units in the final layer.

  Returns:
      tf.keras.Model: A compiled Transformer model for time series forecasting.
  """

  # Define Transformer layers
  encoder_input = tf.keras.layers.Input(shape=input_shape)
  decoder_input = tf.keras.layers.Input(shape=(1, d_model))  # Single value for decoder start

  # Time2Vec positional encoding for encoder input
  time_embeddings = time2vec(input_shape[0], d_model)  # Assuming time is the first dimension
  encoded_input = encoder_input * time_embeddings

  encoder_layers = [
      tf.keras.layers.TransformerEncoder(d_model=d_model, num_heads=num_heads)
      for _ in range(num_encoder_layers)
  ]
  for layer in encoder_layers:
    encoder_output = layer(encoded_input)

  decoder_layers = [
      tf.keras.layers.TransformerDecoder(d_model=d_model, num_heads=num_heads)
      for _ in range(num_decoder_layers)
  ]
  for layer in decoder_layers:
    decoder_output = layer(decoder_input, encoder_output)

  # Final dense layer for prediction
  output = tf.keras.layers.Dense(output_shape)(decoder_output)

  # Create and compile the model (adjust optimizer and loss as needed)
  model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=output)

  return model

