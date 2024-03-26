import tensorflow as tf
from typing import Tuple

MODEL_NAME = "CNN"
MODEL_VERSION = 1

################# PARAMETERS #################
filters: list[int] = [32, 64, 128]
kernel_size: int = 3
hidden_units: int = 128

# train_model(MODEL_NAME, MODEL_VERSION, filters=filters, kernel_size=kernel_size, hidden_units=hidden_units)

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
