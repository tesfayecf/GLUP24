import tensorflow as tf
from typing import Tuple

MODEL_NAME = "DR"
MODEL_VERSION = 1

################# PARAMETERS #################
num_blocks: int = 7
hidden_units: int = 300
embedding_size: int = 32
auxiliary_variables: int = 4

# train_model(MODEL_NAME, MODEL_VERSION, num_blocks=num_blocks, hidden_units=hidden_units, embedding_size=embedding_size, auxiliary_variables=auxiliary_variables)

def DR_1(input_shape: Tuple[int, ...], output_shape: int = 1, num_blocks: int = 7, hidden_units: int = 300, embedding_size: int = 32, auxiliary_variables: int = 4) -> tf.keras.Model:
    """
    Implements the deep residual time series forecasting model with RNN blocks, additional input variables, and auxiliary losses.

    Args:
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        output_shape (int): The number of output units in the final dense layer.
        num_blocks (int, optional): The number of residual blocks in the model. Defaults to 7.
        hidden_units (int, optional): The number of hidden units in each residual block's LSTM layer. Defaults to 300.
        embedding_size (int, optional): The size of the embedding layer. Defaults to 32.
        auxiliary_variables (int, optional): The number of additional input variables besides the primary time series. Defaults to 4.

    Returns:
        tf.keras.Model: A compiled deep residual time series forecasting model.
    """

    # Input layers
    input_features = tf.keras.layers.Input(shape=input_shape)
    auxiliary_input = tf.keras.layers.Input(shape=(input_shape[0], auxiliary_variables))

    # Residual blocks
    residual_output = input_features
    for block_idx in range(num_blocks):
        block_input = tf.keras.layers.concatenate([residual_output, auxiliary_input], axis=-1)

        # RNN block
        lstm_output = tf.keras.layers.LSTM(hidden_units, return_sequences=True)(block_input)
        lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)

        # Split output into backcast and forecast
        backcast, forecast = tf.keras.layers.Lambda(lambda x: tf.split(x, [input_shape[0], output_shape], axis=1))(lstm_output)

        # Residual connection
        residual_output = input_features - backcast

        # Auxiliary losses
        backcast_loss = tf.keras.losses.MeanSquaredError()(input_features, backcast)
        forecast_loss = tf.keras.losses.MeanSquaredError()(tf.zeros_like(forecast), forecast)
        magnitude_loss = tf.keras.backend.sum(1 / (block_idx + 1) * (1 / tf.keras.backend.sum(tf.abs(backcast), axis=1)))

    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='relu')(forecast)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)

    # Create the model
    model = tf.keras.models.Model(inputs=[input_features, auxiliary_input], outputs=output)

    # Auxiliary losses
    backcast_loss_weight = 0.3
    forecast_loss_weight = 1.0
    magnitude_loss_weight = 1e-4

    model.add_loss(backcast_loss_weight * backcast_loss)
    model.add_loss(forecast_loss_weight * forecast_loss)
    model.add_loss(magnitude_loss_weight * magnitude_loss)

    return model

