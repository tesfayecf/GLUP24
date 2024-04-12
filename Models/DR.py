import tensorflow as tf
from typing import Tuple

# https://gitlab.eecs.umich.edu/mld3/deep-residual-time-series-forecasting/-/blob/master/drtf.py?ref_type=heads

# Function to define the deep residual network architecture
def DR_1(input_shape: Tuple[int, ...], output_shape: int = 1, num_blocks: int = 7, hidden_units: int = 300) -> tf.keras.Model:
    """
    Defines a deep residual neural network architecture for time series forecasting.

    This function builds a multi-block residual network that takes a sequence of historical
    data points (`backcast`) as input and predicts future values (`horizon`) in the time series.

    Args:
        backcast (int): The number of past time steps used as input for the model.
        hidden_units (int): The number of hidden units in each dense layer of the residual blocks.

    Returns:
        tf.keras.Model: A compiled Keras model representing the deep residual network.
    """

    # Input layer
    input_features = tf.keras.layers.Input(shape=input_shape)
    
    
    # Residual blocks
    x = input_features
    for _ in range(num_blocks):
        shortcut = x
        x = tf.keras.layers.Dense(hidden_units, activation="relu")(x)
        x = tf.keras.layers.Dense(input_shape[1])(x)
        # Add the residual connection (shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation("relu")(x)
    
    # Dense layers
    outputs = tf.keras.layers.Dense(output_shape)(x)
    
    # Create the model
    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    return model


def DR_2(input_shape: Tuple[int, ...], output_shape: int = 1, num_blocks: int = 7, hidden_units: int = 300, embedding_size: int = 32, auxiliary_variables: int = 17) -> tf.keras.Model:
    """ Implements the deep residual time series forecasting model with RNN blocks, additional input variables, and auxiliary losses.
    Args:
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        output_shape (int): The number of output units in the final dense layer.
        num_blocks (int, optional): The number of residual blocks in the model. Defaults to 7.
        hidden_units (int, optional): The number of hidden units in each residual block's LSTM layer. Defaults to 300.
        embedding_size (int, optional): The size of the embedding layer. Defaults to 32.
        auxiliary_variables (int, optional): The number of additional input variables besides the primary time series. Defaults to 17.
    Returns:
        tf.keras.Model: A compiled deep residual time series forecasting model.
    """
    block_input = tf.keras.layers.Input(shape=(input_shape))
    
    input_features, auxiliary_input = tf.keras.layers.Lambda(
        lambda x: tf.split(x, [1, auxiliary_variables], axis=1), 
        output_shape=((input_shape[0], 1), (input_shape[0], auxiliary_variables))
    )(block_input)
    
    # Input layers
    # input_features = tf.keras.layers.Input(shape=(input_shape[0], 1))
    # auxiliary_input = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1] - 1))

    # Residual blocks
    backcast_losses, forecast_losses, magnitude_losses = [], [], []
    residual_output = input_features
    for block_idx in range(num_blocks):
        block_input = tf.keras.layers.concatenate([residual_output, auxiliary_input], axis=-1)
        # RNN block
        lstm_output = tf.keras.layers.LSTM(hidden_units, return_sequences=False)(block_input)
        lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)
        # Split output into backcast and forecast
        backcast, forecast = tf.keras.layers.Lambda(lambda x: tf.split(x, [1, auxiliary_variables], axis=1), output_shape=((input_shape[0], 1), (input_shape[0], auxiliary_variables)))(lstm_output)
        # Residual connection
        residual_output = input_features - backcast
        # Auxiliary losses
        backcast_loss = MeanSquaredErrorLayer()([input_features, backcast])
        zeros_like_forecast = ZerosLikeLayer()(forecast)
        forecast_loss = MeanSquaredErrorLayer()([zeros_like_forecast, forecast])
        magnitude_loss = MagnitudeLossLayer()(backcast, block_idx=block_idx)

        backcast_losses.append(backcast_loss)
        forecast_losses.append(forecast_loss)
        magnitude_losses.append(magnitude_loss)

    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='relu')(forecast)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)

    # Create the model
    model = tf.keras.models.Model(inputs=[input_features, auxiliary_input], outputs=output)

    # Auxiliary losses
    backcast_loss_weight = 0.3
    forecast_loss_weight = 1.0
    magnitude_loss_weight = 1e-4
    
    # model.add_loss(backcast_loss_weight * backcast_loss)
    # model.add_loss(forecast_loss_weight * forecast_loss)
    # model.add_loss(magnitude_loss_weight * magnitude_loss)
    
    def custom_loss(y_true, y_pred):
        main_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        backcast_loss = tf.reduce_mean(backcast_loss_weight * tf.stack(backcast_losses))
        forecast_loss = tf.reduce_mean(forecast_loss_weight * tf.stack(forecast_losses))
        magnitude_loss = tf.reduce_mean(magnitude_loss_weight * tf.stack(magnitude_losses))
        return main_loss + backcast_loss + forecast_loss + magnitude_loss

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=custom_loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
        ]
    )

    return model


class MeanSquaredErrorLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        if isinstance(inputs, list):
            y_true, y_pred = inputs
        else:
            y_true, y_pred = inputs
        return tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    
class ZerosLikeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.zeros_like(inputs)
    
class MagnitudeLossLayer(tf.keras.layers.Layer):
    def call(self, inputs, block_idx):
        backcast = inputs
        return tf.keras.backend.sum(1 / (block_idx + 1) * (1 / tf.keras.backend.sum(tf.abs(backcast), axis=1)))
