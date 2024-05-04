import tensorflow as tf
from typing import Tuple

def DR_1(input_shape: Tuple[int, ...], output_shape: int = 1, num_blocks: int = 8, hidden_units: int = 128, dropout: float = 0.25) -> tf.keras.Model:
    """
    NO FUNCIONA !!
    (Model implementat basant-me en el paper: https://ceur-ws.org/Vol-2675/paper18.pdf)

    Defines a deep residual neural network architecture for time series forecasting.
    

    This function builds a multi-block residual network that takes a sequence of historical
    data points (`backcast`) as input and predicts future values (`horizon`) in the time series.

    Args:
        input_shape (Tuple[int, ...]): The shape of the input data (excluding batch dimension).
        output_shape (int): The number of output units in the final dense layer.
        num_blocks (int): The number of residual blocks to use in the network.
        hidden_units (int): The number of hidden units in the LSTM layers.
        dropout (float): The dropout rate to use in the LSTM layers.
        
    Returns:
        tf.keras.Model: A compiled Keras model representing the deep residual network.
    """
    # Input layers
    block_input = tf.keras.layers.Input(shape=(input_shape))
    
    # Split layer
    main_features, auxiliary_features = tf.keras.layers.Lambda(
        lambda x: tf.split(x, [1, input_shape[1] - 1], axis=1), 
        output_shape=((input_shape[0], 1), (input_shape[0], input_shape[1] - 1))
    )(block_input)
    
    # Residual blocks
    residual_output = main_features
    # List to hold forecasts from each block
    forecast_list = []  
    for _ in range(num_blocks):
        block_input = tf.keras.layers.concatenate([residual_output, auxiliary_features], axis=-1)
        # BiLSTM block 1
        block_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units))(block_input)
        # Droput block
        block_output = tf.keras.layers.Dropout(dropout)(block_output)
        # Batch normalization block
        block_output = tf.keras.layers.BatchNormalization()(block_output)
        # Fully Connected block
        block_output = tf.keras.layers.Dense(output_shape + input_shape[0], activation='relu')(block_output)
        # Split output into backcast and forecast
        block_forecast, block_backcast = tf.keras.layers.Lambda(
            lambda x: tf.split(x, [1, input_shape[0]], axis=1), 
            output_shape=((1, 1), (input_shape[0], 1))
        )(block_output)
        # Residual connection
        residual_output = tf.keras.layers.subtract([residual_output, block_backcast])
        # Append block forecast to the list
        forecast_list.append(block_forecast)
    
    # Sum forecasts from all blocks
    forecast = tf.keras.layers.add(forecast_list)
        
    # Create the model
    model = tf.keras.models.Model(inputs=block_input, outputs=forecast)

    return model
