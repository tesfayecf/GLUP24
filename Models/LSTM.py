import tensorflow as tf
from typing import Tuple

################# V1 ################# 
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
    lstm_output = tf.keras.layers.LSTM(hidden_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))(input_features)
    lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)
    lstm_output = tf.keras.layers.BatchNormalization()(lstm_output)
    
    # Additional LSTM layer
    lstm_output = tf.keras.layers.LSTM(hidden_units, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))(lstm_output)
    lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)
    lstm_output = tf.keras.layers.BatchNormalization()(lstm_output)

    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='relu')(lstm_output)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)

    # Create the model
    model = tf.keras.models.Model(inputs=input_features, outputs=output)

    return model

################# V2 ################# 
def LSTM_2(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, embedding_size: int = 32) -> tf.keras.Model:
    """
    Implements a time series prediction model using stacked bidirectional LSTMs with dropout regularization.
    This function constructs an LSTM model with two stacked bidirectional LSTM layers followed by dense layers.

    Args:
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        hidden_units (int, optional): The number of hidden units in each LSTM layer. Defaults to 32.
        embedding_size (int, optional): The size of the embedding layer. Defaults to 32.
        output_shape (int): The number of output units in the final dense layer.

    Returns:
        tf.keras.Model: A compiled time series prediction model.
    """
    # Feature input layer
    input_features = tf.keras.layers.Input(shape=input_shape)

    # Bidirectional LSTM layers for input features
    lstm_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))(input_features)
    lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)
    lstm_output = tf.keras.layers.BatchNormalization()(lstm_output)

    # Additional bidirectional LSTM layer
    lstm_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units))(lstm_output)
    lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)
    lstm_output = tf.keras.layers.BatchNormalization()(lstm_output)

    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='relu')(lstm_output)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)

    # Create the model
    model = tf.keras.models.Model(inputs=input_features, outputs=output)

    return model

################# V3 ################# 
def LSTM_3(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, embedding_size: int = 32) -> tf.keras.Model:
    """
    Implements a time series prediction model using stacked bidirectional LSTMs with an attention mechanism, batch normalization, and dropout regularization.
    This function constructs an LSTM model with two stacked bidirectional LSTM layers, followed by an attention layer and dense layers.

    Args:
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        hidden_units (int, optional): The number of hidden units in each LSTM layer. Defaults to 32.
        embedding_size (int, optional): The size of the embedding layer. Defaults to 32.
        output_shape (int): The number of output units in the final dense layer.

    Returns:
        tf.keras.Model: A compiled time series prediction model.
    """
    # Feature input layer
    input_features = tf.keras.layers.Input(shape=input_shape)

    # Bidirectional LSTM layers for input features
    lstm_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))(input_features)
    lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)
    lstm_output = tf.keras.layers.BatchNormalization()(lstm_output)

    # Additional bidirectional LSTM layer
    lstm_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))(lstm_output)
    lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)
    lstm_output = tf.keras.layers.BatchNormalization()(lstm_output)

    # Attention layer
    attention_layer = AttentionLayer(hidden_units)
    context_vector = attention_layer(lstm_output, lstm_output)

    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='relu')(context_vector)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)

    # Create the model
    model = tf.keras.models.Model(inputs=input_features, outputs=output)

    return model

################# V4 ################# 
def LSTM_4(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, embedding_size: int = 32, num_residual_blocks: int = 2, dropout_rate: float = 0.3) -> tf.keras.Model:
    """
    Implements a time series prediction model using stacked bidirectional LSTMs with deep residual blocks, an attention mechanism, batch normalization, and dropout regularization.
    This function constructs an LSTM model with deep residual blocks, followed by an attention layer and dense layers.

    Args:
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        hidden_units (int, optional): The number of hidden units in each LSTM layer. Defaults to 32.
        embedding_size (int, optional): The size of the embedding layer. Defaults to 32.
        output_shape (int): The number of output units in the final dense layer.
        num_residual_blocks (int, optional): The number of residual blocks to use. Defaults to 2.
        dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.3.

    Returns:
        tf.keras.Model: A compiled time series prediction model.
    """
    # Feature input layer
    input_features = tf.keras.layers.Input(shape=input_shape)

    # Residual blocks
    x = input_features
    for _ in range(num_residual_blocks):
        x = residual_block(x, hidden_units, dropout_rate)

    # Attention layer
    attention_layer = AttentionLayer(hidden_units)
    context_vector = attention_layer(x, x)

    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='relu')(context_vector)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)

    # Create the model
    model = tf.keras.models.Model(inputs=input_features, outputs=output)

    return model

################# V5 ################# 
def LSTM_5(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, embedding_size: int = 32, num_residual_blocks: int = 2, dropout_rate: float = 0.3, conv_filters: int = 64, conv_kernel_size: int = 3) -> tf.keras.Model:
    """
    Implements a time series prediction model using stacked bidirectional LSTMs with deep residual blocks, an attention mechanism, batch normalization, dropout regularization, and convolutional layers.
    This function constructs an LSTM model with convolutional layers, followed by deep residual blocks, an attention layer, and dense layers.

    Args:
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        hidden_units (int, optional): The number of hidden units in each LSTM layer. Defaults to 32.
        embedding_size (int, optional): The size of the embedding layer. Defaults to 32.
        output_shape (int): The number of output units in the final dense layer.
        num_residual_blocks (int, optional): The number of residual blocks to use. Defaults to 2.
        dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.3.
        conv_filters (int, optional): The number of filters in the convolutional layers. Defaults to 64.
        conv_kernel_size (int, optional): The kernel size for the convolutional layers. Defaults to 3.

    Returns:
        tf.keras.Model: A compiled time series prediction model.
    """
    # Feature input layer
    input_features = tf.keras.layers.Input(shape=input_shape)

    # Convolutional layers
    x = input_features
    x = tf.keras.layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, hidden_units, dropout_rate)

    # Attention layer
    attention_layer = AttentionLayer(hidden_units)
    context_vector = attention_layer(x, x)

    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='relu')(context_vector)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)

    # Create the model
    model = tf.keras.models.Model(inputs=input_features, outputs=output)

    return model

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, values, query):
        # Compute attention scores
        scores = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))
        scores = tf.squeeze(scores, -1)
        scores = tf.nn.softmax(scores, axis=1)

        # Apply attention
        context_vector = tf.reduce_sum(values * tf.expand_dims(scores, -1), axis=1)

        return context_vector

def residual_block(x, hidden_units, dropout_rate):
    shortcut = x
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))(x)

    x = tf.keras.layers.add([x, shortcut])
    return x