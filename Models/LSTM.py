import tensorflow as tf
from typing import Tuple

################# V1 ################# 
def LSTM_1(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, embedding_size: int = 32, dropout: float = 0.25) -> tf.keras.Model:
    """
    Implements a time series prediction model using stacked LSTMs with dropout regularization.
    This function constructs an LSTM model with two stacked LSTM layers followed by dense layers.

    Args:
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        output_shape (int): The number of output units in the final dense layer.
        hidden_units (int, optional): The number of hidden units in each LSTM layer.
        embedding_size (int, optional): The size of the embedding layer.
        dropout (float, optional): The dropout rate to use.

    Returns:
        tf.keras.Model: A compiled time series prediction model.
    """
    # Feature input layer
    input_features = tf.keras.layers.Input(shape=input_shape)

    # LSTM layers for input features
    lstm_output = tf.keras.layers.LSTM(hidden_units, return_sequences=True)(input_features)
    lstm_output = tf.keras.layers.Dropout(dropout)(lstm_output)
    lstm_output = tf.keras.layers.BatchNormalization()(lstm_output)
    
    # Additional LSTM layer
    lstm_output = tf.keras.layers.LSTM(hidden_units)(lstm_output)
    lstm_output = tf.keras.layers.Dropout(dropout)(lstm_output)
    lstm_output = tf.keras.layers.BatchNormalization()(lstm_output)

    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='relu')(lstm_output)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)

    # Create the model
    model = tf.keras.models.Model(inputs=input_features, outputs=output)

    return model

################# V2 ################# 
def LSTM_2(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, embedding_size: int = 32, dropout: float = 0.25) -> tf.keras.Model:
    """
    Implements a time series prediction model using stacked bidirectional LSTMs with dropout regularization.
    This function constructs an LSTM model with two stacked bidirectional LSTM layers followed by dense layers.

    Args:
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        output_shape (int): The number of output units in the final dense layer.
        hidden_units (int, optional): The number of hidden units in each LSTM layer.
        embedding_size (int, optional): The size of the embedding layer.
        dropout (float, optional): The dropout rate to use.

    Returns:
        tf.keras.Model: A compiled time series prediction model.
    """
    # Feature input layer
    input_features = tf.keras.layers.Input(shape=input_shape)

    # Bidirectional LSTM layers for input features
    lstm_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))(input_features)
    lstm_output = tf.keras.layers.Dropout(dropout)(lstm_output)
    lstm_output = tf.keras.layers.BatchNormalization()(lstm_output)

    # Additional bidirectional LSTM layer
    lstm_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units))(lstm_output)
    lstm_output = tf.keras.layers.Dropout(dropout)(lstm_output)
    lstm_output = tf.keras.layers.BatchNormalization()(lstm_output)

    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='relu')(lstm_output)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)

    # Create the model
    model = tf.keras.models.Model(inputs=input_features, outputs=output)

    return model

################# V3 ################# 
def LSTM_3(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, embedding_size: int = 32, dropout: float = 0.25) -> tf.keras.Model:
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
    lstm_output = tf.keras.layers.Dropout(dropout)(lstm_output)
    lstm_output = tf.keras.layers.BatchNormalization()(lstm_output)

    # Additional bidirectional LSTM layer
    lstm_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True))(lstm_output)
    lstm_output = tf.keras.layers.Dropout(dropout)(lstm_output)
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

class AttentionLayer(tf.keras.layers.Layer):
    """
    Attention layer that computes the context vector using weighted attention.

    Args:
        units (int): Dimensionality of the neural network layers.

    Attributes:
        W1 (tf.keras.layers.Dense): Fully connected layer for the first weight.
        W2 (tf.keras.layers.Dense): Fully connected layer for the second weight.
        V (tf.keras.layers.Dense): Fully connected layer for the attention weight.
    """
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, values, query):
        """
        Applies the weighted attention operation to compute the context vector.

        Args:
            values (tf.Tensor): Input values for which attention is applied, of shape (batch_size, seq_length, embedding_dim).
            query (tf.Tensor): Query vector to compute attention weights, of shape (batch_size, units).

        Returns:
            tf.Tensor: The context vector, of shape (batch_size, embedding_dim).
        """
        # Compute attention scores
        scores = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))
        scores = tf.squeeze(scores, -1)
        scores = tf.nn.softmax(scores, axis=1)

        # Apply weighted attention
        context_vector = tf.reduce_sum(values * tf.expand_dims(scores, -1), axis=1)

        return context_vector


