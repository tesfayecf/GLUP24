import tensorflow as tf
from typing import Tuple

def Transformer_1(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, filters: int = 128, num_layers: int = 3, num_heads: int = 256, dropout: float = 0.25) -> tf.keras.Model:
    """
    Build a Transformer model.
    
    Args:
        input_shape: Shape of the input data.
        output_shape: Number of output classes.
        hidden_units: Number of hidden units in the feedforward part of the model.
        filters: Number of filters in the convolutional part of the model.
        num_layers: Number of layers in the model.
        num_heads: Number of attention heads in the model.
        dropout: Dropout rate.

    Returns:
        A TensorFlow model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_layers):
        # Normalization and Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=hidden_units, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x += inputs  # Residual connection

        # Feed Forward Part
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x += inputs  # Residual connection

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = tf.keras.layers.Dense(hidden_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(output_shape)(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def Transformer_2(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, num_layers: int = 2, num_heads: int = 256, dropout: float = 0.25) -> tf.keras.Model:
    """
    Creates an Inverse transformer model.
    
    (Model implementat a partir del paper: https://arxiv.org/abs/2310.06625)

    Args:
        input_shape (Tuple[int, ...]): Shape of the input features.
        output_shape (int, optional): Shape of the output.
        hidden_units (int, optional): Number of hidden units in the model.
        num_layers (int, optional): Number of transformer layers.
        num_heads (int, optional): Number of attention heads.
        dropout (float, optional): Dropout rate.

    Returns:
        tf.keras.Model: The iTransformer model.
    """
    # Input layer
    input_features = tf.keras.layers.Input(shape=input_shape)
    
    # Transpose the input features
    inv_input = tf.keras.layers.Permute((2,1))(input_features)
    
    # Embedding layer
    embedding = tf.keras.layers.Dense(hidden_units)(inv_input)
    
    # Positional encoding
    position_encoding = tf.keras.layers.Embedding(input_dim=input_shape[0], output_dim=hidden_units)(tf.range(input_shape[1]))
    embedding_with_position = embedding + position_encoding
    
    # Transformer Encoder
    for _ in range(num_layers):
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout)(embedding_with_position, embedding_with_position)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + embedding_with_position)
        
        # Feed forward network
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(hidden_units)
        ])
        ffn_output = ffn(attention)
        embedding_with_position = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention)
    
    # Output projection
    output = tf.keras.layers.GlobalAveragePooling1D()(embedding_with_position)
    output = tf.keras.layers.Dense(output_shape)(output)
    
    # Create model
    model = tf.keras.Model(inputs=input_features, outputs=output)
    
    return model