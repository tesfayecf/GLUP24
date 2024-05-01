import tensorflow as tf
from typing import Tuple

def iTransformer_1(input_shape: Tuple[int, ...], output_shape: int = 1, hidden_units: int = 32, num_layers: int = 2, num_heads: int = 256, dropout_rate: float = 0.25) -> tf.keras.Model:
    """
    Creates an iTransformer model with 1 hidden layer.

    Args:
        input_shape (Tuple[int, ...]): Shape of the input features.
        output_shape (int, optional): Shape of the output. Defaults to 1.
        hidden_units (int, optional): Number of hidden units in the model. Defaults to 32.
        num_layers (int, optional): Number of transformer layers. Defaults to 2.
        num_heads (int, optional): Number of attention heads. Defaults to 256.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.25.

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
        attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout_rate)(embedding_with_position, embedding_with_position)
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