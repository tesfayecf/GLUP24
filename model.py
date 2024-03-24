import tensorflow as tf
from train import train_model

def build_model(input_shape, hidden_units, embedding_size, output_shape):
    """
    Builds a time series prediction model using stacked LSTMs with dropout.

    Args:
        input_shape: The expected input shape for the model.
        hidden_units: The number of hidden units in each LSTM layer.
        output_shape: The number of output units in the final dense layer.

    Returns:
        A compiled and trained time series prediction model.
    """
    input_features = tf.keras.layers.Input(shape=input_shape)

    # LSTM layers for input features
    lstm_output = tf.keras.layers.LSTM(hidden_units, return_sequences=True)(input_features)
    lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)
    
    # Additional LSTM layer
    lstm_output = tf.keras.layers.LSTM(hidden_units)(lstm_output)
    lstm_output = tf.keras.layers.Dropout(0.3)(lstm_output)

    # Dense layers
    dense_output = tf.keras.layers.Dense(embedding_size, activation='linear')(lstm_output)
    output = tf.keras.layers.Dense(output_shape, activation='linear')(dense_output)

    model = tf.keras.models.Model(inputs=input_features, outputs=output)
    return model

class CustomModel(tf.keras.Model):
    """
    Time series prediction model using stacked LSTMs with dropout.
    This class provides functionalities for building, compiling, training, evaluating, and predicting with a time series model using stacked LSTMs with dropout.

    Attributes:
        lstm_layer1: The first LSTM layer in the model.
        lstm_layer2: The second LSTM layer in the model.
        dropout: The dropout layer used for regularization.
        dense_layer: The output layer of the model.
        input_shape: The expected input shape for the model.
        optimizer: The optimizer used for training (set during compilation).
        loss: The loss function used for training (set during compilation).
        metrics: A list of metrics tracked during training and evaluation (set during compilation).
    """

    def __init__(self, input_shape, hidden_units, output_shape):
        super(CustomModel, self).__init__()
        self.lstm_layer1 = tf.keras.layers.LSTM(hidden_units, return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        
        self.lstm_layer2 = tf.keras.layers.LSTM(hidden_units)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        
        self.dense_layer = tf.keras.layers.Dense(units=output_shape)
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.optimizer = None
        self.loss = None

    def call(self, inputs):
        """
        Defines the forward pass of the model.

        Args:
            inputs: The input data to the model.

        Returns:
            The model's output after processing the input data.
        """
        # First LSTM layer
        x = self.lstm_layer1(inputs)
        # First dropout layer
        x = self.dropout1(x)
        # Second LSTM layer
        x = self.lstm_layer2(x)
        # Second dropout layer
        x = self.dropout2(x)
        # Dense layer
        output = self.dense_layer(x)
        return output

    def compile(self, optimizer, loss, metrics=['mae']):
        """
        Compiles the model with the specified optimizer, loss function, and metrics.

        Args:
            optimizer: The optimizer to use for training (e.g., tf.keras.optimizers.Adam).
            loss: The loss function to use (e.g., tf.keras.losses.MeanSquaredError).
            metrics: A list of metrics to track during training and evaluation (optional).
        """
        self.optimizer = optimizer
        self.loss = loss
        super(CustomModel, self).compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, train_dataset, validation_dataset, test_dataset, epochs=1, steps_per_epoch=1, callbacks=None):
        """
        Trains the model using a custom training loop with early stopping (optional).

        Args:
            train_dataset: A TensorFlow Dataset for training data.
            validation_dataset: A TensorFlow Dataset for validation data (optional).
            epochs: Number of training epochs (default 1).
            steps_per_epoch: Number of training steps per epoch (optional).
            callbacks: A list of Keras callbacks to use during training (optional).
        """
        train_model(self, train_dataset, validation_dataset, test_dataset, epochs, steps_per_epoch)