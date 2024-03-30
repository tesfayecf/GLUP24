import tensorflow as tf
from typing import Tuple

from Models.LSTM import LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
from Models.GRU import GRU_1
from Models.CNN import CNN_1
from Models.DR import DR_1, DR_2

def get_model(model_id: str, input_shape: Tuple[int, ...], output_shape: int, **model_parameters) -> tf.keras.Model:
    """
    Constructs a time series prediction model based on the specified architecture.

    This function serves as a factory, dispatching model creation based on the provided name.

    Args:
        model_id (str): The name of the model to build.
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        output_shape (int): The number of output units in the final dense layer.
        kwargs (dict, optional): Additional keyword arguments specific to the chosen model architecture.

    Returns:
        tf.keras.Model: A compiled time series prediction model.

    Raises:
        ValueError: If an invalid model name is provided.
    """

    if model_id == "LSTM_1":
        hidden_units = model_parameters.get("hidden_units", 32)
        embedding_size = model_parameters.get("embedding_size", 32)
        return LSTM_1(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size)
    if model_id == "LSTM_2":
        hidden_units = model_parameters.get("hidden_units", 32)
        embedding_size = model_parameters.get("embedding_size", 32)
        return LSTM_2(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size)
    if model_id == "LSTM_3":
        hidden_units = model_parameters.get("hidden_units", 32)
        embedding_size = model_parameters.get("embedding_size", 32)
        return LSTM_3(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size)
    if model_id == "LSTM_4":
        hidden_units = model_parameters.get("hidden_units", 32)
        embedding_size = model_parameters.get("embedding_size", 32)
        num_residual_blocks = model_parameters.get("num_residual_blocks", 2)
        dropout_rate = model_parameters.get("dropout_rate", 0.3)
        return LSTM_4(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size, 
                      num_residual_blocks=num_residual_blocks, dropout_rate=dropout_rate)
    if model_id == "LSTM_5":
        hidden_units = model_parameters.get("hidden_units", 32)
        embedding_size = model_parameters.get("embedding_size", 32)
        num_residual_blocks = model_parameters.get("num_residual_blocks", 2)
        dropout_rate = model_parameters.get("dropout_rate", 0.3)
        conv_filters = model_parameters.get("conv_filters", 32)
        conv_kernel_size = model_parameters.get("conv_kernel_size", 3)
        return LSTM_5(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size, 
                      num_residual_blocks=num_residual_blocks, dropout_rate=dropout_rate,
                      conv_filters=conv_filters, conv_kernel_size=conv_kernel_size)
    elif model_id == "GRU_1":
        hidden_units = model_parameters.get("hidden_units", 32)
        embedding_size = model_parameters.get("embedding_size", 32)
        return GRU_1(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size)
    elif model_id == "CNN_1":
        filters = model_parameters.get("filters", [32, 64, 128])
        kernel_size = model_parameters.get("kernel_size", 3)
        hidden_units = model_parameters.get("hidden_units", 128)
        return CNN_1(input_shape, output_shape, filters=filters, kernel_size=kernel_size, hidden_units=hidden_units)
    elif model_id == "DR_1":
        num_blocks = model_parameters.get("num_blocks", 3)
        hidden_units = model_parameters.get("hidden_units", 32)
        embedding_size = model_parameters.get("embedding_size", 32)
        auxiliary_variables = model_parameters.get("auxiliary_variables", 17)
        return DR_2(input_shape, output_shape, num_blocks=num_blocks, hidden_units=hidden_units, embedding_size=embedding_size, auxiliary_variables=auxiliary_variables)
    else:
        raise ValueError(f"Invalid model id: {model_id}")