import tensorflow as tf
from typing import Tuple

from Models.LSTM import LSTM_1
from Models.GRU import GRU_1
from Models.CNN import CNN_1
from Models.DR import DR_1

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
        return LSTM_1(input_shape, output_shape, **model_parameters)
    elif model_id == "GRU_1":
        return GRU_1(input_shape, output_shape, **model_parameters)
    elif model_id == "CNN_1":
        return CNN_1(input_shape, output_shape, **model_parameters)
    elif model_id == "DR_1":
        return DR_1(input_shape, output_shape, **model_parameters)
    else:
        raise ValueError(f"Invalid model id: {model_id}")