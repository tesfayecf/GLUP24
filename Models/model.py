import tensorflow as tf
from typing import Tuple, Literal

from Models.LSTM import LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5
from Models.GRU import GRU_1
from Models.CNN import CNN_1
from Models.DR import DR_1
from Models.RNN import RNN_1
from Models.Transoformer import Transformer_1, Transformer_2

def get_model(model_name: Literal["LSTM", "GRU", "CNN", "DR", "RNN", "Transformer"], model_version: int, input_shape: Tuple[int, ...], output_shape: int, **model_parameters) -> tf.keras.Model:
    """
    Constructs a time series prediction model based on the specified architecture.

    This function serves as a factory, dispatching model creation based on the provided name.

    Args:
        model_name:   The name of the model architecture to use.
        model_version (int): The version of the model architecture being trained.
        input_shape (Tuple[int, ...]): The expected input shape for the model.
        output_shape (int): The number of output units in the final dense layer.
        kwargs (dict, optional): Additional keyword arguments specific to the chosen model architecture.

    Returns:
        tf.keras.Model: A compiled time series prediction model.

    Raises:
        ValueError: If an invalid model name is provided.
    """
    
    ### LSTM ###
    if model_name == "LSTM":
        if model_version == 1:
            hidden_units = model_parameters.get("hidden_units", 32)
            embedding_size = model_parameters.get("embedding_size", 32)
            dropout = model_parameters.get("dropout", 0.25)
            return LSTM_1(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size, dropout=dropout)
        
        elif model_version == 2:
            hidden_units = model_parameters.get("hidden_units", 32)
            embedding_size = model_parameters.get("embedding_size", 32)
            dropout = model_parameters.get("dropout", 0.25)
            return LSTM_2(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size, dropout=dropout)
        
        elif model_version == 3:
            hidden_units = model_parameters.get("hidden_units", 32)
            embedding_size = model_parameters.get("embedding_size", 32)
            dropout = model_parameters.get("dropout", 0.25)
            return LSTM_3(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size, dropout=dropout)
        
        elif model_version == 4:
            hidden_units = model_parameters.get("hidden_units", 32)
            embedding_size = model_parameters.get("embedding_size", 32)
            num_residual_blocks = model_parameters.get("num_residual_blocks", 2)
            dropout = model_parameters.get("dropout", 0.25)
            return LSTM_4(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size, 
                          num_residual_blocks=num_residual_blocks, dropout=dropout)
        
        elif model_version == 5:
            hidden_units = model_parameters.get("hidden_units", 32)
            embedding_size = model_parameters.get("embedding_size", 32)
            num_residual_blocks = model_parameters.get("num_residual_blocks", 2)
            dropout = model_parameters.get("dropout", 0.25)
            conv_filters = model_parameters.get("conv_filters", 32)
            conv_kernel_size = model_parameters.get("conv_kernel_size", 3)
            return LSTM_5(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size, 
                          num_residual_blocks=num_residual_blocks, dropout=dropout, conv_filters=conv_filters, 
                          conv_kernel_size=conv_kernel_size)
    
    ### GRU ###
    elif model_name == "GRU":
        if model_version == 1:
            hidden_units = model_parameters.get("hidden_units", 32)
            embedding_size = model_parameters.get("embedding_size", 32)
            dropout = model_parameters.get("dropout", 0.25)
            return GRU_1(input_shape, output_shape, hidden_units=hidden_units, embedding_size=embedding_size, dropout=dropout)

    ### CNN ###  
    elif model_name == "CNN":
        if model_version == 1:
            filters = model_parameters.get("filters", [32, 64, 128])
            kernel_size = model_parameters.get("kernel_size", 4)
            hidden_units = model_parameters.get("hidden_units", 128)
            dropout = model_parameters.get("dropout", 0.25)
            return CNN_1(input_shape, output_shape, filters=filters, kernel_size=kernel_size, hidden_units=hidden_units, dropout=dropout)
        
    ### RNN ###
    elif model_name == "RNN":
        if model_version == 1:
            hidden_units = model_parameters.get("hidden_units", 32)
            dropout = model_parameters.get("dropout", 0.25)
            return RNN_1(input_shape, output_shape, hidden_units=hidden_units, dropout=dropout)
        
    ### DR ###  
    elif model_name == "DR":
        if model_version == 1:
            num_blocks = model_parameters.get("num_blocks", 8)
            hidden_units = model_parameters.get("hidden_units", 32)
            dropout = model_parameters.get("dropout", 0.25)
            return DR_1(input_shape, output_shape, num_blocks=num_blocks, hidden_units=hidden_units, dropout=dropout)
    
    ### Transformer ###
    elif model_name == "Transformer":
        if model_version == 1:
            hidden_units = model_parameters.get("hidden_units", 32)
            filters = model_parameters.get("filters", 128)
            num_layers = model_parameters.get("num_layers", 2)
            num_heads = model_parameters.get("num_heads", 256)
            dropout = model_parameters.get("dropout", 0.25)
            return Transformer_1(input_shape, output_shape, hidden_units=hidden_units, filters=filters, num_layers=num_layers,
                                 num_heads=num_heads, dropout=dropout)
    
        elif model_version == 2:
            hidden_units = model_parameters.get("hidden_units", 32)
            num_layers = model_parameters.get("num_layers", 2)
            num_heads = model_parameters.get("num_heads", 256)
            dropout = model_parameters.get("dropout", 0.25)
            return Transformer_2(input_shape, output_shape, hidden_units=hidden_units, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    
    else:
        raise ValueError(f"Invalid model id: {model_name}_{model_version}")