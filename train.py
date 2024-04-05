import time
import mlflow
import logging
# import structlog
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Other.utils import *
from Other.plot import *
from Other.columns import columns, feature_columns, target_columns
from Models.model import get_model

DATASET_NUMBER = 559
PREDICTION = 30
# HOST = "192.168.1.19"
HOST = "localhost"
PORT = 7777
EXPERIMENT_NAME = "TEST"
LOG_LEVEL = logging.INFO

datasets_dir = "./Datasets"
# datasets_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Dataset'
logs_dir = "./Logs"
# logs_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Logs'

######### DEFAULT DATASET PARAMETERS #########
sequence_size_ = 12
prediction_time_ = int(PREDICTION / 5)
validation_split_ = 0.2
######### DEFAULT TRAINING PARAMETERS #########
learning_rate_ = 0.001
epochs_ = 50
optimizer_ = 'Adam'
loss_ = 'mse'
batch_size_ = 128 # best: 128
##############################################

def train_model(model_name: str, model_version: int, **parameters) -> float:
    """Trains a machine learning model on a given dataset, evaluates its perfomance
    and logs results along with the model to MLflow.

    Args:
        model_name: The name of the model architecture to use (e.g., "LSTM_1").
        model_version: The version of the model architecture being trained.
        parameters: A dictionary containing optional hyperparameter values to override defaults.
            - sequence_size (int): The number of timesteps to include in a sequence for training (default: 12).
            - prediction_time (int): The number of timesteps to predict in the future (default: PREDICTION / 5, where PREDICTION is a constant).
            - validation_split (float): The proportion of the training data to use for validation (default: 0.2).
            - learning_rate (float): The learning rate for the optimizer (default: 0.001).
            - epochs (int): The number of epochs to train the model for (default: 50).
            - optimizer (str): The optimizer to use for training (default: "Adam"). Supported optimizers include "Adam" and others (refer to TensorFlow documentation).
            - loss (str): The loss function to use for training (default: "mse"). Supported loss functions include "mse" (mean squared error) and others (refer to TensorFlow documentation).
            - batch_size (int): The number of samples to process in each batch during training (default: 32).

    Returns:
        The mean absolute error (MAE) achieved by the model on the validation dataset.

    Raises:
        Exception: If any errors occur during training, evaluation, or logging.
    """
    
    # log = structlog.get_logger()
    # structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVEL))
    log = logging.getLogger(__name__)
    logging.basicConfig(level=LOG_LEVEL)
    
    # Generate model name based on parameters and timestamp
    timestamp = int(time.time())
    run_name = f"{model_name}_{model_version}-{DATASET_NUMBER}-{PREDICTION}-[{timestamp}]"
    log.info(f"Model run name: {run_name}")
    
    # Get dataset parameters
    sequence_size = parameters.get('sequence_size', sequence_size_)
    prediction_time = parameters.get('prediction_time', prediction_time_)
    validation_split = parameters.get('validation_split', validation_split_)
    # Get training parameters
    learning_rate = parameters.get('learning_rate', learning_rate_)
    epochs = parameters.get('epochs', epochs_)
    optimizer = parameters.get('optimizer', 'Adam')
    loss = parameters.get('loss', 'mae')
    batch_size = parameters.get('batch_size', batch_size_)
    
    ################# TRACKING #################
    try:
        # Connect to tracking server
        log.info("Connecting to tracking server")
        mlflow.set_tracking_uri(f"http://{HOST}:{PORT}")
        # Set experiment name and create experiment if it doesn't exist
        mlflow.set_experiment(f"{EXPERIMENT_NAME}-{DATASET_NUMBER}-{PREDICTION}")
        log.debug(f"Experiment: {mlflow.get_experiment_by_name(f'{EXPERIMENT_NAME}-{DATASET_NUMBER}-{PREDICTION}')}")
    except Exception as e:
        # log.exception(f"Error connecting to tracking server", exec_info=e)
        log.exception(f"Error connecting to tracking server: {e}")
        return

    ################# DATASETS #################
    try:
        log.info("Reading datasets")
        # Get training and validation data
        train_val_data = pd.read_csv(f'{datasets_dir}/{DATASET_NUMBER}/{DATASET_NUMBER}_train.csv', sep=';',encoding = 'unicode_escape', names=columns)
        train_val_data = process_data(train_val_data, scale_data=True)
        log.debug(f"Training data info: {train_val_data.describe()}")
    except Exception as e:
        # log.exception(f"Error reading datasets", exec_info=e)
        log.exception(f"Error reading datasets: {e}")
        return

    ################# PREPROCESSING #################
    try:
        log.info("Creating training and validation datasets")
        # Create sequences and targets for training
        X_train_val, Y_train_val = to_sequences_multi(train_val_data,  sequence_size, prediction_time, target_columns)
        # Split training and validation data
        x_train, x_val, y_train, y_val = train_test_split(X_train_val, Y_train_val, test_size=validation_split, random_state=42)
        # Create TensorFlow datasets for training
        train_dataset = (tf.data.Dataset
                        .from_tensor_slices((x_train, y_train))
                        .batch(batch_size)
                        .shuffle(len(x_train))
                        .prefetch(tf.data.experimental.AUTOTUNE)
                    )
        log.debug(f"Training dataset size: {len(x_train)}")
        
        # Filter values that are 0 for validation data
        x_val = x_val[y_val[:,0] != 0]
        y_val = y_val[y_val[:,0] != 0]
        
        # Create TensorFlow datasets for validation
        validation_dataset = (tf.data.Dataset
                        .from_tensor_slices((x_val, y_val))
                        .batch(batch_size)
                        .shuffle(len(x_val))
                        .prefetch(tf.data.experimental.AUTOTUNE)
                    )
        log.debug(f"Validation dataset size: {len(x_val)}")
    except Exception as e:
        # log.exception(f"Error creating datasets", exec_info=e)
        log.exception(f"Error creating datasets: {e}")
        return
    
    ################# BUILD MODEL #################
    try:
        log.info("Building the model")
        # Get input shape from training data  NOTE: This can change from config if select_features == True
        input_shape = x_train.shape[1:]
        # Create the model with default parameters if there's a ValueError
        try:
            model_id = f"{model_name}_{model_version}"
            model = get_model(model_id, input_shape, len(target_columns), **parameters)                            
            log.debug(f"Using model: {model}")
        except ValueError as e:
            log.warning(f"Invalid model: {model_id}, using defualt (LSTM_1)")
            model = get_model("LSTM_1", input_shape, len(target_columns))
        except Exception as e:
            # log.exception(f"Error getting model", exec_info=e)
            log.exception(f"Error getting model: {e}")
            return

        # Get optimizer
        try:
            optimizer_ = get_optimizer(optimizer, learning_rate)
            log.debug(f"Using optimizer: {optimizer_}")
        except ValueError as e:
            log.warning(f"Invalid optimizer: {optimizer}, using default (Adam)")
            optimizer_ = get_optimizer("Adam", learning_rate) 

        # Get loss
        try:
            loss_ = get_loss(loss)
            log.debug(f"Using loss function: {loss_}")
        except ValueError as e:
            log.warning(f"Invalid loss function: {loss}, using default (mse)")
            loss_ = get_loss("mse")  # Default loss
        
        log.info("Compiling the model")
        # Compile model
        model.compile(
            optimizer=optimizer_, 
            loss=loss_, 
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanSquaredError(),
            ]
        )
        log.info("Model built successfully")
    except Exception as e:
        # log.exception(f"Error building model", exec_info=e)
        log.exception(f"Error building model: {e}")
        return
    
    ################# TRAIN AND TEST MODEL #################
    try:
        with mlflow.start_run(run_name=run_name) as run:  
            # Define callbacks
            callbacks = [
                mlflow.keras.MLflowCallback(run),
                tf.keras.callbacks.TensorBoard(f"{logs_dir}/{run_name}"),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.2),
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            ]
            
            ############## PARAMETERS ##############
            log.info("Logging model parameters")
            # Create params for mlflow
            params = {
                ### DATASET ###
                "sequence_size": sequence_size,
                "prediction_time": prediction_time,
                "validation_split": validation_split,
                
                ### TRAINING ###
                "learning_rate": learning_rate,
                "epochs": epochs,
                "optimizer": optimizer,
                "loss": loss,
                "batch_size": batch_size,
                
                ### MODEL ### 
                **parameters
            }
        
            mlflow.log_params(params)
            mlflow.log_dict(params, "parameters.json")
            log.debug(f"Model parameters: {params}")
            
            ################# TRAIN #################
            # Train the model
            log.info("Starting training")
            start_time = time.time()
            model.fit(train_dataset, epochs=epochs, callbacks=callbacks)
            log.debug(f"Training time: {time.time() - start_time}")     
        
            ################# TEST #################
            # Run test inference
            log.info("Running test inference")
            start_time = time.time()
            y_pred = model.predict(validation_dataset)
            log.debug(f"Inference time: {time.time() - start_time}")
            
            ################# MODEL #################
            # Log an instance of the trained model for later use
            log.info("Logging trained model")
            mlflow.tensorflow.log_model(model, artifact_path="model")

            ################# METRICS #################
            log.info("Logging test metrics")
            # Calculate test metrics
            metrics = {
                "mae": mean_absolute_error(y_val, y_pred), 
                "mse": mean_squared_error(y_val, y_pred), 
                "rmse": np.sqrt(mean_squared_error(y_val, y_pred)), 
                "r2": r2_score(y_val, y_pred)
            }
            mlflow.log_metrics(metrics)
            log.debug(f"Test metrics: {metrics}")
            
            ################# CHARTS #################
            # Generate chart of real values vs prediction over the test data
            # https://safjan.com/regression-model-errors-plot/
            log.info("Generating charts")
            line_plot = generate_line_plot(y_val, y_pred)
            mlflow.log_figure(line_plot, "real_vs_prediction.png")
            scatter_plot = generate_scatter_plot(y_val, y_pred)
            mlflow.log_figure(scatter_plot, "scatter.png")
            histogram = generate_histogram_residuals(y_val, y_pred)
            mlflow.log_figure(histogram, "histogram.png")
            
    except Exception as e:
        # log.exception(f"Error training model", exec_info=e)
        log.exception(f"Error training model: {e}")
        return
    
    log.info("Model training complete")
    return metrics['mae']