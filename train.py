import time
import mlflow
import logging
import structlog
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import *
from plot import *
from model import get_model

DATASET_NUMBER = 559
PREDICTION = 30

# MLFlow
PORT = 5000

models_dir = "./Models"
# models_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Models'
datasets_dir = "./Datasets"
# datasets_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Dataset'
logs_dir = "./Logs"
# logs_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Logs'

columns = [
    'year', 'month', 'day', 'hour', 'minute', 'second', 
    'glucose_level', 
    'finger_stick', 'basal', 'bolus', 'sleep', 'work', 'stressors', 'hypo_event', 'illness', 'exercise', 'basis_heart_rate',
    'basis_gsr', 'basis_skin_temperature', 'basis_air_temperature', 'basis_step', 'basis_sleep', 'meal', 'type_of_meal'
]

feature_columns = [
    'glucose_level',
    'basis_heart_rate',
    'basis_sleep',
    'basis_step',
    'basal',
    'finger_stick',
    'basis_skin_temperature',
    'basis_air_temperature',
    'work',
    'exercise',
    'bolus',
    'meal',
    'stressors',
    'illness',
    'type_of_meal',
    'hypo_event',
    'basis_gsr',
    'sleep'
]

target_columns = [
    'glucose_level'
]

################# MODEL #################
################ DATASET ################
number = DATASET_NUMBER
sequence_size = 12
prediction_time = int(PREDICTION / 5)
impute_strategy = False
scale_data = False
encode_categorical = False
select_features = False
validation_split = 0.2
############### TRAINING ################
learning_rate = 0.001
epochs = 40
optimizer = 'Adam'
loss = 'rmse'
batch_size = 64
##########################################


def train(model_name: str, model_version: int, log_level: int = logging.INFO, **model_parameters):
    log = structlog.get_logger()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))
    
    # Generate model name based on parameters and timestamp
    timestamp = int(time.time())
    model_name = f"{model_name}_{model_version}-{DATASET_NUMBER}-{PREDICTION}-[{timestamp}]"
    model_dir = f"{models_dir}/{model_name}"
    model_path = f"{model_dir}/{model_name}"
    log.info(f"Model name: {model_name}")
    
    ################# TRACKING #################
    try:
        # Connect to tracking server
        log.info("Connecting to tracking server")
        mlflow.set_tracking_uri(f"http://localhost:{PORT}")
        # Set experiment name and create experiment if it doesn't exist
        mlflow.set_experiment(f"GLUP24-{DATASET_NUMBER}-{PREDICTION}")
        log.debug(f"Experiment: {mlflow.get_experiment_by_name(f'GLUP24-{DATASET_NUMBER}-{PREDICTION}')}")
    except Exception as e:
        log.exception(f"Error connecting to tracking server", exec_info=e)
        return

    ################# DATASETS #################
    try:
        log.info("Reading datasets")
        # Get training and validation data
        train_val_data = pd.read_csv(f'{datasets_dir}/{number}/{number}_train.csv', sep=';',encoding = 'unicode_escape', names=columns)
        train_val_data = process_data(train_val_data, True, impute_strategy, scale_data, encode_categorical, select_features)
        log.debug(f"Training data info: {test_data.describe()}")
        # Get testing data
        test_data = pd.read_csv(f'{datasets_dir}/{number}/{number}_test.csv', sep=';', encoding = 'unicode_escape', names=columns)
        test_data = process_data(test_data, False, impute_strategy, scale_data, encode_categorical, select_features)
        log.debug(f"Testing data info: {test_data.describe()}")
    except Exception as e:
        log.exception(f"Error reading datasets", exec_info=e)
        return

    ################# PREPROCESSING #################
    try:
        log.info("Creating training and validation datasets")
        # Create sequences and targets for training
        X_train_val, Y_train_val = to_sequences_multi(train_val_data,  sequence_size, prediction_time, feature_columns, target_columns)
        # Split training and validation data
        x_train, x_val, y_train, y_val = train_test_split(X_train_val, Y_train_val, test_size=validation_split, random_state=42)
        # Create TensorFlow datasets for training
        train_dataset = (tf.data.Dataset
                        .from_tensor_slices((x_train, y_train))
                        .batch(batch_size)
                        .shuffle(len(x_train))
                        .prefetch(tf.data.experimental.AUTOTUNE)
                        # .filter(lambda x, y: tf.reduce_all(y != 0))
                    )
        log.debug(f"Training dataset size: {len(train_dataset)}")
        
        # Create TensorFlow datasets for validation
        validation_dataset = (tf.data.Dataset
                        .from_tensor_slices((x_val, y_val))
                        .batch(batch_size)
                        .prefetch(tf.data.experimental.AUTOTUNE)
                        # .filter(lambda x, y: tf.reduce_all(y != 0))
                    )
        log.debug(f"Validation dataset size: {len(validation_dataset)}")
    
        log.info("Creating testing dataset")
        # Create sequences and targets for testing
        X_test, Y_test = to_sequences_multi(test_data, sequence_size, prediction_time, feature_columns, target_columns)
        # Create test dataset
        test_dataset = (tf.data.Dataset
                        .from_tensor_slices((X_test, Y_test))
                        .batch(1)
                        .prefetch(tf.data.experimental.AUTOTUNE)
                        # .filter(lambda x, y: tf.reduce_all(y != 0))
                    )    
        log.debug(f"Test dataset size: {len(test_dataset)}")       
    except Exception as e:
        log.exception(f"Error creating datasets", exec_info=e)
        return
    
    ################# BUILD MODEL #################
    try:
        log.info("Building the model")
        # Get input shape from training data  NOTE: This can change from config if select_features == True
        input_shape = x_train.shape[1:]
        # Create the model with default parameters if there's a ValueError
        try:
            model_id = f"{model_name}_{model_version}"
            model = get_model(model_id, input_shape, 1,
                                ### LSTM ###
                                # hidden_units=32, embedding_size=32,
                                ### GRU ###
                                # hidden_units=32, embedding_size=32,
                                ### CNN ###
                                # filters=[32, 64, 128], kernel_size=3, hidden_units=128,
                                ### Deep Residual ###
                                # num_blocks=7, hidden_units=300, embedding_size=32, auxiliary_variables=4
                            )
            log.debug(f"Using model: {model}")
        except ValueError as e:
            log.warning(f"Invalid model: {model_id}, using defualt (LSTM_1)")
            model = get_model("LSTM_1", input_shape, 1)
        except Exception as e:
            log.exception(f"Error getting model", exec_info=e)
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
        log.exception(f"Error building model", exec_info=e)
        return
    
    try:
        with mlflow.start_run(run_name=model_name) as run:  
            # Define callbacks
            callbacks = [
                mlflow.keras.MLflowCallback(run),
                tf.keras.callbacks.TensorBoard(f"{logs_dir}/{model_name}"),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.2,),
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            ]
            ################# TRAIN #################
            # Train the model
            log.info("Starting training")
            start_time = time.time()
            model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks)
            log.debug(f"Training time: {time.time() - start_time}")     
        
            ################# TEST #################
            # Run test inference
            log.info("Running test inference")
            start_time = time.time()
            y_pred = model.predict(test_dataset)
            log.debug(f"Inference time: {time.time() - start_time}")
            
            ################# MODEL #################
            # Log an instance of the trained model for later use
            log.info("Logging trained model")
            mlflow.tensorflow.log_model(model, artifact_path="model")
            
            ############## PARAMETERS ##############
            log.info("Logging model parameters")
            # Create params for mlflow
            # params = {
            #     ### DATASET ###
            #     "sequence_size": sequence_size,
                
            #     ### TRAINING ###
            #     "optimizer": optimizer,
            #     "learning_rate": learning_rate,
            #     "loss": loss,
            #     "epochs": epochs,
            #     "batch_size": batch_size,
                
            #     ### LSTM ###
            #     # "hidden_units": hidden_units,
            #     # "embedding_size": embedding_size,
            #     ### GRU ### 
            #     # "hidden_units": hidden_units,
            #     # "embedding_size": embedding_size,
            #     ### CNN ###
            #     # "filters": filters,
            #     # "kernel_size": kernel_size,
            #     # "hidden_units": hidden_units,
            #     ### Deep Residual ###
            #     # "num_blocks": num_blocks,
            #     # "hidden_units": hidden_units,
            #     # "embedding_size": embedding_size,
            #     # "auxiliary_variables": auxiliary_variables,
            # }
        
        
            mlflow.log_params(model_parameters)
            log.debug(f"Model parameters: {model_parameters}")

            ################# METRICS #################
            log.info("Logging test metrics")
            # Calculate test metrics
            metrics = {
                "mae": mean_absolute_error(Y_test, y_pred), 
                "mse": mean_squared_error(Y_test, y_pred), 
                "rmse": np.sqrt(mean_squared_error(Y_test, y_pred)), 
                "r2": r2_score(Y_test, y_pred)
            }
            mlflow.log_metrics(metrics)
            log.debug(f"Test metrics: {metrics}")
            
            ################# CHARTS #################
            # Generate chart of real values vs prediction over the test data
            log.info("Generating charts")
            line_plot = generate_line_plot(Y_test, y_pred)
            mlflow.log_figure(line_plot, "real_vs_prediction.png")
            scatter_plot = generate_scatter_plot(Y_test, y_pred)
            mlflow.log_figure(scatter_plot, "scatter.png")
            histogram = generate_histogram_residuals(Y_test, y_pred)
            mlflow.log_figure(histogram, "histogram.png")
    except Exception as e:
        log.exception(f"Error training model", exec_info=e)
        return
    
    log.info("Model training complete")