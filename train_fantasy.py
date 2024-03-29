import os
import time
import mlflow
import logging
import structlog
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Other.utils import *
from Other.plot import *
from Models.model import get_model

DATASET_NUMBER = 559
PREDICTION = 30
HOST = "192.168.1.19"
PORT = 7777
EXPERIMENT_NAME = "FANTASY-ACC"
LOG_LEVEL = logging.DEBUG

datasets_dir = "./Fantasy"
# datasets_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Dataset'
logs_dir = "./Logs"
# logs_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Logs'

columns = [
    "PointsClass",

    "HomeAway",
    "PlayerPosition",
    "PlayerLastSeasonPoints",
    
    "MinutesPlayed",
    "Goals",
    "Assists",
    "OfftargetAttAssist",
    "PenAreaEntries",
    "PenaltyWon",
    "PenalytSave",
    "Saves",
    "EffectiveClearances",
    "PenaltyFailed",
    "OwnGoals",
    "GoalsConceded",
    "YellowCards",
    "SecondYellowCard",
    "RedCards",
    "TotalScoringAtt",
    "WonContest",
    "BallRecovery",
    "PossLostAll",
    "PenaltiesConceded",

    "Points",
    "GamesPlayed",
    "GoalsParticipation",

    # Add team stats
    "TeamGoals",
    "TeamAssists",
    "TeamOfftargetAttAssist",
    "TeamPenAreaEntries",
    "TeamPenaltyWon",
    "TeamPenaltySave",
    "TeamSaves",
    "TeamEffectiveClearances",
    "TeamPenaltyFailed",
    "TeamOwnGoals",
    "TeamGoalsConceded",
    "TeamYellowCards",
    "TeamSecondYellowCard",
    "TeamRedCards",
    "TeamTotalScoringAtt",
    "TeamWonContest",
    "TeamBallRecovery",
    "TeamPossLostAll",
    "TeamPenaltiesConceded",

    # Add opposing team stats
    "OpposingTeamGoals",
    "OpposingTeamAssists",
    "OpposingTeamOfftargetAttAssist",
    "OpposingTeamPenAreaEntries",
    "OpposingTeamPenaltyWon",
    "OpposingTeamPenaltySave",
    "OpposingTeamSaves",
    "OpposingTeamEffectiveClearances",
    "OpposingTeamPenaltyFailed",
    "OpposingTeamOwnGoals",
    "OpposingTeamGoalsConceded",
    "OpposingTeamYellowCards",
    "OpposingTeamSecondYellowCard",
    "OpposingTeamRedCards",
    "OpposingTeamTotalScoringAtt",
    "OpposingTeamWonContest",
    "OpposingTeamBallRecovery",
    "OpposingTeamPossLostAll",
    "OpposingTeamPenaltiesConceded",
]

feature_columns = [
    "HomeAway",
    "PlayerPosition",
    "PlayerLastSeasonPoints",
    
    "MinutesPlayed",
    "Goals",
    "Assists",
    "OfftargetAttAssist",
    "PenAreaEntries",
    "PenaltyWon",
    "PenalytSave",
    "Saves",
    "EffectiveClearances",
    "PenaltyFailed",
    "OwnGoals",
    "GoalsConceded",
    "YellowCards",
    "SecondYellowCard",
    "RedCards",
    "TotalScoringAtt",
    "WonContest",
    "BallRecovery",
    "PossLostAll",
    "PenaltiesConceded",

    "Points",
    "GamesPlayed",
    "GoalsParticipation",

    # Add team stats
    "TeamGoals",
    "TeamAssists",
    "TeamOfftargetAttAssist",
    "TeamPenAreaEntries",
    "TeamPenaltyWon",
    "TeamPenaltySave",
    "TeamSaves",
    "TeamEffectiveClearances",
    "TeamPenaltyFailed",
    "TeamOwnGoals",
    "TeamGoalsConceded",
    "TeamYellowCards",
    "TeamSecondYellowCard",
    "TeamRedCards",
    "TeamTotalScoringAtt",
    "TeamWonContest",
    "TeamBallRecovery",
    "TeamPossLostAll",
    "TeamPenaltiesConceded",

    # Add opposing team stats
    "OpposingTeamGoals",
    "OpposingTeamAssists",
    "OpposingTeamOfftargetAttAssist",
    "OpposingTeamPenAreaEntries",
    "OpposingTeamPenaltyWon",
    "OpposingTeamPenaltySave",
    "OpposingTeamSaves",
    "OpposingTeamEffectiveClearances",
    "OpposingTeamPenaltyFailed",
    "OpposingTeamOwnGoals",
    "OpposingTeamGoalsConceded",
    "OpposingTeamYellowCards",
    "OpposingTeamSecondYellowCard",
    "OpposingTeamRedCards",
    "OpposingTeamTotalScoringAtt",
    "OpposingTeamWonContest",
    "OpposingTeamBallRecovery",
    "OpposingTeamPossLostAll",
    "OpposingTeamPenaltiesConceded",
]

target_columns = [
    'PointsClass'
]

######### DEFAULT DATASET PARAMETERS #########
sequence_size_ = 1
prediction_time_ = 1
validation_split_ = 0.2
######### DEFAULT TRAINING PARAMETERS #########
learning_rate_ = 0.001
epochs_ = 50
optimizer_ = 'Adam'
loss_ = 'mse'
batch_size_ = 32
##############################################

def train_model(model_name: str, model_version: int, **parameters) -> float:
    log = structlog.get_logger()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVEL))
    # log = logging.getLogger(__name__)
    # logging.basicConfig(level=LOG_LEVEL)
    
    # Generate model name based on parameters and timestamp
    timestamp = int(time.time())
    run_name = f"{model_name}_{model_version}-[{timestamp}]"
    log.info(f"Model run name: {run_name}")
    
    # Get dataset parameters
    sequence_size = parameters.get('sequence_size', 1)
    prediction_time = parameters.get('prediction_time', 1)
    validation_split = parameters.get('validation_split', 0.2)
    # Get training parameters
    learning_rate = parameters.get('learning_rate', 0.001)
    epochs = parameters.get('epochs', 50)
    optimizer = parameters.get('optimizer', 'Adam')
    loss = parameters.get('loss', 'mse')
    batch_size = parameters.get('batch_size', 32)
    
    ################# TRACKING #################
    try:
        # Connect to tracking server
        log.info("Connecting to tracking server")
        mlflow.set_tracking_uri(f"http://{HOST}:{PORT}")
        # Set experiment name and create experiment if it doesn't exist
        mlflow.set_experiment(f"{EXPERIMENT_NAME}")
        log.debug(f"Experiment: {mlflow.get_experiment_by_name(f'{EXPERIMENT_NAME}')}")
    except Exception as e:
        log.exception(f"Error connecting to tracking server", exec_info=e)
        # log.exception(f"Error connecting to tracking server: {e}")
        return

    ################# DATASETS #################
    try:
        log.info("Reading datasets")
        # Get training and validation data
        train_val_test_data = pd.DataFrame()
        for index, sub_dataset in enumerate(os.listdir(datasets_dir)):
            sub = pd.read_csv(f"{datasets_dir}/{sub_dataset}", sep=',')
            train_val_test_data = pd.concat([train_val_test_data, sub], axis=0, ignore_index=True)
        log.debug(f"Data info: {train_val_test_data.describe()}")
    except Exception as e:
        log.exception(f"Error reading datasets", exec_info=e)
        # log.exception(f"Error reading datasets: {e}")
        return

    ################# PREPROCESSING #################
    try:
        log.info("Creating training and validation datasets")
        # Create sequences and targets for training
        X_train_val_test, Y_train_val_test = to_target(train_val_test_data, feature_columns, target_columns)
        # Split training, validation and test data
        x_train, x_val_test, y_train, y_val_test = train_test_split(X_train_val_test, Y_train_val_test, test_size=0.4, random_state=42)
        # Split validation and test data
        x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42)
        # Create TensorFlow datasets for training
        train_dataset = (tf.data.Dataset
                        .from_tensor_slices((x_train, y_train))
                        .batch(batch_size)
                        .shuffle(len(x_train))
                        .prefetch(tf.data.experimental.AUTOTUNE)
                        # .filter(lambda x, y: tf.reduce_all(y != 0))
                    )
        log.debug(f"Training dataset size: {len(x_train)}")
        
        # Create TensorFlow datasets for validation
        validation_dataset = (tf.data.Dataset
                        .from_tensor_slices((x_val, y_val))
                        .batch(batch_size)
                        .prefetch(tf.data.experimental.AUTOTUNE)
                        # .filter(lambda x, y: tf.reduce_all(y != 0))
                    )
        log.debug(f"Validation dataset size: {len(x_val)}")
    
        log.info("Creating testing dataset")
        # Create test dataset
        test_dataset = (tf.data.Dataset
                        .from_tensor_slices((x_test, y_test))
                        .batch(1)
                        .prefetch(tf.data.experimental.AUTOTUNE)
                        # .filter(lambda x, y: tf.reduce_all(y != 0))
                    )    
        log.debug(f"Test dataset size: {len(test_dataset)}")       
    except Exception as e:
        log.exception(f"Error creating datasets", exec_info=e)
        # log.exception(f"Error creating datasets: {e}")
        return
    
    ################# BUILD MODEL #################
    try:
        log.info("Building the model")
        # Get input shape from training data  NOTE: This can change from config if select_features == True
        input_shape = x_train.shape[1:]
        # Create the model with default parameters if there's a ValueError
        try:
            model_id = f"{model_name}_{model_version}"
            model = get_model(model_id, input_shape, 1, **parameters)                            
            log.debug(f"Using model: {model}")
        except ValueError as e:
            log.warning(f"Invalid model: {model_id}, using defualt (LSTM_1)")
            model = get_model("LSTM_1", input_shape, 1)
        except Exception as e:
            log.exception(f"Error getting model", exec_info=e)
            # log.exception(f"Error getting model: {e}")
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
        # log.exception(f"Error building model: {e}")
        return
    
    try:
        with mlflow.start_run(run_name=run_name) as run:  
            # Define callbacks
            callbacks = [
                mlflow.keras.MLflowCallback(run),
                tf.keras.callbacks.TensorBoard(f"{logs_dir}/{run_name}"),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.2),
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
            log.debug(f"Model parameters: {params}")

            ################# METRICS #################
            log.info("Logging test metrics")
            # Calculate test metrics
            metrics = {
                "mae": mean_absolute_error(y_test, y_pred), 
                "mse": mean_squared_error(y_test, y_pred), 
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)), 
                "r2": r2_score(y_test, y_pred)
            }
            mlflow.log_metrics(metrics)
            log.debug(f"Test metrics: {metrics}")
            
            ################# CHARTS #################
            # Generate chart of real values vs prediction over the test data
            log.info("Generating charts")
            line_plot = generate_line_plot(y_test, y_pred)
            mlflow.log_figure(line_plot, "real_vs_prediction.png")
            scatter_plot = generate_scatter_plot(y_test, y_pred)
            mlflow.log_figure(scatter_plot, "scatter.png")
            histogram = generate_histogram_residuals(y_test, y_pred)
            mlflow.log_figure(histogram, "histogram.png")
    except Exception as e:
        log.exception(f"Error training model", exec_info=e)
        # log.exception(f"Error training model: {e}")
        return
    
    log.info("Model training complete")
    return metrics['mae']