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
HOST = "192.168.1.19"
PORT = 7777
EXPERIMENT_NAME = "GLUP24"
LOG_LEVEL = logging.INFO

datasets_dir = "./Datasets"
# datasets_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Dataset'
logs_dir = "./Logs"
# logs_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Logs'

def evaluate(model, mlflow_run=None, **parameters):
    """Evaluates the trained model on the test dataset and logs results to MLflow.

    Args:
        model: The trained TensorFlow model.
        mlflow_run: An existing MLflow run object (optional).

    Returns:
        A dictionary containing the calculated test metrics.
    """

    # log = structlog.get_logger()
    # structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVEL))
    log = logging.getLogger(__name__)
    logging.basicConfig(level=LOG_LEVEL)
    
    # Get dataset parameters
    sequence_size = parameters.get('sequence_size', 12)
    prediction_time = parameters.get('prediction_time', int(PREDICTION / 5))
    
    # ################# TRACKING #################
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
        # Get testing data
        test_data = pd.read_csv(f'{datasets_dir}/{DATASET_NUMBER}/{DATASET_NUMBER}_test.csv', sep=';', encoding = 'unicode_escape', names=columns)
        test_data = process_data(test_data, False)
        log.debug(f"Testing data info: {test_data.describe()}")
    except Exception as e:
        # log.exception(f"Error reading datasets", exec_info=e)
        log.exception(f"Error reading datasets: {e}")
        return

    ################# PREPROCESSING #################
    try:   
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
        # log.exception(f"Error creating datasets", exec_info=e)
        log.exception(f"Error creating datasets: {e}")
        return
    
    ################# EVALUATE MODEL #################
    try:
        log.info("Running model evaluation on test dataset")

        ################# EVALUATE #################
        # Run model inference on the test dataset
        y_pred = model.predict(test_dataset)

        ################# METRICS #################
        # Calculate evaluation metrics
        metrics = {
            "mae": mean_absolute_error(Y_test, y_pred),
            "mse": mean_squared_error(Y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(Y_test, y_pred)),
            "r2": r2_score(Y_test, y_pred)
        }
        log.info(f"Evaluation metrics: {metrics}")

        ################# CHARTS #################
        # Generate chart of real values vs prediction over the test data
        log.info("Generating charts")
        line_plot = generate_line_plot(Y_test, y_pred)
        mlflow.log_figure(line_plot, "test_real_vs_prediction.png")
        scatter_plot = generate_scatter_plot(Y_test, y_pred)
        mlflow.log_figure(scatter_plot, "test_scatter.png")
        histogram = generate_histogram_residuals(Y_test, y_pred)
        mlflow.log_figure(histogram, "test_histogram.png")
               
        # # Log metrics to MLflow
        # if mlflow_run is not None:
        #     mlflow.log_metrics(metrics, step=mlflow_run.info.run_id)  # Log to existing run
        # else:
        #     mlflow.log_metrics(metrics)  # Log to new automatic run

    except Exception as e:
        log.exception(f"Error evaluating model: {e}")

    return metrics
