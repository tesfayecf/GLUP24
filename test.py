import os
import json
import mlflow
import logging
import tempfile
import structlog
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error, median_absolute_error, 
    mean_squared_log_error, explained_variance_score
)
from Misc.utils import *
from Misc.plot import *
from Misc.columns import columns, target_columns

LOG_LEVEL = logging.INFO

######### ENVIRONMENT PARAMETERS #########
datasets_dir = os.getenv("DATASETS_DIR")
prediction_ = int(os.getenv("PREDICTION"))
host_ = os.getenv("MLFLOW_HOST")
port_ = int(os.getenv("MLFLOW_PORT"))

######### DEFAULT DATASET PARAMETERS #########
sequence_size_: int = 12
scale_data_: bool = False
select_features_: bool = False

def test(run_id: str):
    """Test the trained model on the test dataset.

    Args:
        run_id: The run identifier the evaluation is being made.
        
    Returns:
        A dictionary containing the calculated test metrics.
    """

    log = structlog.get_logger()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVEL))
    
    ################# SERVER #################
    try:
        # Connect to tracking server
        log.info("Connecting to tracking server")
        mlflow.set_tracking_uri(f"http://{host_}:{port_}")
    except Exception as e:
        log.exception(f"Error connecting to tracking server", exec_info=e)
        return
    
    ################# MODEL #################
    try:
        log.info("Getting model")
        # Get run
        run = mlflow.get_run(run_id)
        model_uri = run.info.artifact_uri + "/model/data/model.keras"
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download model to temporary directory
            mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=temp_dir)
            log.debug(f"Loading model from: {temp_dir}")
            # Load model
            download_path = os.path.join(temp_dir, "model.keras")
            model = tf.keras.models.load_model(download_path)
            log.debug(f"Model info: {model.summary()}")
    except mlflow.exceptions.MlflowException as e:
        log.exception(f"Error getting model from MLflow: {e}")
        raise
    except Exception as e:
        log.exception(f"Error loading model", exec_info=e)
        return
    
    ################# PARAMETERS #################
    try:
        log.info("Getting model parameters")
        # Get dataset number from run name
        run_name = run.info.run_name
        dataset_number = int(run_name.split("-")[1])
        prediction_ = int(run_name.split("-")[2])
        log.debug(f"Run name: {run_name}")
        # Get model parameters
        parameters_uri = run.info.artifact_uri + "/parameters.json"
        model_params = mlflow.artifacts.load_dict(parameters_uri)
        # Extract dataset parameters
        sequence_size = model_params.get("sequence_size", sequence_size_)
        prediction_time = model_params.get("prediction_time", int(prediction_ / 5))
        scale_data = model_params.get("scale_data", scale_data_)
        select_features = model_params.get("select_features", select_features_)
        log.debug(f"Model parameters: {model_params}")
    except Exception as e:
        log.exception(f"Error getting model parameters", exec_info=e)
        return

    ################# DATASETS #################
    try:
        log.info("Reading testing data")
        # Get testing data
        test_data = pd.read_csv(f'{datasets_dir}/{dataset_number}/{dataset_number}_test.csv', sep=';', encoding = 'unicode_escape', names=columns)
        test_data = process_data(test_data, scale_data=scale_data, select_features=select_features)
        log.debug(f"Testing data info: {test_data.describe()}")
    except Exception as e:
        log.exception(f"Error reading testing data", exec_info=e)
        return

    ################# PREPROCESSING #################
    try:   
        log.info("Creating testing dataset")
        # Create sequences and targets for testing
        X_test, Y_test = to_sequences_multi(test_data, sequence_size, prediction_time, target_columns)
        
        # Filter values that are 0 for test data
        X_test = X_test[Y_test[:,0] != 0]
        Y_test = Y_test[Y_test[:,0] != 0]
        
        # Create test dataset
        test_dataset = (tf.data.Dataset
                        .from_tensor_slices((X_test, Y_test))
                        .batch(1)
                        .prefetch(tf.data.experimental.AUTOTUNE)
                    )    
        
        log.debug(f"Test dataset size: {len(test_dataset)}")       
    except Exception as e:
        log.exception(f"Error creating testing dataset", exec_info=e)
        return
    
    ################# TEST MODEL #################
    try:
        log.info("Running model evaluation on test dataset")

        ################# TEST #################
        # Run model inference on the test dataset
        y_pred = model.predict(test_dataset)

        with mlflow.start_run(run_id=run_id) as log_run:
            ################# METRICS #################
            # Calculate evaluation metrics
            TEST_metrics = {
                "TEST_mae": mean_absolute_error(Y_test, y_pred), 
                "TEST_mse": mean_squared_error(Y_test, y_pred), 
                "TEST_rmse": np.sqrt(mean_squared_error(Y_test, y_pred)), 
                "TEST_r2": r2_score(Y_test, y_pred),
                "TEST_mape": mean_absolute_percentage_error(Y_test, y_pred),
                "TEST_medae": median_absolute_error(Y_test, y_pred),
                "TEST_explained_variance": explained_variance_score(Y_test, y_pred)
            }
            # Log metrics to existing MLflow run
            mlflow.log_metrics(TEST_metrics, run_id=run_id)  
            log.debug(f"Test metrics: {TEST_metrics}")

            ################# CHARTS #################
            # Generate chart of real values vs prediction over the test data
            log.info("Generating charts")
            line_plot = get_line_plot(Y_test, y_pred)
            mlflow.log_figure(line_plot, "TEST_real_vs_prediction.png")
            scatter_plot = get_scatter_plot(Y_test, y_pred)
            mlflow.log_figure(scatter_plot, "TEST_scatter.png")
            histogram = get_histogram_residuals(Y_test, y_pred)
            mlflow.log_figure(histogram, "TEST_histogram.png")
            residual_plot = get_residual_plot(Y_test, y_pred)
            mlflow.log_figure(residual_plot, "TEST_residual.png")                       

    except Exception as e:
        log.exception(f"Error evaluating model: {e}")

    return TEST_metrics
