import time
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import *
from model import build_model

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
model_name = 'LSTM'
hidden_units = 32
embedding_size = 32
################ DATASET ################
number = DATASET_NUMBER
sequence_size = 12
prediction_time = int(PREDICTION / 5)
impute_strategy = False
scale_data = False
encode_categorical = False
select_features = False
batch_size = 32
validation_split = 0.2
############### TRAINING ################
learning_rate = 0.001
epochs = 25
optimizer = 'Adam'
loss = 'mse'
##########################################


def main():
    # Generate model name based on parameters and timestamp
    timestamp = int(time.time())
    filename = f"{model_name}-{DATASET_NUMBER}-{PREDICTION}-[{timestamp}]"
    model_path = f"{models_dir}/{filename}"

    # Connect to tracking server
    print("Connecting to tracking server")
    mlflow.set_tracking_uri(f"http://localhost:{PORT}")
    mlflow.set_experiment(f"GLUP24-{DATASET_NUMBER}-{PREDICTION}")
    print(f"Experiment: {mlflow.get_experiment_by_name(f'GLUP24-{DATASET_NUMBER}-{PREDICTION}')}")

    ################# DATASETS #################
    print("Reading datasets")
    # Get training and validation data
    train_val_data = pd.read_csv(f'{datasets_dir}/{number}/{number}_train.csv', sep=';',encoding = 'unicode_escape', names=columns)
    train_val_data = process_data(train_val_data, True, impute_strategy, scale_data, encode_categorical, select_features)
    # Get testing data
    test_data = pd.read_csv(f'{datasets_dir}/{number}/{number}_test.csv', sep=';', encoding = 'unicode_escape', names=columns)
    test_data = process_data(test_data, False, impute_strategy, scale_data, encode_categorical, select_features)

    print("Creating training and validation datasets")
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
    # Create TensorFlow datasets for validation
    validation_dataset = (tf.data.Dataset
                    .from_tensor_slices((x_val, y_val))
                    .batch(batch_size)
                    .prefetch(tf.data.experimental.AUTOTUNE)
                    # .filter(lambda x, y: tf.reduce_all(y != 0))
                )
    
    print("Creating testing dataset")
    # Create sequences and targets for testing
    X_test, Y_test = to_sequences_multi(test_data,  sequence_size, prediction_time, feature_columns, target_columns)
    # Create test dataset
    test_dataset = (tf.data.Dataset
                    .from_tensor_slices((X_test, Y_test))
                    .batch(1)
                    .prefetch(tf.data.experimental.AUTOTUNE)
                    # .filter(lambda x, y: tf.reduce_all(y != 0))
                )           
    
    ################# BUILD MODEL #################
    print("Building the model")
    # Get input shape from training data  NOTE: This can change from config if select_features == True
    input_shape = x_train.shape[1:]
    # Create the model
    model = build_model(input_shape, hidden_units, embedding_size, 1)
    # Get optimizer
    optimizer_ = get_optimizer(optimizer, learning_rate)
    # Get loss
    loss_ = get_loss(loss)
    print("Compiling the model")
    # Compile model
    model.compile(
        optimizer=optimizer_, 
        loss=loss_, 
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
        ]
    )
    # Show model summary
    print("Model summary:")
    model.summary()
    
    ################# TRAIN #################
    with mlflow.start_run(run_name=filename) as run:
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.2,),
            tf.keras.callbacks.TensorBoard(f"{logs_dir}/{filename}"),
            mlflow.keras.MLflowCallback(run)
        ]
        
        # Train the model
        print("Training model")
        model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks)
        print("Model training completed succesfully")
        
        # Save the model after training (assuming training completes all epochs)
        model.save(f"{model_path}.keras")
        print("Trained model saved")
        
        # Crete summary json file
        create_summary_json(
            model_path, model_name, 
            # Dataset parameters
            number, feature_columns, target_columns, sequence_size, prediction_time, impute_strategy, scale_data, encode_categorical, select_features,
            # Trainin parameters
            optimizer, learning_rate, loss, epochs, batch_size,
            # Model parameters
            {
                "hidden_units": hidden_units,
                "embedding_size": embedding_size,
            }
        )       
    
        ################# TEST #################
        # Run test inference
        print("Running test inference")
        y_pred = model.predict(test_dataset)
        print("Test inference completed succesfully")
        
        ################# PARAMS #################
        print("Logging parameters of the model")
        # Create params for mlflow
        params = {
            "sequence_size": sequence_size,
            
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "loss": loss,
            "epochs": epochs,
            "batch_size": batch_size,
            
            "hidden_units": hidden_units,
            "embedding_size": embedding_size,
        }
    
        # Log the parameters used for the model fit
        mlflow.log_params(params)

        ################# METRICS #################
        print("Logging test metrics")
        # Calculate test metrics
        metrics = {
            "mae": mean_absolute_error(Y_test, y_pred), 
            "mse": mean_squared_error(Y_test, y_pred), 
            "rmse": np.sqrt(mean_squared_error(Y_test, y_pred)), 
            "r2": r2_score(Y_test, y_pred)
        }
        print(f"Test metrics: {metrics}")
         # Log the test metrics that were calculated during validation
        mlflow.log_metrics(metrics)
        
        ################# CHARTS #################
        # Generate chart of real values vs prediction over the test data
        print("Generating charts")
        line_plot = generate_line_plot(Y_test, y_pred)
        mlflow.log_figure(line_plot, "real_vs_prediction.png")
        scatter_plot = generate_scatter_plot(Y_test, y_pred)
        mlflow.log_figure(scatter_plot, "scatter.png")
        histogram = generate_histogram_residuals(Y_test, y_pred)
        mlflow.log_figure(histogram, "histogram.png")
        density = generate_density_residuals(Y_test, y_pred)
        mlflow.log_figure(density, "density.png")
        qq_plot = generate_qq_plot(Y_test, y_pred)
        mlflow.log_figure(qq_plot, "qq_plot.png")

        ################# MODEL #################
        # Log an instance of the trained model for later use
        print("Logging the trained model")
        mlflow.tensorflow.log_model(model, artifact_path="model")

if __name__ == '__main__':
    main()