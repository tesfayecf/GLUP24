import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import *
from model import build_model

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

DATASET_NUMBER = 559
PREDICTION = 30

# MlFlow
port = 8080

model_dir = "./Models"
# model_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Models'
dataset_dir = "./Dataset"
# dataset_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Dataset'
log_dir = "./Logs"
# log_dir = '/content/drive/MyDrive/UNIVERSITAT/Inteligencia artificial/Treball/Logs'

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
sequence_size = 10
prediction_time = int(PREDICTION / 5)
impute_strategy = False
scale_data = False
encode_categorical = False
select_features = False
batch_size = 32
validation_split = 0.2
############### TRAINING ################
learning_rate = 0.001
epochs = 5
optimizer = 'Adam'
loss = 'mse'
##########################################

# Get current timestamp
timestamp = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
filename = f"{model_name}_{DATASET_NUMBER}-{PREDICTION}_{timestamp}"
filepath = f"{model_dir}/{filename}"

def main():
    # Start tracking server
    mlflow.set_tracking_uri(f"http://localhost:{port}")
    mlflow.set_experiment(f"GLUP24-{DATASET_NUMBER}-{PREDICTION}")

    # Get training data
    train_val_data = pd.read_csv(f'{dataset_dir}/{number}/{number}_train.csv', sep=';',encoding = 'unicode_escape', names=columns)
    train_val_data = process_data(train_val_data, True, impute_strategy, scale_data, encode_categorical, select_features)
    
    # Get validation data
    test_data = pd.read_csv(f'{dataset_dir}/{number}/{number}_test.csv', sep=';', encoding = 'unicode_escape', names=columns)
    test_data = process_data(test_data, False, impute_strategy, scale_data, encode_categorical, select_features)

    # Create sequences and targets for training
    X, Y = to_sequences_multi(train_val_data,  sequence_size, prediction_time, feature_columns, target_columns)
    
    # Create train and validation data
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=validation_split, random_state=42)
    
    # Create train and validation datasets
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
    
    # Create sequences and targets for testing
    X_test, Y_test = to_sequences_multi(test_data,  sequence_size, prediction_time, feature_columns, target_columns)
    
    # Create test dataset
    test_dataset = (tf.data.Dataset
                    .from_tensor_slices((X_test, Y_test))
                    .batch(1)
                    .prefetch(tf.data.experimental.AUTOTUNE)
                    # .filter(lambda x, y: tf.reduce_all(y != 0))
                )           
    
    # Get input shape from training data
    input_shape = x_train.shape[1:]
    # Create the model
    model = build_model(input_shape, hidden_units, embedding_size, 1)
    # Get optimizer
    optimizer_ = get_optimizer(optimizer, learning_rate)
    # Get loss
    loss_ = get_loss(loss)
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
    model.summary()
    
    with mlflow.start_run(run_name=filename) as run:
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.2,),
            tf.keras.callbacks.TensorBoard(f"{log_dir}/{filename}"),
            mlflow.keras.MLflowCallback(run)
        ]
        
        # Train the model
        model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=callbacks)
    
        # Save the model after training (assuming training completes all epochs)
        model.save(f"{filepath}")
        
        # Crete summary json file
        create_summary_json(
            filepath, model_name, 
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
    
        # Run test inference
        y_pred = model.predict(test_dataset)
        
        # Calculate error metrics
        metrics = {
            "mae": mean_absolute_error(Y_test, y_pred), 
            "mse": mean_squared_error(Y_test, y_pred), 
            "rmse": np.sqrt(mean_squared_error(Y_test, y_pred)), 
            "r2": r2_score(Y_test, y_pred)
        }
        
        print(f"Error metrics: {metrics}")
        
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

        # Log the error metrics that were calculated during validation
        mlflow.log_metrics(metrics)

        # Log an instance of the trained model for later use
        mlflow.tensorflow.log_model(model, artifact_path="model")

if __name__ == '__main__':
    main()