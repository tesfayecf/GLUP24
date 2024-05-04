from dotenv import load_dotenv
load_dotenv()

import os
from train import train
from optuna import create_study, Trial


def objective(trial: Trial):
    model_name = trial.suggest_categorical("model_name", ["Transformer"])
    model_version = 1
    
    ### MODEL ###
    hidden_units = trial.suggest_categorical("hidden_units", [64, 128, 256, 512])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    num_heads = trial.suggest_categorical("num_heads", [32, 64, 128, 256])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.25, 0.3, 0.5])
    
    ### DATASET ###
    sequence_size = 12
    prediction_time = 6
    validation_split = 0.2
    create_timestamp = True
    create_fatigue = True
    scale_data = True
    perform_pca = True
    pca_components = trial.suggest_categorical("pca_components", [6, 7, 8, 9, 10])

    ### TRAINING ###
    learning_rate = trial.suggest_categorical("learning_rate", [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]) 
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "Adagrad", "Adadelta"])
    loss = trial.suggest_categorical("loss", ["mse", "mae"])
    batch_size = trial.suggest_int("batch_size", 32, 256, step=16)
    
    metric = train(
        model_name, model_version, 
        ### MODEL ###
        hidden_units=hidden_units, num_layers=num_layers, num_heads=num_heads, dropout_rate=dropout_rate,
        ### DATASET ###
        sequence_size=sequence_size, prediction_time=prediction_time, validation_split=validation_split, 
        create_timestamp=create_timestamp, create_fatigue=create_fatigue, scale_data=scale_data, perform_pca=perform_pca, pca_components=pca_components,
        ### TRAINING ###
        learning_rate=learning_rate, optimizer=optimizer, loss=loss, batch_size=batch_size
    )
    
    return metric


def optimize():
    study = create_study(direction="minimize",
                         load_if_exists=True,
                         study_name=f"{os.getenv('EXPERIMENT_NAME')}-{int(os.getenv('DATASET_NUMBER'))}-{int(os.getenv('PREDICTION'))}", 
                         storage=f"sqlite:///{os.getenv('EXPERIMENT_NAME')}-{int(os.getenv('DATASET_NUMBER'))}-{int(os.getenv('PREDICTION'))}.db")
    study.optimize(objective, n_trials=250, n_jobs=1, show_progress_bar=True)
    pass

if __name__ == "__main__":
    optimize()