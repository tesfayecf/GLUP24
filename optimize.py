from dotenv import load_dotenv
load_dotenv()

from optuna import create_study, Trial
from train import train
import os


def objective(trial: Trial):
    
    model_name = trial.suggest_categorical("model_name", ["LSTM"])
    model_version = trial.suggest_categorical("model_version", [1, 2, 3, 4, 5])
    
    hidden_units = trial.suggest_int("hidden_units", 16, 256, step=16)
    embedding_size = trial.suggest_int("embedding_size", 16, 128, step=16)
    
    # num_blocks_residual = trial.suggest_categorical("num_blocks_residual", [1, 2, 3, 4])
    # dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    # conv_filters = trial.suggest_categorical("conv_filters", [32, 64, 96, 128])
    # conv_kernel_size = trial.suggest_categorical("conv_kernel_size", [3, 5, 7])
    
    sequence_size = 12
    prediction_time = 6
    validation_split = 0.2
    select_features = trial.suggest_categorical("select_features", [True, False])
    scale_data = trial.suggest_categorical("scale_data", [True, False])
    use_differences = trial.suggest_categorical("use_differences", [True, False])
    
    learning_rate = trial.suggest_categorical("learning_rate", [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]) 
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "Adagrad", "Adadelta"])
    loss = trial.suggest_categorical("loss", ["mse", "mae"])
    batch_size = trial.suggest_int("batch_size", 32, 256, step=16)
    
    metric = train(
        model_name, model_version, 
        ### MODEL ###
        hidden_units=hidden_units, embedding_size=embedding_size, 
        # num_blocks_residual=num_blocks_residual, dropout_rate=dropout_rate,
        # conv_filters=conv_filters, conv_kernel_size=conv_kernel_size,
        ### DATASET ###
        sequence_size=sequence_size, prediction_time=prediction_time, validation_split=validation_split, 
        select_features=select_features, scale_data=scale_data, use_differences=use_differences,
        ### TRAINING ###
        learning_rate=learning_rate, optimizer=optimizer, loss=loss, batch_size=batch_size
    )
    
    return metric


def optimize():
    study = create_study(direction="minimize",
                         load_if_exists=True,
                         study_name=f'{os.getenv('EXPERIMENT_NAME')}-{int(os.getenv('DATASET_NUMBER'))}-{int(os.getenv('PREDICTION'))}', 
                         storage=f'sqlite:///{os.getenv("EXPERIMENT_NAME")}-{int(os.getenv("DATASET_NUMBER"))}-{int(os.getenv("PREDICTION"))}.db')
    study.optimize(objective, n_trials=250, n_jobs=10, show_progress_bar=True)
    pass

if __name__ == "__main__":
    optimize()