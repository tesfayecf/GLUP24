from optuna import create_study 
from train import train_model

MODEL_NAME = "LSTM"
MODEL_VERSION = 1

def objective(trial):
    
    model_name = trial.suggest_categorical("model_name", ["LSTM", "GRU"])
    
    hidden_units = trial.suggest_int("hidden_units", 16, 128, step=16)
    embedding_size = trial.suggest_int("embedding_size", 16, 256, step=16)
    
    sequence_size = trial.suggest_int("sequence_size", 6, 30, step=6)
    
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True) 
    # epochs 
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "Adagrad", "Adadelta"])
    loss = trial.suggest_categorical("loss", ["mse", "mae", "mape", "msle"])
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    
    metric = train_model(
        MODEL_NAME, MODEL_VERSION, 
        ### MODEL ###
        hidden_units=hidden_units, embedding_size=embedding_size, 
        ### DATASET ###
        sequence_size=sequence_size,
        ### TRAINING ###
        learning_rate=learning_rate, optimizer=optimizer, loss=loss, batch_size=batch_size
    )
    
    return metric


def main():
    study = create_study(direction="minimize", storage="sqlite:///hyperparameter_search_1.db")
    study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed
    pass

if __name__ == "__main__":
    main()