from dotenv import load_dotenv
from optuna import create_study 
from train import train

load_dotenv()

def objective(trial):
    
    model_name = trial.suggest_categorical("model_name", ["LSTM"])
    
    model_version = trial.suggest_int("model_version", 1, 2, 3, 4)
    
    # hidden_units = trial.suggest_int("hidden_units", 32, 64, 96, 128)
    # embedding_size = trial.suggest_int("embedding_size", 16, 32, 64, 80)
    hidden_units = trial.suggest_int("hidden_units", 16, 128, step=16)
    embedding_size = trial.suggest_int("embedding_size", 16, 256, step=16)
    
    num_blocks_residual = trial.suggest_int("num_blocks_residual", 1, 2, 3, 4)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3, 0.5)
    conv_filters = trial.suggest_int("conv_filters", 32, 64, 96, 128)
    conv_kernel_size = trial.suggest_int("conv_kernel_size", 3, 5, 7)
    
    # sequence_size = trial.suggest_int("sequence_size", 12)
    sequence_size = trial.suggest_int("sequence_size", 6, 30, step=6)
    
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True) 
    # optimizer = trial.suggest_categorical("optimizer", ["Adam"])
    # loss = trial.suggest_categorical("loss", ["mse", "mae", "mape", "msle"])
    # batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "Adagrad", "Adadelta"])
    loss = trial.suggest_categorical("loss", ["mse", "mae", "mape", "msle"])
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    
    metric = train(
        model_name, model_version, 
        ### MODEL ###
        hidden_units=hidden_units, embedding_size=embedding_size, 
        num_blocks_residual=num_blocks_residual, dropout_rate=dropout_rate,
        conv_filters=conv_filters, conv_kernel_size=conv_kernel_size,
        ### DATASET ###
        sequence_size=sequence_size,
        ### TRAINING ###
        learning_rate=learning_rate, optimizer=optimizer, loss=loss, batch_size=batch_size
    )
    
    return metric


def main():
    study = create_study(direction="minimize", storage="sqlite:///hyperparameter_search_1.db")
    study.optimize(objective, n_trials=75)
    pass

if __name__ == "__main__":
    main()