from train import train_model

MODEL_NAME = "LSTM"
MODEL_VERSION = 1

################# PARAMETERS #################
hidden_units: int = 32
embedding_size: int = 32

def main():
    train_model(MODEL_NAME, MODEL_VERSION, hidden_units=hidden_units, embedding_size=embedding_size)
    return 0

if __name__ == "__main__":
    main()