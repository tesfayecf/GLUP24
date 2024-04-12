from dotenv import load_dotenv
load_dotenv()

from train import train
from test import test
from evaluate import evaluate

def train_model():
    model_name = "LSTM"
    model_version = 1
    
    hidden_units = 96
    embedding_size = 80
    sequence_size = 12
    
    learning_rate = 0.001
    epochs = 1
    optimizer = "Adam"
    loss = "mae"
    batch_size = 16
    
    metric = train(
        model_name, model_version, 
        ### MODEL ###
        hidden_units=hidden_units, embedding_size=embedding_size, 
        ### DATASET ###
        sequence_size=sequence_size,
        ### TRAINING ###
        learning_rate=learning_rate, optimizer=optimizer, loss=loss, batch_size=batch_size, epochs=epochs
    )
    
    return metric

def test_model():
    run_id = "e48c057ce2744afda3f64721b6a7301b"
    test(run_id)

def evaluate_model():
    run_id = "e48c057ce2744afda3f64721b6a7301b"
    evaluate(run_id)

if __name__ == "__main__":
    evaluate_model() 