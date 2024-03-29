from train_fantasy import train_model

def main():
    model_name = "LSTM"
    model_version = 1
    
    num_blocks = 3
    hidden_units = 32
    embedding_size = 32
    auxiliary_variables = 17
    
    sequence_size = 12
    
    learning_rate = 0.004
    # epochs 
    optimizer = "Adam"
    loss = "mae"
    batch_size = 16
    
    metric = train_model(
        model_name, model_version, 
        ### MODEL ###
        hidden_units=hidden_units, embedding_size=embedding_size,
        ### DATASET ###
        sequence_size=sequence_size,
        ### TRAINING ###
        learning_rate=learning_rate, optimizer=optimizer, loss=loss, batch_size=batch_size
    )
    
    return metric

if __name__ == "__main__":
    main()