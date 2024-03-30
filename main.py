from train import train_model

def main():
    model_name = "LSTM"
    model_version = 5
    
    hidden_units = 96
    embedding_size = 80
    num_residual_blocks = 2
    dropout_rate = 0.3
    conv_filters = 64
    conv_kernel_size = 3

    sequence_size = 12
    
    learning_rate = 0.001
    epochs = 100
    optimizer = "Adam"
    loss = "mae"
    batch_size = 128
    
    metric = train_model(
        model_name, model_version, 
        ### MODEL ###
        hidden_units=hidden_units, embedding_size=embedding_size, 
        num_residual_blocks=num_residual_blocks, dropout_rate=dropout_rate, 
        conv_filters=conv_filters, conv_kernel_size=conv_kernel_size,
        ### DATASET ###
        sequence_size=sequence_size,
        ### TRAINING ###
        learning_rate=learning_rate, optimizer=optimizer, loss=loss, batch_size=batch_size, epochs=epochs
    )
    
    return metric

if __name__ == "__main__":
    main()