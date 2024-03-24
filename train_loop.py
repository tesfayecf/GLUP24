import tensorflow as tf

def train_loop(model, train_dataset, validation_dataset, test_dataset=None, epochs=1, steps_per_epoch=1, callbacks=None):
    """
    Trains the model using a custom training loop with early stopping (optional).

    Args:
        model: An instance of the Model class.
        train_dataset: A TensorFlow Dataset for training data.
        validation_dataset: A TensorFlow Dataset for validation data (optional).
        epochs: Number of training epochs (default 1).
        steps_per_epoch: Number of training steps per epoch (optional).
        callbacks: A list of Keras callbacks to use during training (optional).
    """
    
     # Initialize callbacks
    callback_list = tf.keras.callbacks.CallbackList(callbacks, model=model)
        
    # Training loop
    for epoch in range(epochs):
        
        # Call callbacks before epoch
        callback_list.on_epoch_begin(epoch)

        train_data_subset = train_dataset.take(steps_per_epoch)
        for step, data in enumerate(train_data_subset):
            x, y = data
            # Call callbacks before training step
            callback_list.on_train_batch_begin(step)
            # Run train step
            model.train_step(x)
            # Call callbacks after training step
            callback_list.on_train_batch_end(step)

        # Evaluate after each epoch (if validation data provided)
        val_data_subset = validation_dataset.take(1)
        for step, data in enumerate(val_data_subset):
            val_loss, val_metrics = model.evaluate(data)                
            print(f"Epoch: {epoch+1}/{epochs} - Train Loss: {model.loss.result().numpy():.4f}, Val Loss: {val_loss:.4f}")
            for metric, value in zip(model.metrics, val_metrics):
                print(f"\t- {metric}: {value:.4f}")
        
        # Evaluate on test dataset after each epoch (if test data provided)
        if test_dataset:
            test_data_subset = test_dataset.take(1)
            for step, data in enumerate(test_data_subset):
                test_loss, test_metrics = model.evaluate(data)
                print(f"Test Loss after Epoch {epoch+1}: {test_loss:.4f}")
                for metric, value in zip(model.metrics, test_metrics):
                    print(f"\t- {metric}: {value:.4f}")
        
        # Call callbacks after epoch
        callback_list.on_epoch_end(epoch)