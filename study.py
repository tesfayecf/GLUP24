# import optuna
# from optuna.samplers import TPESampler

# def objective(trial):
#     # Define the hyperparameters to be optimized
#     hidden_units = trial.suggest_int("hidden_units", 16, 128)
#     embedding_size = trial.suggest_int("embedding_size", 16, 128)

#     # Call your training function with the suggested hyperparameters
#     mae, mse, rmse, r2 = train_function(
#         hidden_units=hidden_units,
#         embedding_size=embedding_size,
#         # Pass other required arguments
#     )

#     # Return the metric you want to optimize (e.g., rmse)
#     return rmse

# # Create a study object
# study = optuna.create_study(
#     study_name="lstm_hyperparameter_optimization",
#     sampler=TPESampler(seed=42),  # Specify the sampler if needed
#     direction="minimize",  # "minimize" or "maximize" based on your objective
# )

# # Run the optimization
# study.optimize(objective, n_trials=100)  # Adjust the number of trials as needed

# # Get the best hyperparameters
# best_trial = study.best_trial
# best_hidden_units = best_trial.params["hidden_units"]
# best_embedding_size = best_trial.params["embedding_size"]
# print(f"Best hidden units: {best_hidden_units}")
# print(f"Best embedding size: {best_embedding_size}")

# with mlflow.start_run() as run:
#     # Log the best hyperparameters
#     mlflow.log_param("best_hidden_units", best_hidden_units)
#     mlflow.log_param("best_embedding_size", best_embedding_size)

#     # Log the best metric
#     best_metric_value = best_trial.value
#     mlflow.log_metric("best_rmse", best_metric_value)

#     # Log any other relevant information
#     # ...