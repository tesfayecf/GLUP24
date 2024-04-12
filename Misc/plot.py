import numpy as np
import matplotlib.pyplot as plt

# Set a more modern and minimalistic color palette
color_palette = ["#1E88E5", "#ED2F7E", "#EAED2F", "#F44336"]

# Set figure size
fig_size = (10, 6)

def get_line_plot(y_true, y_pred):
   fig = plt.figure(figsize=fig_size)  # Create figure object
   plt.plot(y_true, label="True values", color=color_palette[0])
   plt.plot(y_pred, label="Predictions", color=color_palette[1])
   plt.xlabel("Time", fontsize=14)
   plt.ylabel("Glucose Level", fontsize=14)
   plt.title("Real Values vs Predictions", fontsize=16)
   plt.legend(fontsize=12)
   plt.grid(axis="y")
   plt.tight_layout()
   plt.close(fig)  # Close the figure to prevent display
   return fig

def get_scatter_plot(y_true, y_pred):
   y_true = np.ravel(y_true)
   y_pred = np.ravel(y_pred)
   fig = plt.figure(figsize=fig_size)  # Create figure object
   plt.scatter(y_true, y_pred, alpha=0.5, color=color_palette[0], marker="o", s=50)
   plt.xlabel("True Values", fontsize=14)
   plt.ylabel("Predictions", fontsize=14)
   plt.title("Scatter Plot of True Values vs Predictions", fontsize=16)
   plt.plot([0, 500], [0, 500], color="grey", linestyle="--", alpha=0.5, label="Perfect Fit")
   plt.legend(fontsize=12)
   plt.xlim(0, 500)
   plt.ylim(0, 500)
   plt.grid(axis="x")
   plt.tight_layout()
   plt.close(fig)  # Close the figure to prevent display
   return fig

def get_histogram_residuals(y_true, y_pred):
   residuals = y_pred - y_true
   fig = plt.figure(figsize=fig_size)  # Create figure object
   plt.hist(residuals, bins=50, color=color_palette[0])
   plt.xlabel("Residuals", fontsize=14)
   plt.ylabel("Frequency", fontsize=14)
   plt.title("Histogram of Residuals", fontsize=16)
   plt.grid(axis="y")
   plt.tight_layout()
   plt.close(fig)  # Close the figure to prevent display
   return fig

def get_residual_plot(y_true, y_pred):
    residuals = y_pred - y_true
    fig = plt.figure(figsize=fig_size)  # Create figure object
    plt.scatter(y_pred, residuals, alpha=0.5, color=color_palette[0], marker="o", s=50)
    plt.xlabel("Predicted Values", fontsize=14)
    plt.ylabel("Residuals", fontsize=14)
    plt.title("Residual Plot", fontsize=16)
    plt.axhline(y=0, color="grey", linestyle="--", alpha=0.5, label="Zero Line")
    plt.legend(fontsize=12)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.close(fig)  # Close the figure to prevent display
    return fig