import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

# Set seaborn style and color palette
sns.set_style("darkgrid")
color_palette = sns.color_palette("muted")

def generate_line_plot(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_true, label='True values', color=color_palette[0])
    ax.plot(y_pred, label='Predictions', color=color_palette[1])
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Glucose Level', fontsize=14)
    ax.set_title('Real Values vs Predictions', fontsize=16)
    ax.legend(fontsize=12)
    sns.despine()
    return fig


def generate_scatter_plot(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, color=color_palette[0], ax=ax)
    ax.set_xlabel('True Values', fontsize=14)
    ax.set_ylabel('Predictions', fontsize=14)
    ax.set_title('Scatter Plot of True Values vs Predictions', fontsize=16)
    ax.plot([0, 500], [0, 500], color='red', linestyle='--', label='Perfect Fit')
    ax.legend(fontsize=12)
    sns.despine()
    return fig

def generate_histogram_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, bins=20, edgecolor='black', color=color_palette[0], ax=ax)
    ax.set_xlabel('Residuals', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Histogram of Residuals', fontsize=16)
    sns.despine()
    return fig

# def generate_qq_plot(y_true, y_pred):
#     residuals = y_true - y_pred
#     fig, ax = plt.subplots(figsize=(8, 6))
#     stats.probplot(residuals, dist="norm", plot=plt)
#     ax.set_xlabel('Theoretical Quantiles', fontsize=14)
#     ax.set_ylabel('Sample Quantiles', fontsize=14)
#     ax.set_title('QQ Plot', fontsize=16)
#     sns.despine()
#     return fig