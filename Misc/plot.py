import numpy as np
import matplotlib.pyplot as plt

# Set matplolib style
plt.style.use("seaborn-v0_8")

# Set a more modern and minimalistic color palette
color_palette = ["#1E88E5", "#ED2F7E", "#EAED2F", "#F44336"]

# Set figure size
fig_size = (10, 6)

def line_plot(y_true, y_pred, show=False):
   fig = plt.figure(figsize=fig_size)
   plt.plot(y_true, label="Valor real", color=color_palette[0])
   plt.plot(y_pred, label="Predicció", color=color_palette[1])
   plt.xlabel("Temps", fontsize=14)
   plt.ylabel("Nivell de glucosa", fontsize=14)
   plt.title("Valor real vs Predicció", fontsize=16)
   plt.legend(fontsize=12)
   plt.grid(axis="y")
   plt.tight_layout()
   if show:
      plt.show()
   else:
      plt.close(fig)
   return fig

def scatter_plot(y_true, y_pred):
   y_true = np.ravel(y_true)
   y_pred = np.ravel(y_pred)
   fig = plt.figure(figsize=fig_size)
   plt.scatter(y_true, y_pred, alpha=0.5, color=color_palette[0], marker="o", s=50)
   plt.xlabel("Valor real", fontsize=14)
   plt.ylabel("Predicció", fontsize=14)
   plt.title("Valor real vs Predicció", fontsize=16)
   plt.plot([0, 500], [0, 500], color="grey", linestyle="--", alpha=0.5, label="Perfect Fit")
   plt.legend(fontsize=12)
   plt.xlim(0, 500)
   plt.ylim(0, 500)
   plt.grid(axis="x")
   plt.tight_layout()
   plt.close(fig)
   return fig

def histogram_residuals_plot(y_true, y_pred):
   residuals = y_pred - y_true
   fig = plt.figure(figsize=fig_size)
   plt.hist(residuals, bins=50, color=color_palette[0])
   plt.xlabel("Residus", fontsize=14)
   plt.ylabel("Frequüència", fontsize=14)
   plt.title("Histograms dels residus", fontsize=16)
   plt.grid(axis="y")
   plt.tight_layout()
   plt.close(fig)
   return fig