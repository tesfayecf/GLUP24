import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os
from sklearn import metrics

class CreateGifCallback(tf.keras.callbacks.Callback):
    """
    Callback personalitzat per generar gràfics durant l'entrenament i compilar-los en un GIF al final de l'entrenament.

    Args:
        x_test (ndarray): Dades de prova.
        y_test (ndarray): Etiquetes de prova.
        gif_path (str): Ruta on es guardarà el GIF.

    Attributes:
        x_test (ndarray): Dades de prova.
        y_test (ndarray): Etiquetes de prova.
        gif_path (str): Ruta on es guardarà el GIF.
        epoch (int): Nombre d'èpoques realitzades durant l'entrenament.
        images (list): Llista d'imatges per al GIF.
        temp_dir (str): Directori temporal per desar les imatges temporals.
    """
    def __init__(self, x_test, y_test, gif_path, show: bool = False):
        super(CreateGifCallback, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.gif_path = gif_path
        self.show = show
        self.epoch = 0
        self.images = []
        self.temp_dir = tempfile.mkdtemp()

    def on_epoch_end(self, epoch, logs=None):
        """
        Genera el gràfic de comparació entre les dades reals i les prediccions al final de cada època.

        Args:
            epoch (int): Nombre de l'època actual.
            logs (dict): Valors de les mètriques del model.
        """
        pred = self.model.predict(self.x_test)

        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(len(self.y_test)), self.y_test, label='Dades reals')
        plt.plot(np.arange(len(pred)), pred, label='Prediccions')
        plt.xlabel('Temps')
        plt.ylabel('Valor')
        plt.title('Comparació de les dades d\'entrenament, dades reals i prediccions')
        plt.legend()

        # Càlcul de l'error RMSE i MAE
        rmse = np.sqrt(metrics.mean_squared_error(pred, self.y_test))
        mae = metrics.mean_absolute_error(pred, self.y_test)

        # Anotació de l'RMSE i MAE al gràfic
        plt.text(0.8, 0.8, f'Epoch: {self.epoch:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}', transform=plt.gca().transAxes, horizontalalignment='center', fontsize=12)

        # Guardar el gràfic com a imatge
        filename = f"epoch_{self.epoch}.png"
        plt.savefig(os.path.join(self.temp_dir, filename))
        if self.show:
            plt.show()
        plt.close()
        self.epoch += 1

        # Afegir la imatge a la llista d'imatges
        self.images.append(Image.open(os.path.join(self.temp_dir, filename)))

    def on_train_end(self, logs=None):
        """
        Compila les imatges generades durant l'entrenament en un GIF.

        Args:
            logs (dict): Valors de les mètriques del model al final de l'entrenament.
        """
        # Guardar la llista d'imatges com a GIF
        self.images[0].save(self.gif_path, save_all=True, append_images=self.images[1:], optimize=False, duration=100, loop=0)
        # Eliminar els fitxers temporals d'imatges
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # Eliminar el directori temporal
        os.rmdir(self.temp_dir)
