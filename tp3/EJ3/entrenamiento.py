"""
entrenamiento.py

Define el ENCODER y el DECODER de un autoencoder para los sprites de
pokemon (40x40, escala de grises), entrena el autoencoder completo con
los datos de ENTRADAS, y guarda los tres modelos (encoder, decoder,
autoencoder) en disco para que grafico.py pueda usarlos.

Arquitectura:

    Entrada (40x40 = 1600 valores, normalizados a [0,1])
        |
        v
    ENCODER:
        Dense(256, relu)
        Dense(64,  relu)
        Dense(2,   linear)   <-- espacio latente de 2 dimensiones
        |
        v
    DECODER:
        Dense(64,   relu)
        Dense(256,  relu)
        Dense(1600, sigmoid) --> reshape a (40,40)

Se elige un espacio latente de exactamente 2 dimensiones para poder
graficar directamente los puntos en el plano (x, y) sin tener que
aplicar PCA / t-SNE para reducir dimensionalidad.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from entradas import ENTRADAS

ANCHO = 40
ALTO = 40
PIXELES = ANCHO * ALTO
DIM_LATENTE = 2
EPOCAS = 5000

RUTA_ENCODER = "encoder.keras"
RUTA_DECODER = "decoder.keras"
RUTA_AUTOENCODER = "autoencoder.keras"


def preparar_datos(entradas):
    """
    Convierte la lista de arrays (40,40) con valores 0-255 en una matriz
    numpy (N, 1600) normalizada a [0, 1], lista para entrenar.
    """
    datos = np.array(entradas, dtype=np.float32)        # (N, 40, 40)
    datos = datos / 255.0                                 # normalizar a [0,1]
    datos = datos.reshape((datos.shape[0], PIXELES))       # (N, 1600)
    return datos


def crear_encoder():
    """
    Red codificadora: toma una imagen aplanada de 1600 valores y la
    comprime a un punto de 2 coordenadas en el espacio latente.
    """
    entrada = keras.Input(shape=(PIXELES,), name="entrada_encoder")
    x = layers.Dense(256, activation="relu")(entrada)
    x = layers.Dense(64, activation="relu")(x)
    salida_latente = layers.Dense(DIM_LATENTE, activation="linear",
                                   name="espacio_latente")(x)

    encoder = keras.Model(entrada, salida_latente, name="encoder")
    return encoder


def crear_decoder():
    """
    Red decodificadora: toma un punto de 2 coordenadas del espacio
    latente y reconstruye una imagen de 1600 valores (40x40 aplanada).
    """
    entrada = keras.Input(shape=(DIM_LATENTE,), name="entrada_decoder")
    x = layers.Dense(64, activation="relu")(entrada)
    x = layers.Dense(256, activation="relu")(x)
    salida = layers.Dense(PIXELES, activation="sigmoid",
                           name="salida_decoder")(x)

    decoder = keras.Model(entrada, salida, name="decoder")
    return decoder


def crear_autoencoder(encoder, decoder):
    """
    Conecta encoder + decoder en un solo modelo entrenable de punta a punta.
    """
    entrada = keras.Input(shape=(PIXELES,), name="entrada_autoencoder")
    latente = encoder(entrada)
    salida = decoder(latente)

    autoencoder = keras.Model(entrada, salida, name="autoencoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                         loss="mse")
    return autoencoder


def entrenar():
    datos = preparar_datos(ENTRADAS)
    print(f"Datos de entrenamiento: {datos.shape[0]} pokemones, "
          f"{datos.shape[1]} pixeles cada uno.")

    encoder = crear_encoder()
    decoder = crear_decoder()
    autoencoder = crear_autoencoder(encoder, decoder)

    encoder.summary()
    decoder.summary()
    autoencoder.summary()

    callback_early_stop = keras.callbacks.EarlyStopping(
        monitor="loss", patience=30, restore_best_weights=True
    )

    historia = autoencoder.fit(
        datos, datos,
        epochs=1500,
        batch_size=16,
        shuffle=True,
        verbose=2,
        callbacks=[callback_early_stop],
    )

    encoder.save(RUTA_ENCODER)
    decoder.save(RUTA_DECODER)
    autoencoder.save(RUTA_AUTOENCODER)

    print(f"\nModelos guardados: {RUTA_ENCODER}, {RUTA_DECODER}, {RUTA_AUTOENCODER}")
    print(f"Loss final (mse): {historia.history['loss'][-1]:.5f}")

    return encoder, decoder, autoencoder, historia


if __name__ == "__main__":
    entrenar()