"""
entrenamiento.py

Version con arquitectura CONVOLUCIONAL (Conv2D / Conv2DTranspose) en
lugar de capas Dense. Para imagenes, las convolucionales son mas
adecuadas porque aprovechan la estructura espacial (forma, bordes,
contornos) en lugar de tratar cada pixel como independiente.

Arquitectura:

    Entrada (40, 40, 1) - imagen normalizada a [0,1]
        |
        v
    ENCODER:
        Conv2D(16, 3x3, stride 2, relu)   -> (20, 20, 16)
        Conv2D(32, 3x3, stride 2, relu)   -> (10, 10, 32)
        Conv2D(64, 3x3, stride 2, relu)   -> ( 5,  5, 64)
        Flatten                            -> (1600,)
        Dense(2, linear)                   -> espacio latente (2,)
        |
        v
    DECODER:
        Dense(5*5*64, relu)
        Reshape                            -> ( 5,  5, 64)
        Conv2DTranspose(64, 3x3, stride 2, relu) -> (10, 10, 64)
        Conv2DTranspose(32, 3x3, stride 2, relu) -> (20, 20, 32)
        Conv2DTranspose(16, 3x3, stride 2, relu) -> (40, 40, 16)
        Conv2D(1, 3x3, sigmoid)                  -> (40, 40, 1)

40 se eligio porque es divisible por 8 (40 = 5 * 2^3), lo que permite
reducir la imagen con 3 capas de stride 2 hasta llegar a 5x5, y luego
revertir exactamente esa reduccion en el decoder.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from entradas import ENTRADAS

ANCHO = 40
ALTO = 40
DIM_LATENTE = 2
EPOCAS = 2000
RUTA_ENCODER = "encoder.keras"
RUTA_DECODER = "decoder.keras"
RUTA_AUTOENCODER = "autoencoder.keras"


def preparar_datos(entradas):
    """
    Convierte la lista de arrays (40,40) con valores 0-255 en un tensor
    numpy (N, 40, 40, 1) normalizado a [0,1], formato que esperan las
    capas Conv2D (alto, ancho, canales).
    """
    datos = np.array(entradas, dtype=np.float32)         # (N, 40, 40)
    datos = datos / 255.0                                  # normalizar a [0,1]
    datos = datos.reshape((datos.shape[0], ALTO, ANCHO, 1))  # agregar canal
    return datos


def crear_encoder():
    """
    Red codificadora convolucional: reduce la imagen 40x40x1 paso a paso
    (40 -> 20 -> 10 -> 5) extrayendo patrones espaciales, y finalmente
    comprime todo a un punto de 2 coordenadas en el espacio latente.
    """
    entrada = keras.Input(shape=(ALTO, ANCHO, 1), name="entrada_encoder")

    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(entrada)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)

    x = layers.Flatten()(x)
    salida_latente = layers.Dense(DIM_LATENTE, activation="linear",
                                   name="espacio_latente")(x)

    encoder = keras.Model(entrada, salida_latente, name="encoder")
    return encoder


def crear_decoder():
    """
    Red decodificadora convolucional: toma un punto de 2 coordenadas y
    lo expande paso a paso (5 -> 10 -> 20 -> 40) hasta reconstruir la
    imagen completa de 40x40x1.
    """
    entrada = keras.Input(shape=(DIM_LATENTE,), name="entrada_decoder")

    x = layers.Dense(5 * 5 * 64, activation="relu")(entrada)
    x = layers.Reshape((5, 5, 64))(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)

    salida = layers.Conv2D(1, 3, padding="same", activation="sigmoid",
                            name="salida_decoder")(x)

    decoder = keras.Model(entrada, salida, name="decoder")
    return decoder


def crear_autoencoder(encoder, decoder):
    """
    Conecta encoder + decoder en un solo modelo entrenable de punta a punta.
    """
    entrada = keras.Input(shape=(ALTO, ANCHO, 1), name="entrada_autoencoder")
    latente = encoder(entrada)
    salida = decoder(latente)

    autoencoder = keras.Model(entrada, salida, name="autoencoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
                         loss="mse")
    return autoencoder


def entrenar():
    datos = preparar_datos(ENTRADAS)
    print(f"Datos de entrenamiento: {datos.shape[0]} pokemones, "
          f"forma de cada uno: {datos.shape[1:]}")

    encoder = crear_encoder()
    decoder = crear_decoder()
    autoencoder = crear_autoencoder(encoder, decoder)

    encoder.summary()
    decoder.summary()
    autoencoder.summary()

    callback_early_stop = keras.callbacks.EarlyStopping(
        monitor="loss", patience=300, restore_best_weights=True
    )

    historia = autoencoder.fit(
        datos, datos,
        epochs=EPOCAS,
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