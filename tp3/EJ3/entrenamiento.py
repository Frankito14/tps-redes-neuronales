import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from entradas import ENTRADAS

#CONFIG
ANCHO = 40
ALTO = 40
DIM_LATENTE = 2
EPOCAS = 2000
RUTA_ENCODER = "encoder.keras"
RUTA_DECODER = "decoder.keras"
RUTA_AUTOENCODER = "autoencoder.keras"

def preparar_datos(entradas):
    #Formatear las entradas para mandarlas a la red.
    datos = np.array(entradas, dtype=np.float32)         # (N, 40, 40)
    datos = datos / 255.0                                  # normalizar a [0,1]
    datos = datos.reshape((datos.shape[0], ALTO, ANCHO, 1))  # agregar canal
    return datos


def crear_encoder():
    entrada = keras.Input(shape=(ALTO, ANCHO, 1), name="entrada_encoder")

    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(entrada)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x) #Enchorizar
    salida_latente = layers.Dense(DIM_LATENTE, activation="linear",name="espacio_latente")(x)
    encoder = keras.Model(entrada, salida_latente, name="encoder")
    return encoder


def crear_decoder():

    entrada = keras.Input(shape=(DIM_LATENTE,), name="entrada_decoder")

    x = layers.Dense(5 * 5 * 64, activation="relu")(entrada)  #Reconstruir de 2 a 1600
    x = layers.Reshape((5, 5, 64))(x) #Desenchorizar

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)

    salida = layers.Conv2D(1, 3, padding="same", activation="sigmoid",name="salida_decoder")(x)
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
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),loss="mse")
    return autoencoder


def entrenar():
    datos = preparar_datos(ENTRADAS)

    encoder = crear_encoder()
    decoder = crear_decoder()
    autoencoder = crear_autoencoder(encoder, decoder)

    encoder.summary()
    decoder.summary()
    autoencoder.summary()

    callback_early_stop = keras.callbacks.EarlyStopping(
        monitor="loss", 
        patience=300, #Si en 300 epocas no mejora, corta el entrenamiento
        restore_best_weights=True
    )

    historia = autoencoder.fit(
        datos, datos,
        epochs=EPOCAS,
        batch_size=16, #Lotes de procesmiento
        shuffle=True,
        verbose=2,
        callbacks=[callback_early_stop], #Funcion para saber si cortar antes o no.
    )

    encoder.save(RUTA_ENCODER)
    decoder.save(RUTA_DECODER)
    autoencoder.save(RUTA_AUTOENCODER)

    print(f"Modelos guardados: {RUTA_ENCODER}, {RUTA_DECODER}, {RUTA_AUTOENCODER}")
    print(f"Loss final (mse): {historia.history['loss'][-1]:.5f}")

    return encoder, decoder, autoencoder, historia


if __name__ == "__main__": #Ejecutar solo si se ejecuta este script directamente, no si se importa.
    entrenar()