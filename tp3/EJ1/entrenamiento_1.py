import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
from config_1 import DIM_ENTRADA, DIM_LATENTE, X_ENTRENAMIENTO, EPOCAS


#Compresor (Encoder)

entradas_encoder = layers.Input(shape=(DIM_ENTRADA,), name="DIM_CHAR_ENCHORIZADO")

# 35N a 16N
x_enc = layers.Dense(16, activation='relu', name="35_A_16")(entradas_encoder)

# 16N a 8N
x_enc = layers.Dense(8, activation='relu', name="16_A_8")(x_enc)

# 8N a 4N
x_enc = layers.Dense(4, activation='relu', name="8_A_4")(x_enc)

# Latente: E35N -> 16N -> 8N -> 4N -> 2N
salidas_encoder = layers.Dense(DIM_LATENTE, activation='linear', activity_regularizer=regularizers.l2(1e-4), name="ESPACIO_LATENTE_2D")(x_enc)

#tanh -> Dejaba las letras muy amontonadas en un borde (Poco espacio para crecer) -> Usamos linear (Mas valores para crecer)
#1e-4 -> 0.0001

# Encoder
encoder = models.Model(inputs=entradas_encoder, outputs=salidas_encoder, name="MODELO_ENCODER")

#Descompresor

# Entradas (Las 2 del espacio latente)
entradas_decoder = layers.Input(shape=(DIM_LATENTE,), name="SALIDA_LATENTE_ENTRADA")

# 2N -> 4N
x_dec = layers.Dense(4, activation='relu', name="2_A_4")(entradas_decoder)

# 4N -> 8N
x_dec = layers.Dense(8, activation='relu', name="4_A_8")(x_dec)

# 8N -> 16N
x_dec = layers.Dense(16, activation='relu', name="8_A_16")(x_dec)

# 16N -> 35N
x_dec = layers.Dense(35, activation='relu', name="16_A_35")(x_dec)

# DECODER: 2N -> 4N -> 8N -> 16N -> 35N
# 'sigmoid' = valores entre 0 y 1
salidas_decoder = layers.Dense(DIM_ENTRADA, activation='sigmoid', name="SALIDA_DECODER")(x_dec)

# Creamos el modelo del Decompresor
decoder = models.Model(inputs=entradas_decoder, outputs=salidas_decoder, name="MODELO_DECODER")

# Unir Compresor y Descompresor

# Conectamos la salida del encoder como la entrada del decoder
salidas_autoencoder = decoder(encoder(entradas_encoder))

# Creamos el modelo final que junta ambas partes
autoencoder = models.Model(inputs=entradas_encoder, outputs=salidas_autoencoder, name="AUTOENCODER_COMPLETO")

# Compilamos usando binary_crossentropy porque los píxeles son binarios (0 o 1)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='binary_crossentropy')

# Entrenamiento (Explotacion laboral)
print("Poniendo a laburar al encoder...")
autoencoder.fit(X_ENTRENAMIENTO, X_ENTRENAMIENTO, epochs=EPOCAS, batch_size=4, verbose=0)
print("¡Jornada laboral completa!")

# GUARDAR EL ENTRENAMIENTO DE LOS MODELOS

encoder.save("encoder_1.keras")
decoder.save("decoder_1.keras")

print("Modelos guardados")

