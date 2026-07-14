import numpy as np
import tensorflow as tf
from keras import layers, models, regularizers
from config_2 import DIM_ENTRADA, DIM_LATENTE, X_ENTRENAMIENTO, EPOCAS, X_ENTRENAMIENTO_RUIDO



#Compresor (Encoder)

entradas_encoder = layers.Input(shape=(DIM_ENTRADA,), name="DIM_CHAR_ENCHORIZADO")

# 35N a 16N
x_enc = layers.Dense(16, activation='relu', name="35_A_16")(entradas_encoder)

# Latente: E35N -> 16N -> 8N 
salidas_encoder = layers.Dense(DIM_LATENTE, activation='linear', activity_regularizer=regularizers.l2(1e-4), name="ESPACIO_LATENTE_2D")(x_enc)

#tanh -> Dejaba las letras muy amontonadas en un borde (Poco espacio para crecer) -> Usamos linear (Mas valores para crecer)
#1e-4 -> 0.0001

# Encoder
encoder = models.Model(inputs=entradas_encoder, outputs=salidas_encoder, name="MODELO_ENCODER")

#Descompresor

# Entradas (Las 8 del espacio latente)
entradas_decoder = layers.Input(shape=(DIM_LATENTE,), name="SALIDA_LATENTE_ENTRADA")

# 8N -> 16N
x_dec = layers.Dense(16, activation='relu', name="8_A_16")(entradas_decoder)

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

# Juntar los modelos
autoencoder = models.Model(inputs=entradas_encoder, outputs=salidas_autoencoder, name="AUTOENCODER_COMPLETO")

# binary_crossentropy porque los píxeles son binarios (0 o 1)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='binary_crossentropy')

# Entrenamiento (Explotacion laboral)
# Entradas -> Ruido
# Salida esperada -> Limpio
print("Poniendo a laburar al autoencoder ruidoso...")
autoencoder.fit(X_ENTRENAMIENTO_RUIDO, X_ENTRENAMIENTO, epochs=EPOCAS, batch_size=4, verbose=0)
print("¡Jornada laborrrrrral completa!")

# GUARDAR EL ENTRENAMIENTO DE LOS MODELOS

encoder.save("encoder_2.keras")
decoder.save("decoder_2.keras")

print("Modelos guardados")



