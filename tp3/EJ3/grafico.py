import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

from entradas import ENTRADAS, NOMBRES
from entrenamiento import (
    preparar_datos,
    RUTA_ENCODER,
    RUTA_DECODER,
    ANCHO,
    ALTO,
)

print("Cargando modelos...")

encoder = tf.keras.models.load_model(RUTA_ENCODER)
decoder = tf.keras.models.load_model(RUTA_DECODER)

print("Modelos cargados")

X_ENTRENAMIENTO = np.array(ENTRADAS, dtype=np.float32) / 255.0
X_ENTRENAMIENTO = X_ENTRENAMIENTO.reshape((len(ENTRADAS), 40, 40, 1))

puntos_latentes = encoder.predict(X_ENTRENAMIENTO, verbose=0)

print("\nCoordenadas de los caracters:")
for i in range(len(ENTRADAS)):
    print(f"{NOMBRES[i]} -> X={puntos_latentes[i][0]:.2f}, Y={puntos_latentes[i][1]:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(puntos_latentes[:, 0], puntos_latentes[:, 1], color='red', s=100, zorder=5)

for i, txt in enumerate(NOMBRES):
    plt.annotate(txt, (puntos_latentes[i, 0], puntos_latentes[i, 1]), xytext=(5, 5), textcoords='offset points')

plt.title('Espacio Latente 2D de los caracteres (Autocoder)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()


