"""
grafico.py

1) Carga el encoder entrenado y proyecta todos los pokemones de ENTRADAS
   al espacio latente de 2 dimensiones, graficando cada uno como un punto
   (x, y) en el plano. Cada punto se etiqueta/colorea segun su "forma"
   (en el caso de datos reales, se puede colorear por tipo de pokemon).

2) Usa el decoder entrenado para generar un pokemon nuevo: se elige un
   punto del espacio latente que NO corresponde a ningun punto de
   entrenamiento (por ejemplo, un punto intermedio entre dos clusters,
   o un punto fuera de la nube de puntos conocida) y se reconstruye la
   imagen 40x40 correspondiente, mostrando que la red puede generar
   sprites "nuevos" que no estaban en el conjunto de entrenamiento.
"""

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
X_ENTRENAMIENTO = X_ENTRENAMIENTO.reshape((len(ENTRADAS), 40 * 40))

# Obtener puntos latentes (Puntos en el plano)
puntos_latentes = encoder.predict(X_ENTRENAMIENTO, verbose=0)

print("\nCoordenadas de los caracters:")
for i in range(len(ENTRADAS)):
    print(f"{NOMBRES[i]} -> X={puntos_latentes[i][0]:.2f}, Y={puntos_latentes[i][1]:.2f}")

# 5. GENERAR EL GRÁFICO 2D
plt.figure(figsize=(8, 6))
# Graficamos los puntos usando la primera columna como X y la segunda como Y
plt.scatter(puntos_latentes[:, 0], puntos_latentes[:, 1], color='red', s=100, zorder=5)

# Le ponemos el nombre a cada punto en el gráfico
for i, txt in enumerate(NOMBRES):
    plt.annotate(txt, (puntos_latentes[i, 0], puntos_latentes[i, 1]), xytext=(5, 5), textcoords='offset points')

plt.title('Espacio Latente 2D de los caracteres (Autocoder)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()


