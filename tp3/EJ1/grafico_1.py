import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config_1 import X_ENTRENAMIENTO, VALOR_CARACTER

def mostrar_caracter():
    for fila in grilla_2d:
    # 1 = BLOQUE; 0 = .
        print(" ".join(["■" if pixel == 1 else "." for pixel in fila]))

print("Cargando modelos...")

encoder = tf.keras.models.load_model("encoder_1.keras")
decoder = tf.keras.models.load_model("decoder_1.keras")

print("Modelos cargados")

#Mostrar nuevo caracter:
#Coordanad random
coordenada_inventada = np.array([[1.0, -4.0]], dtype=np.float32)

pixel_generado = decoder.predict(coordenada_inventada, verbose=0)[0]

matriz_binaria = (pixel_generado > 0.5).astype(int)

print(f"Caracter nuevo: X={coordenada_inventada[0][0]}, Y={coordenada_inventada[0][1]}")
grilla_2d = matriz_binaria.reshape(7, 5)

for fila in grilla_2d:
    print(" ".join(["■" if pixel == 1 else "." for pixel in fila]))

# Obtener puntos latentes (Puntos en el plano)
puntos_latentes = encoder.predict(X_ENTRENAMIENTO, verbose=0)

print("\nCoordenadas de los caracters:")
for i in range(len(X_ENTRENAMIENTO)):
    print(f"{VALOR_CARACTER[i]} -> X={puntos_latentes[i][0]:.2f}, Y={puntos_latentes[i][1]:.2f}")

# 5. GENERAR EL GRÁFICO 2D
plt.figure(figsize=(8, 6))
# Graficamos los puntos usando la primera columna como X y la segunda como Y
plt.scatter(puntos_latentes[:, 0], puntos_latentes[:, 1], color='red', s=100, zorder=5)

# Le ponemos el nombre a cada punto en el gráfico
for i, txt in enumerate(VALOR_CARACTER):
    plt.annotate(txt, (puntos_latentes[i, 0], puntos_latentes[i, 1]), xytext=(5, 5), textcoords='offset points')

plt.title('Espacio Latente 2D de los caracteres (Autocoder)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()


