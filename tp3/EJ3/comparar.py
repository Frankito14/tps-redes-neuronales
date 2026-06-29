"""
comparar.py

Funcion inversa "conceptualmente" a crear_entrada_sprite: en lugar de
convertir una imagen en un array normalizado, toma un array (la salida
del autoencoder) y lo muestra como imagen, al lado del original, para
verificar visualmente si la reconstruccion se parece a la entrada real.

Sirve para validar que el autoencoder esta funcionando: si el sprite
reconstruido es muy distinto del original, algo anda mal (arquitectura
muy chica, pocas epocas, learning rate alto, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from entradas import ENTRADAS, NOMBRES

#Config
ANCHO = 40
ALTO = 40


def mostrar_varios_original_vs_reconstruido(autoencoder, cantidad=5, entradas=ENTRADAS, nombres=NOMBRES):
    cantidad = min(cantidad, len(entradas))
    fig, axes = plt.subplots(2, cantidad, figsize=(cantidad * 2, 4))
    for col in range(cantidad):
        original_0_255 = entradas[col]
        original_normalizado = original_0_255.astype(np.float32) / 255.0
        entrada_red = original_normalizado.reshape(1, ALTO, ANCHO, 1)

        salida_red = autoencoder.predict(entrada_red, verbose=0)[0]
        reconstruido_0_255 = (salida_red.reshape(ALTO, ANCHO) * 255.0).astype(np.uint8)

        nombre = nombres[col] if nombres else f"pokemon_{col}"

        axes[0, col].imshow(original_0_255, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(nombre, fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(reconstruido_0_255, cmap="gray", vmin=0, vmax=255)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruido")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    print("Cargando autoencoder...")
    autoencoder = tf.keras.models.load_model("autoencoder.keras")
    print("Autoencoder cargado")
    mostrar_varios_original_vs_reconstruido(autoencoder, cantidad=25)