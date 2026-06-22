"""
decodificar_punto.py

Funcion que recibe un punto (x, y) del espacio latente, lo pasa por el
decoder entrenado, y muestra la imagen 40x40 resultante. Sirve para
explorar el espacio latente "a mano": probar distintas coordenadas y
ver que pokemon (real o inventado) le corresponde a cada una.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

ANCHO = 40
ALTO = 40


def decodificar_punto(decoder, x, y, mostrar=True, titulo=None, descripcion=""):
    """
    Recibe un punto (x, y) del espacio latente y devuelve la imagen
    40x40 (valores 0-255) que el decoder reconstruye a partir de ese
    punto. Si mostrar=True, tambien la grafica con matplotlib.
    """
    punto = np.array([[x, y]], dtype=np.float32)  # forma (1, 2)

    salida = decoder.predict(punto, verbose=0)[0]          # (40,40,1) en [0,1]
    imagen = (salida.reshape(ALTO, ANCHO) * 255.0).astype(np.uint8)

    if mostrar:
        plt.figure(figsize=(4, 4))
        plt.imshow(imagen, cmap="gray", vmin=0, vmax=255)
        plt.title(titulo or f"({x:.2f}, {y:.2f}) - {descripcion}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return imagen


def decodificar_varios_puntos(decoder, puntos_a_probar):
    """
    Recibe una lista de diccionarios con la forma:
        {"puntos": (x, y), "descripcion": "texto descriptivo"}
    y muestra todas las imagenes decodificadas en una sola fila, usando
    la descripcion como titulo de cada subplot. Sirve para documentar
    que representa cada punto que se decide explorar en el espacio
    latente (ej. "entre Metapod y Kakuna").
    """
    cantidad = len(puntos_a_probar)
    fig, axes = plt.subplots(1, cantidad, figsize=(cantidad * 2.5, 3))
 
    if cantidad == 1:
        axes = [axes]
 
    for i, item in enumerate(puntos_a_probar):
        x, y = item["puntos"]
        descripcion = item.get("descripcion", "")
 
        punto = np.array([[x, y]], dtype=np.float32)
        salida = decoder.predict(punto, verbose=0)[0]
        imagen = (salida.reshape(ALTO, ANCHO) * 255.0).astype(np.uint8)
 
        axes[i].imshow(imagen, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"{descripcion}\n({x:.1f}, {y:.1f})", fontsize=9)
        axes[i].axis("off")
 
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Cargando decoder...")
    decoder = tf.keras.models.load_model("decoder.keras")
    print("Decoder cargado")


    # Ejemplo: decodificar varios puntos a la vez, para explorar el espacio
    puntos_a_probar = [
        {
            "puntos": (8, 2.0),
            "descripcion": "Entre Mew y Horsea" 
        },
        {
            "puntos": (1, 15),
            "descripcion": "Entre Magnemite y Voltorb" 
        },
        {
            "puntos": (6, 13),
            "descripcion": "Entre Metapod y Kakuna" 
        },
        {
            "puntos": (7.5, -4),
            "descripcion": "Entre Squirtle y Diglett" 
        },
        {
            "puntos": (0, 0),
            "descripcion": "Abominacion" 
        },
        {
            "puntos": (2.5, -2),
            "descripcion": "Otra abominacion" 
        },
        {
            "puntos": (1.8, 0.7),
            "descripcion": "Venontat + Mew + Charmander" 
        }
      
    ]
    decodificar_varios_puntos(decoder, puntos_a_probar)