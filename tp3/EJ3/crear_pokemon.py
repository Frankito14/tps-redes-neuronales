import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Config
ANCHO = 40
ALTO = 40

def decodificar_varios_puntos(decoder, puntos_a_probar):
    
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