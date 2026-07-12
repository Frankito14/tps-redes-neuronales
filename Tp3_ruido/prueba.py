import numpy as np
import tensorflow as tf
from config import *

print("Cargando modelos...")

encoder = tf.keras.models.load_model("encoder_2.keras")
decoder = tf.keras.models.load_model("decoder_2.keras")

indice = 2
letra_limpia = X_ENTRENAMIENTO[indice:indice+1]
etiqueta = VALOR_CARACTER[indice]

niveles_ruido = [5,10,15,20,30]

print(f"Prueba para letra {etiqueta.upper()}")

for ruido in niveles_ruido:
    print(f"Nivel de ruido {ruido}%")
    # Se aplica el ruido del nivel dado
    entrada_rota = entradas_con_ruido(letra_limpia,probabilidad_ruido=ruido/100)
    # modelo que entrenamos
    latente = encoder.predict(entrada_rota,verbose=0)
    salida_red = decoder.predict(latente,verbose=0)
    # Los ponemos en 7x5
    grilla_rota = entrada_rota[0].reshape(7,5)
    grilla_limpia = (salida_red[0] > 0.5).astype(int).reshape(7, 5)
    #A consola

    print("Entrada Ruidosa              Red Reconstruida")
    for fila_rota,fila_limpia in zip(grilla_rota,grilla_limpia):
        str_rota = " ".join(["■" if p == 1.0 else "." for p in fila_rota])
        str_limpia = " ".join(["■" if p == 1 else "." for p in fila_limpia])
        print(f"{str_rota}              {str_limpia}")


