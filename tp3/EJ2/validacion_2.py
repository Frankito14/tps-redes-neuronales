import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config_2 import X_ENTRENAMIENTO, VALOR_CARACTER, X_ENTRENAMIENTO_RUIDO

def validar_caracter(entrada_ruidosa, prediccion_red, etiqueta):

    # Reformatear la entrada
    grilla_ruidosa = entrada_ruidosa.reshape(7, 5)
    
    # Pasar de signmoidea a 0 o 1 y reformatear
    grilla_limpiada = (prediccion_red > 0.5).astype(int).reshape(7, 5)
    
    print(f"CARÁCTER: '{etiqueta}'")
    
    # Mostrar Entrada con Ruido
    print("\nENTRADA CON RUIDO:")
    for fila in grilla_ruidosa:
        print(" ".join(["■" if pixel == 1.0 else "." for pixel in fila]))
            
    print("PREDICCION (SIN RUIDO):")
    for fila in grilla_limpiada:
        print(" ".join(["■" if pixel == 1 else "." for pixel in fila]))
        

print("Cargando modelos...")

encoder = tf.keras.models.load_model("encoder_2.keras")
decoder = tf.keras.models.load_model("decoder_2.keras")

print("Modelos cargados")

espacio_latente_ruidoso = encoder.predict(X_ENTRENAMIENTO_RUIDO, verbose=0)
predicciones_limpias = decoder.predict(espacio_latente_ruidoso, verbose=0)

for i in range(len(predicciones_limpias)):
    validar_caracter(
        entrada_ruidosa=X_ENTRENAMIENTO_RUIDO[i],
        prediccion_red=predicciones_limpias[i],
        etiqueta=VALOR_CARACTER[i]
    )