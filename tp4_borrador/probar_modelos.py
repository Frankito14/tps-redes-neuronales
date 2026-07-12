import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
#Clases en el mismo roden que el de las carpetas
CLASES = ["Aves", "Caballos", "Gatos", "Hipopotamos","Perros","Pinguinos","Serpientes","Tortugas"]

print("Cargando modelo")

#Cargamos en "modelo" el que queramos utilizar
modelo=tf.keras.models.load_model("modelo_entrenado.keras")
#modelo=tf.keras.models.load_model("modelo_trasfer_base.keras")
#modelo=tf.keras.models.load_model("modelo_trasfer_finetuning.keras")

print("Modelo cargado")

def predecir_imagen(ruta_imagen,es_escala_grises=True):
    print(f"Foto seleccionada: {ruta_imagen}")

    img_para_mostrar = image.load_img(ruta_imagen)

    # Dependiendo de que modelo estemos usando uamos uno u otro
    modo_color = "grayscale" if es_escala_grises else "rgb"

    img_para_red = image.load_img(ruta_imagen,target_size=(512,512),color_mode=modo_color)
    img_array = image.img_to_array(img_para_red)

    #Agregamos la dimencion batch que exige keras, queda de (1,512,512,canales)
    img_array = np.expand_dims(img_array,axis=0)

    #Prediccion
    prediccion_one_hot = modelo.predict(img_array)

    #Buscamos la posicion de la neurona con mayor porcentaje
    indice_ganador = np.argmax(prediccion_one_hot[0])
    animal_predicho =  CLASES[indice_ganador]
    probabilidad = prediccion_one_hot[0][indice_ganador]*100

    #Mostramos por pantalla
    plt.fugure(figsize=(6,6))
    plt.imshow(img_para_mostrar)
    plt.title(f"Prediccion: {animal_predicho}\nSeguridad:{probabilidad:.2f}%",fontsize=14,fontweight="bold")
    plt.axis("off")
    plt.show()
ruta_de_prueba = "imagen_prueba_1.jpg"#Aca va la imagen que elegimos
predecir_imagen(ruta_de_prueba,es_escala_grises=True)
