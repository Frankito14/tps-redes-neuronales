import tensorflow as tf
from tensorflow.keras import layers, models
import json

carpeta_datos = "./tp4_borrador/img"
alto_img = 512
ancho_img = 512
batch_size= 32


print("\n--- LEYENDO CONFIGURACIÓN JSON ---")
# 1. Abrimos el archivo y cargamos las variables
with open("config.json", "r") as archivo:
    config = json.load(archivo)

print(f"Iniciando experimento: {config['nombre_prueba']}")
print(f"Cantidad de capas convolucionales: {len(config['bloques_convolucionales'])}")

# 2. Inicializacion del modelo
modelo_custom = models.Sequential()

# Normalización 
modelo_custom.add(layers.Rescaling(1./255, input_shape=(512, 512, 1)))

# 3. Armador
# Esto lee el JSON y crea las capas
for capa in config["bloques_convolucionales"]:
    filtros = capa["filtros"]
    modelo_custom.add(layers.Conv2D(filtros, (3, 3), activation='relu'))
    modelo_custom.add(layers.MaxPooling2D((2, 2)))

# 4.
modelo_custom.add(layers.Flatten())

# Cantidad de capas densas
modelo_custom.add(layers.Dense(config["neuronas_densas"], activation='relu'))

# Salida de 8 neuronas
modelo_custom.add(layers.Dense(8, activation='softmax'))

#Compila y usa el lr del json
optimizador = tf.keras.optimizers.Adam(learning_rate=config["tasa_aprendizaje"])

modelo_custom.compile(
    optimizer=optimizador,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Datos que quedan lindos
modelo_custom.summary()



set_entrenamiento = tf.keras.utils.image_dataset_from_directory(
    carpeta_datos,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(alto_img,ancho_img),
    batch_size=batch_size,
    label_mode="categorical",
    color_mode="grayscale"
)

set_prueba = tf.keras.utils.image_dataset_from_directory(
    carpeta_datos,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(alto_img,ancho_img),
    batch_size=batch_size,
    label_mode="categorical",
    color_mode="grayscale"

)

print("\nInicia el entrenamiento")
historial = modelo_custom.fit(
    set_entrenamiento,
    validation_data=set_prueba,
    epochs=config["epocas"]
)

modelo_custom.save("modelo_entrenado.keras")
print("\nListop")
