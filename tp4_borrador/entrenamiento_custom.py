import tensorflow as tf
from tensorflow.keras import layers, models

carpeta_datos = "./tp4_borrador/img"

alto_img = 512
ancho_img = 512

batch_size= 32

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

nombres_clases = set_entrenamiento.class_names
print(f"{nombres_clases}")

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

for imagenes, etiquetas in set_entrenamiento.take(1):
    print(f"Dimensiones del lote, {imagenes.shape}")
    print(f"Dimensiones de la etiqueta, {etiquetas.shape}")

    print(f"{etiquetas[0].numpy()}")
    break

modelo_custom = models.Sequential()
modelo_custom.add(layers.Rescaling(1./255, input_shape=(512,512,1)))

modelo_custom.add(layers.Conv2D(16, (3, 3), activation='relu'))
modelo_custom.add(layers.MaxPooling2D((2, 2)))

modelo_custom.add(layers.Conv2D(32, (3, 3), activation='relu'))
modelo_custom.add(layers.MaxPooling2D((2, 2)))

modelo_custom.add(layers.Flatten())
modelo_custom.add(layers.Dense(64, activation='relu'))

modelo_custom.add(layers.Dense(8, activation='softmax'))

modelo_custom.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

epocas = 10 

print("\nInicia el entrenamiento")
historial = modelo_custom.fit(
    set_entrenamiento,
    validation_data=set_prueba,
    epochs=epocas
)

modelo_custom.save("modelo_entrenado.keras")
print("\nListop")