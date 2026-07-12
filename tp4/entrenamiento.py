import os
import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMG_SIZE = (512, 512)
BATCH_SIZE = 32
NUM_CLASSES = 9
EPOCHS = 15

# Dataset de Entrenamiento
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",          # toma los nombres de las carpetas como etiquetas
    label_mode="categorical",   # salida one-hot
    color_mode="grayscale",     # escala de grises
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,       # divide el dataset en 80% entrenamiento y 20% validación
    subset="training",
    seed=123,
)

# 2. Dataset de Validación
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset="validation",        # toma el 20% restante para validación
    seed=123,
)

print("Clases detectadas:", train_dataset.class_names)

# Optimización de carga (cache hace que entrene mucho más rápido)
AUTOTUNE = tf.data.AUTOTUNE
#train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=100).cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Distintas configuraciones de la estructura de la red.

config_xs_agresiva = {
    "conv_blocks": [
        {"filters": 16, "kernel_size": (3, 3)}
    ],
    "dropout_rate": 0.5,
    "dense_units": 128
}

config_m_normal = {
    "conv_blocks": [
        {"filters": 16, "kernel_size": (3, 3)},
        {"filters": 32, "kernel_size": (3, 3)},
        {"filters": 64, "kernel_size": (3, 3)},
        {"filters": 128, "kernel_size": (3, 3)},
    ],
    "dropout_rate": 0.5,
    "dense_units": 128
}

# Captura rasgos mas generales en la primera capa 
config_xl_big_kernel = {
    "conv_blocks": [
        {"filters": 32, "kernel_size": (7, 7)},
        {"filters": 64, "kernel_size": (5, 5)},
        {"filters": 128, "kernel_size": (3, 3)}
    ],
    "dropout_rate": 0.5,
    "dense_units": 128
}


def modelo_dinamico(config):
    model_layers = [
        layers.Rescaling(1.0 / 255, input_shape=(*IMG_SIZE, 1)) # Capa de entrada fija con el reescalado
    ]
    
    # Añadir Conv2D de forma dinámica
    for block in config["conv_blocks"]:
        model_layers.append(
            layers.Conv2D(
                filters=block["filters"], 
                kernel_size=block["kernel_size"], 
                activation="relu"
            )
        )
    
        model_layers.append(layers.MaxPooling2D((2, 2))) #A la mitad es un estandar
            
    # Bloque de clasificación final
    model_layers.extend([
        layers.Flatten(),
        layers.Dropout(config["dropout_rate"]), # Porcentaje a desactivar al azar a la mitad de las neuronas para evitar sobreajuste
        layers.Dense(config["dense_units"], activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax") # activación softmax para generar salidas 0 o 1
    ])
    
    return models.Sequential(model_layers)


# CAMBIAR LA CONFIGURACION ACA
model = modelo_dinamico(config_m_normal)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy", # en la documentación de Keras recomiendan usar categorical_crossentropy para salidas tipo one-hot
    metrics=["accuracy"],
)

model.summary()

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
)

model.save("modelo_entrenado.keras")
print("Modelo guardado en modelo_entrenado.keras")