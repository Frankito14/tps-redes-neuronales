import os
import tensorflow as tf
from keras import layers, models

DATASET_DIR = os.path.join(".", "dataset") 
IMG_SIZE = (512, 512)
BATCH_SIZE = 32
NUM_CLASSES = 3
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
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

model = models.Sequential([
    layers.Rescaling(1.0 / 255, input_shape=(*IMG_SIZE, 1)), # input de 512x512x1 en escala de grises
    
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)), # Reduce a 256x256
    
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)), # Reduce a 128x128
    
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)), # Reduce a 64x64

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)), # Reduce a 32x32
    
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)), # Reduce a 16x16
    
    layers.Flatten(),
    layers.Dropout(0.5), # obligamos a desactivarse al azar a la mitad de las neuronas para evitar sobreajuste
    layers.Dense(128, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax"), # activación softmax para generar salidas 0 o 1
])

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