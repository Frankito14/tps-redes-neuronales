import os
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("modelo_entrenado.keras")

DATASET_DIR = os.path.join(".", "dataset") 
IMG_SIZE = (512, 512)
BATCH_SIZE = 32
NUM_CLASSES = 3
EPOCHS = 15

print("\n--- Evaluando predicciones en el conjunto de prueba ---")

buenas = 0
malas = 0

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123,
)

for images, labels in val_dataset:
    predicciones = model.predict(images, verbose=0)

    for i in range(len(images)):
        # Convertimos los vectores one-hot a índices enteros (ej: [0, 1, 0...] -> 1)
        clase_real_id = np.argmax(labels[i])
        clase_predicha_id = np.argmax(predicciones[i])
        
        # Traducimos el índice al nombre del animal
        nombre_real = val_dataset.class_names[clase_real_id]
        nombre_predicho = val_dataset.class_names[clase_predicha_id]
        
        if clase_real_id == clase_predicha_id:
            buenas += 1
            resultado = "✅ CORRECTO"
        else:
            malas += 1
            resultado = f"❌ INCORRECTO (Predijo: {nombre_predicho})"
            
        print(f"Imagen {buenas + malas}: Clase Real: {nombre_real} -> {resultado}")

total = buenas + malas
porcentaje_acierto = (buenas / total) * 100

print("\n========================================")
print("           RESUMEN DE TESTEO            ")
print("========================================")
print(f"Total de imágenes evaluadas: {total}")
print(f"Predicciones CORRECTAS:     {buenas}")
print(f"Predicciones INCORRECTAS:   {malas}")
print(f"Precisión final medida:     {porcentaje_acierto:.2f}%")
print("========================================")