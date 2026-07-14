import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Clases
CLASES = [
    "Aves", 
    "Caballos", 
    "Conejos", 
    "Gatos",
    "Hipopotamos",
    "Perros", 
    "Pinguinos", 
    "Serpientes", 
    "Tortugas"
]

# Rutas de los modelos entrenados y sus configuraciones
MODEL_CONFIGS = {
"Custom (Linear 40 Epocas)": {
        "ruta": "Modelos custom/modelo_entrenado_linear_40.keras", 
        "size": (512, 512),
        "color_mode": "grayscale", # Tu modelo custom usa 1 canal
        "preprocesamiento": "ninguno" # La capa Rescaling ya está DENTRO del modelo
    },
    "Custom (Linear 15 Epocas)": {
        "ruta": "Modelos custom/modelo_entrenado_linear.keras", 
        "size": (512, 512),
        "color_mode": "grayscale", # Tu modelo custom usa 1 canal
        "preprocesamiento": "ninguno" # La capa Rescaling ya está DENTRO del modelo
    },
    "Custom (M)": {
        "ruta": "Modelos custom/modelo_entrenado_m.keras", 
        "size": (512, 512),
        "color_mode": "grayscale", # Tu modelo custom usa 1 canal
        "preprocesamiento": "ninguno" # La capa Rescaling ya está DENTRO del modelo
    },
    "Custom (Tanh)": {
        "ruta": "Modelos custom/modelo_entrenado_tanh.keras", 
        "size": (512, 512),
        "color_mode": "grayscale",
        "preprocesamiento": "ninguno"
    },
    "Custom (XL)": {
        "ruta": "Modelos custom/modelo_entrenado_xl.keras", 
        "size": (512, 512),
        "color_mode": "grayscale", # Tu modelo custom usa 1 canal
        "preprocesamiento": "ninguno" # La capa Rescaling ya está DENTRO del modelo
    }
}

""",
    
    """
"""
    "Custom (XS)": {
        "ruta": "Modelos custom/modelo_entrenado_xs.keras", 
        "size": (512, 512),
        "color_mode": "grayscale", # Tu modelo custom usa 1 canal
        "preprocesamiento": "ninguno" # La capa Rescaling ya está DENTRO del modelo
    },

    
    "MobileNetV2 (Val Loss - Base)": {
        "ruta": "Modelos entrenados con EarlyStop/EarlyStop con _val_loss_/modelo_transfer_base.keras",
        "size": (512, 512), # Asumiendo que mantuviste el tamaño de 512x512 para el transfer learning
        "color_mode": "rgb", # MobileNetV2 requiere 3 canales (RGB)
        "preprocesamiento": "mobilenet_v2"
    },
    "MobileNetV2 (Val Loss - Finetuning)": {
        "ruta": "Modelos entrenados con EarlyStop/EarlyStop con _val_loss_/modelo_transfer_finetuning.keras",
        "size": (512, 512), # Asumiendo que mantuviste el tamaño de 512x512 para el transfer learning
        "color_mode": "rgb", # MobileNetV2 requiere 3 canales (RGB)
        "preprocesamiento": "mobilenet_v2"
    },
    "MobileNetV2 (EarlyStop 3 - Base)": {
        "ruta": "Modelos entrenados con EarlyStop/EarlyStop_3/modelo_transfer_base.keras",
        "size": (512, 512), # Asumiendo que mantuviste el tamaño de 512x512 para el transfer learning
        "color_mode": "rgb", # MobileNetV2 requiere 3 canales (RGB)
        "preprocesamiento": "mobilenet_v2"
    },
    "MobileNetV2 (EarlyStop 3 - Finetuning)": {
        "ruta": "Modelos entrenados con EarlyStop/EarlyStop_3/modelo_transfer_finetuning.keras",
        "size": (512, 512),
        "color_mode": "rgb",
        "preprocesamiento": "mobilenet_v2"
    },
    "MobileNetV2 (EarlyStop 10 - Base)": {
        "ruta": "Modelos entrenados con EarlyStop/EarlyStop_10/modelo_transfer_base.keras",
        "size": (512, 512), # Asumiendo que mantuviste el tamaño de 512x512 para el transfer learning
        "color_mode": "rgb", # MobileNetV2 requiere 3 canales (RGB)
        "preprocesamiento": "mobilenet_v2"
    },
    "MobileNetV2 (EarlyStop 10 - Finetuning)": {
        "ruta": "Modelos entrenados con EarlyStop/EarlyStop_10/modelo_transfer_finetuning.keras",
        "size": (512, 512),
        "color_mode": "rgb",
        "preprocesamiento": "mobilenet_v2"
    }
"""

def preprocesar_imagen(img_path, target_size, color_mode, tipo_prepro):
    # Cargar la imagen con el tamaño y modo de color (grises o rgb) objetivo
    img = tf.keras.utils.load_img(img_path, target_size=target_size, color_mode=color_mode)
    img_array = tf.keras.utils.img_to_array(img)
    
    # Expandir dimensiones (batch dimension): pasa a (1, alto, ancho, canales)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Aplicar el preprocesamiento matemático
    if tipo_prepro == "ninguno":
        # Retorna el array intacto (ideal si el modelo ya tiene layer de Rescaling)
        return img_batch
        
    elif tipo_prepro == "mobilenet_v2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
        
    else:
        # Escala clásica manual [0, 1] en caso de que haga falta
        return img_batch / 255.0

def cargar_modelos_disponibles(configs):
    """Carga los modelos que existan en el disco."""
    modelos_cargados = {}
    print("=== CARGANDO MODELOS ===")
    for nombre, config in configs.items():
        ruta = config["ruta"]
        if os.path.exists(ruta):
            try:
                print(f"Cargando {nombre}...")
                config["modelo"] = tf.keras.models.load_model(ruta)
                modelos_cargados[nombre] = config
                print(f"  [OK] Cargado exitosamente.")
            except Exception as e:
                print(f"  [ERROR] Al cargar '{nombre}': {e}")
        else:
            print(f"  [OMITIDO] Archivo no encontrado: '{ruta}'")
    print("========================\n")
    return modelos_cargados

def predecir_y_visualizar(img_path, modelos_cargados, clases=CLASES):
    if not os.path.exists(img_path):
        print(f"[ERROR] La imagen en '{img_path}' no existe.")
        return

    num_modelos = len(modelos_cargados)
    if num_modelos == 0:
        print("[ERROR] No hay ningún modelo disponible para predecir.")
        return

    fig, axes = plt.subplots(1, num_modelos + 1, figsize=(4 * (num_modelos + 1), 5))
    
    if num_modelos == 1:
        axes = [axes[0], axes[1]]

    # Mostrar la imagen original (siempre en RGB para que nosotros la veamos bien)
    try:
        img_original = tf.keras.utils.load_img(img_path)
        axes[0].imshow(img_original)
        axes[0].set_title("Imagen de Entrada", fontsize=12, fontweight='bold')
        axes[0].axis('off')
    except Exception as e:
        print(f"[ERROR] No se pudo visualizar la imagen original: {e}")
        return

    # Ejecutar inferencia en cada modelo
    for idx, (nombre, config) in enumerate(modelos_cargados.items()):
        modelo = config["modelo"]
        
        # Preprocesar según las reglas específicas de este modelo
        img_lista = preprocesar_imagen(
            img_path=img_path, 
            target_size=config["size"], 
            color_mode=config["color_mode"], 
            tipo_prepro=config["preprocesamiento"]
        )
        
        # Predicción (vector one-hot)
        predicciones = modelo.predict(img_lista, verbose=0)[0]
        
        clase_predicha_idx = np.argmax(predicciones)
        clase_predicha = clases[clase_predicha_idx]
        confianza = predicciones[clase_predicha_idx] * 100
        
        print(f"[{nombre}] -> {clase_predicha} ({confianza:.1f}%)")

        # Configurar el gráfico de barras
        ax = axes[idx + 1]
        y_pos = np.arange(len(clases))
        colores_barras = ['#2ca02c' if i == clase_predicha_idx else '#1f77b4' for i in range(len(clases))]
        
        ax.barh(y_pos, predicciones, color=colores_barras, edgecolor='black', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(clases, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Probabilidad', fontsize=10)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        ax.set_title(f"{nombre}\nPred: {clase_predicha}\nConf: {confianza:.1f}%", fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    modelos_activos = cargar_modelos_disponibles(MODEL_CONFIGS)
    ruta_imagen_prueba = "dataset_comparacion/tortuga.jpg" 
    predecir_y_visualizar(ruta_imagen_prueba, modelos_activos)