import os
from PIL import Image

def convertir_carpeta_a_bn(carpeta_origen, carpeta_destino):
    # 1. Crear la carpeta de destino si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        print(f"Carpeta creada: {carpeta_destino}")
        
    extensiones_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    contador = 0
    
    # 2. Recorrer los archivos de la carpeta
    for archivo in os.listdir(carpeta_origen):
        if archivo.lower().endswith(extensiones_validas):
            ruta_entrada = os.path.join(carpeta_origen, archivo)
            ruta_salida = os.path.join(carpeta_destino, archivo)
            
            try:
                # 3. Abrir, convertir a escala de grises y guardar
                with Image.open(ruta_entrada) as img:
                    img_bn = img.convert('L')
                    img_bn.save(ruta_salida)
                    print(f"Convertida: {archivo} -> Blanco y Negro")
                    contador += 1
            except Exception as e:
                print(f"No se pudo procesar {archivo}. Error: {e}")

    print(f"\n--- Proceso terminado. Se convirtieron {contador} imágenes. ---")

# --- CONFIGURACIÓN DE RUTAS ---
carpeta_de_entrada = "dataset" 
carpeta_de_salida = "dataset/entrenamiento"

# Ejecutar el script
convertir_carpeta_a_bn(carpeta_de_entrada, carpeta_de_salida)