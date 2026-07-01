import os
from PIL import Image

def escalar_carpeta(carpeta_origen, carpeta_destino):
    # 1. Crear la carpeta de destino si aún no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        print(f"Carpeta creada: {carpeta_destino}")
        
    # Formatos de imagen que va a procesar
    extensiones_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # Contador para saber cuántas se procesaron
    contador = 0
    
    # 2. Recorrer los archivos de la carpeta origen
    for archivo in os.listdir(carpeta_origen):
        # Verificar si el archivo es una imagen válida
        if archivo.lower().endswith(extensiones_validas):
            ruta_entrada = os.path.join(carpeta_origen, archivo)
            ruta_salida = os.path.join(carpeta_destino, archivo)
            
            try:
                # 3. Abrir, redimensionar y guardar
                with Image.open(ruta_entrada) as img:
                    img_escalada = img.resize((512, 512), Image.Resampling.LANCZOS)
                    img_escalada.save(ruta_salida)
                    print(f"¡Éxito! -> {archivo} reescalado a 512x512")
                    contador += 1
            except Exception as e:
                print(f"No se pudo procesar el archivo {archivo}. Error: {e}")

    print(f"\n--- Proceso terminado. Se escalaron {contador} imágenes. ---")

# --- CONFIGURACIÓN DE RUTAS ---
# Reemplaza estas rutas con las tuyas locales
carpeta_de_entrada = "./a" 
carpeta_de_salida = "./dataset"

# Ejecutar la función
escalar_carpeta(carpeta_de_entrada, carpeta_de_salida)