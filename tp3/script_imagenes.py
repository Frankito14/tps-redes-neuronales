import os
from PIL import Image

def recortar_imagenes(carpeta_origen, carpeta_destino):
    # Crear la carpeta de destino si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
        print(f"Carpeta creada: {carpeta_destino}")

    # Listar todos los archivos de la carpeta de origen
    archivos = os.listdir(carpeta_origen)
    
    # Extensiones de imagen válidas
    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    for archivo in archivos:
        # Verificar que sea una imagen
        if archivo.lower().endswith(extensiones_validas):
            ruta_original = os.path.join(carpeta_origen, archivo)
            ruta_nueva = os.path.join(carpeta_destino, archivo)

            try:
                with Image.open(ruta_original) as img:
                    # El método crop pide una tupla: (izquierda, arriba, derecha, abajo)
                    # Para quedarte con los primeros 64x64 píxeles:
                    imagen_recortada = img.crop((0, 0, 64, 64))
                    
                    # Guardar la nueva imagen
                    imagen_recortada.save(ruta_nueva)
                    print(f"Procesada con éxito: {archivo}")
                    
            except Exception as e:
                print(f"No se pudo procesar {archivo}. Error: {e}")

# --- CONFIGURACIÓN DE RUTAS ---
# Cambia esto por las rutas de tus carpetas
CARPETA_ORIGEN = "Sprites_GEN_1"
CARPETA_DESTINO = "Sprites_GEN_1_Recortados"

# Ejecutar la función
recortar_imagenes(CARPETA_ORIGEN, CARPETA_DESTINO)