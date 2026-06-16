import os
from PIL import Image

def convertir_a_blanco_y_negro(carpeta_entrada, carpeta_salida):
    """
    Toma todas las imágenes de carpeta_entrada, las convierte a escala de grises
    y las guarda en carpeta_salida.
    """
    # 1. Crear la carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
        print(f"Carpeta creada: {carpeta_salida}")

    # 2. Definir qué extensiones consideraremos como imágenes
    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    # 3. Recorrer todos los archivos en la carpeta de entrada
    archivos = os.listdir(carpeta_entrada)
    
    if not archivos:
        print("La carpeta de entrada está vacía o no existe.")
        return

    for nombre_archivo in archivos:
        # Ignorar archivos que no sean imágenes según su extensión
        if nombre_archivo.lower().endswith(extensiones_validas):
            ruta_entrada = os.path.join(carpeta_entrada, nombre_archivo)
            ruta_salida = os.path.join(carpeta_salida, nombre_archivo)

            try:
                # Abrir la imagen original
                with Image.open(ruta_entrada) as img:
                    # Convertir a blanco y negro ('L' significa "Luminance" o escala de grises)
                    img_bn = img.convert('L')
                    
                    # Guardar la nueva imagen
                    img_bn.save(ruta_salida)
                    print(f"✅ Convertida: {nombre_archivo}")
                    
            except Exception as e:
                print(f"❌ Error al procesar '{nombre_archivo}': {e}")

# --- Configuración de rutas ---
# Cambia estas variables por las rutas reales de tus carpetas.
# Puedes usar rutas absolutas (ej. 'C:/Usuarios/TuNombre/Imagenes') o relativas.
CARPETA_ORIGEN = 'Sprites_GEN_1_Recortados'
CARPETA_DESTINO = 'Sprites_BN'

# Ejecutar la función
if __name__ == '__main__':
    convertir_a_blanco_y_negro(CARPETA_ORIGEN, CARPETA_DESTINO)
    print("¡Proceso terminado!")