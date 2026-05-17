import numpy as np
import random

#PERCEPTRON LINEAL


# Nombres de los archivos
path_entradas = 'entradas.txt'
path_salidas = 'salidas.txt'

entradas = []

# Leemos el archivo de entradas y convertimos cada línea en un np.array de flotantes

with open(path_entradas, 'r') as f:
    for linea in f:
        # Limpiamos espacios y dividimos la línea por columnas
        valores = linea.strip().split()
        if valores:
            # Convertimos cada valor a float y creamos el np.array
            array_data = np.array([float(v) for v in valores])
            array_data = np.append(array_data, 1.0)
            entradas.append(array_data)

ENTRADAS = entradas

#Cantidad de entradas.
P = len(ENTRADAS)

# Leemos el archivo de salidas esperadas y convertimos cada línea en un flotante
with open(path_salidas, 'r') as f:
    # Usamos una lista de comprensión para extraer los datos
    salidas = [float(linea.strip()) for linea in f if linea.strip()]

# Convertimos la lista completa en un solo np.array
ESPERADO = (np.array(salidas)).copy()

# Definimos el conjunto de entrenamiento y prueba
ENTRADAS_ENTRENAMIENTO = ENTRADAS[0:int(P*0.8)].copy()
ENTRADAS_PRUEBA = ENTRADAS[int(P*0.8):P].copy()

ESPERADO_ENTRENAMIENTO = ESPERADO[0:int(P*0.8)].copy()
ESPERADO_PRUEBA = ESPERADO[int(P*0.8):P].copy()

#print(ENTRADAS)

#Actualizamos la de entradas.
P = len(ENTRADAS_ENTRENAMIENTO)
#Tope de iteraciones.
COTA = 500
#Tasa de aprendizaje
N = 0.01#
#Numero de iteración
i = 0
#Pesos iniciales (al azar)
pesos = np.array([1, 1, 1, 1])
#Vector de pesos con error minimo registrado
w_min = pesos
#Inicializar error (se sobreescribe en la primera iteracion)
error = 1


def aproximacionDeError(entradas, esperado, pesos, p):#
    error = 0
    for i in range(p):
        exitacion = entradas[i] @ pesos
        obtenido = exitacion
        resultado = (esperado[i] - obtenido) ** 2
        error = error + resultado
    return error/2

error_min = aproximacionDeError(ENTRADAS_ENTRENAMIENTO, ESPERADO_ENTRENAMIENTO, pesos, P)

while(error > 0 and i < COTA):

    #Selecciona un indice al azar.
    index_random = random.randint(0, P-1)

    #Se toma un ejemplo al azar.
    ejemplo = ENTRADAS_ENTRENAMIENTO[index_random]

    #Se calcula la exitación del ejemplo tomado
    exitacion = ejemplo@pesos
    #Se calcula la salida con la exitacion obtenida
    obtenido = exitacion

    #Tomamos la salida esperada del ejemplo tomado
    esperado = ESPERADO_ENTRENAMIENTO[index_random]

    #Calculamos el delta w para saber que tanto "mover" w. 
    dw = N * (esperado - obtenido) * ejemplo
    
    #Calculamos el nuevo vector de pesos.
    pesos = pesos + dw

    #Calculamos el error para saber que tan buena es nuestra entrada.
    error = aproximacionDeError(ENTRADAS_ENTRENAMIENTO, ESPERADO_ENTRENAMIENTO, pesos, P)

    #print("Iteracion: "+ str(i))
    #print("W = "+ str(pesos))
    #print("Error actual: "+str(error)+" - Error minimo: "+str(error_min))

    #Si el error encontrado es menor al error minimo visto, entonces se guarda el nuevo error.
    #Y tambien actualizamos el vector de pesos con menor error encontrado.

    if error < error_min:
        error_min = error
        w_min = pesos.copy()

    i = i + 1

if(error == 0):
  print("Solucion encontrada")
else:
  print("Limite de iteraciones alcanzado")

print("Vector de pesos (w): "+str(pesos))
print("Error minimo: "+str(error_min))

print("\n--- FASE DE GENERALIZACIÓN (PRUEBA) ---")
print("Cantidad de ejemplos usados: " + str(len(ENTRADAS_PRUEBA)))

# Variables para acumular los errores
error_cuadratico_total = 0
porcentaje_errores = np.array([])
error_absoluto = 0

# Iteramos sobre cada ejemplo del conjunto de prueba
for i in range(len(ENTRADAS_PRUEBA)):
    
    # 1. Calculamos la predicción (Producto punto entre la entrada y los mejores pesos)
    prediccion = ENTRADAS_PRUEBA[i] @ w_min
    
    # 2. Tomamos el valor real que deberíamos haber obtenido
    esperado = ESPERADO_PRUEBA[i]
    
    # 3. Calculamos el error cuadrático de este ejemplo
    error_cuadratico_total += (esperado - prediccion) ** 2
    distancia_absoluta = abs(esperado - prediccion)
    error_absoluto += distancia_absoluta
    
    # 4. Calculamos el porcentaje de error (con protección por si el esperado es 0)
    if esperado != 0:
        porcentaje_error = abs(prediccion - esperado) / abs(esperado) * 100
    else:
        # Si el esperado es 0, tomamos el valor absoluto de la predicción como margen de error
        porcentaje_error = abs(prediccion) * 100 
        
    porcentaje_errores = np.append(porcentaje_errores, porcentaje_error)
    
    # Opcional: Descomenta la siguiente línea si quieres ver el detalle de cada predicción
    print(f"Ejemplo {i} | Esperado: {esperado:.4f} | Predicción: {prediccion:.4f} | Error: {distancia_absoluta:.4f} unidades | Error: {porcentaje_error:.2f}%")

# Calculamos las métricas finales
mse = error_cuadratico_total / len(ENTRADAS_PRUEBA)
mae = error_absoluto / len(ENTRADAS_PRUEBA)
error_porcentual_promedio = np.mean(porcentaje_errores)

print(f"Error Cuadrático Medio (MSE) en prueba: {mse:.4f}")
print(f"Error Absoluto Medio (MAE) en prueba: {mae:.4f}")
print(f"Porcentaje de error promedio: {error_porcentual_promedio:.2f}%")