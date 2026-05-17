import numpy as np
import random

entradas_i = []

# Leemos el archivo de entradas y convertimos cada línea en un np.array de flotantes

with open('entradas.txt', 'r') as f:
    for linea in f:
        # Limpiamos espacios y dividimos la línea por columnas
        valores = linea.strip().split()
        if valores:
            # Convertimos cada valor a float y creamos el np.array
            array_data = np.array([float(v) for v in valores])
            array_data = np.append(array_data, 1.0)
            entradas_i.append(array_data)

entradas= entradas_i.copy()

# Leemos el archivo de salidas esperadas y convertimos cada línea en un flotante
with open('salidas.txt', 'r') as f:
    # Usamos una lista de comprensión para extraer los datos
    salidas = [float(linea.strip()) for linea in f if linea.strip()]

# Convertimos la lista completa en un solo np.array
salidas_esperadas = (np.array(salidas)).copy()

#cantidad de entradas
cant_entradas=len(entradas)

###########################################################################################################

"""
# 1. Mezclar y Dividir
indices = np.arange(cant_entradas)
np.random.shuffle(indices)

entradas_mezcladas = np.array(entradas)[indices]
salidas_mezcladas_originales = np.array(salidas_esperadas)[indices]
"""
limite = int(cant_entradas * 0.8)

entradas_train = entradas[:limite]
entradas_test = entradas[limite:]

salidas_train_originales = salidas_esperadas[:limite]
salidas_test_originales = salidas_esperadas[limite:]

# 2. Calcular s_min y s_max SOLO con los datos de entrenamiento
s_min = np.min(salidas_train_originales)
s_max = np.max(salidas_train_originales)

# 3. Escalar ambos conjuntos usando esos parámetros
salidas_train = (salidas_train_originales - s_min) / (s_max - s_min)
salidas_test = (salidas_test_originales - s_min) / (s_max - s_min)


###########################################################################################################

#Pesos iniciales
pesos = np.array([0.0,0.0,0.0,0.0])
pesos_min = np.array([0.0,0.0,0.0,0.0])

#learning rate
learning_rate = 0.1

#indice inical
i = 0

#error inicial que se actualiza
error_actual = 1

#error minimo detectado
error_min = cant_entradas*2

#Numero de veces maximo que se ejecuta el algoritmo
cota=500

#b
b=1

#Funcion logistica
def g(excitacion):
    return 1/(1+np.exp(-2*b*excitacion))

#Funcion logistica derivada
def g_der(excitacion):
    x=g(excitacion)
    return 2*b*x*(1-x)
#
def calcularExcitacion(fila_entrada,pesos):
    return fila_entrada @ pesos

#Cambio en la activacion al ser no lineal
def calcularActivacion(excitacion):
    return g(excitacion)

def calcular_error(entradas_x,salidas_x,pesos_actuales):
    error=0
    for i in range(len(entradas_x)):
        fila_actual = entradas_x[i]
        excitacion=calcularExcitacion(fila_actual,pesos_actuales)
        activacion=calcularActivacion(excitacion)
        resultado=(salidas_x[i] - activacion) ** 2
        error = error + resultado
    return error/2

######################################### PERCEPTRON ENTRENANDO ####################################################

#bucle de aprendizaje
while error_actual > 0 and i < cota:
    indice_random= random.randint(0,len(entradas_train)-1)
    fila_entrada = entradas_train[indice_random]
    salida_esperada = salidas_train[indice_random]

    #calculo de la excitacion=producto interno
    excitacion=calcularExcitacion(fila_entrada,pesos)

    #La activacion puede ser cualquier numero
    activacion=calcularActivacion(excitacion)

    #calcula diferencia de pesos(Deberia agregarse la derivada de g() pero como es 1 se puede obviar)
    dif_pesos = learning_rate * (salida_esperada-activacion)*g_der(excitacion)*fila_entrada

    #Actualiza pesos
    pesos = dif_pesos+pesos
    error_actual = calcular_error(entradas_train,salidas_train,pesos)
    if error_actual <= error_min:
        error_min = error_actual
        pesos_min = pesos.copy()
    i = i+1
    #print(f"Error Actual:{error_actual} // Error Minimo: {error_min}")

if error_min==0:
    print (f"Solucion encontrada,{pesos}")
else:
    print("Cota alcanzada")

print("Vector de pesos (w): "+str(pesos))
print("Error minimo: "+str(error_min))


###################################################################################################################

###################################################################################################################
######## FASE DE GENERALIZACIÓN (PRUEBA) #############

print("\n--- FASE DE GENERALIZACIÓN (PRUEBA) ---")
print("Cantidad de ejemplos usados: " + str(len(entradas_test)))

# Variables para acumular los errores
error_cuadratico_total = 0
error_absoluto_total = 0
porcentaje_errores = np.array([])

for i in range(len(entradas_test)):
    # 1. Calculamos la predicción normalizada (entre 0 y 1)
    excitacion = calcularExcitacion(entradas_test[i], pesos_min)
    prediccion_norm = calcularActivacion(excitacion)
    
    # 2. DESESCALAMOS la predicción al rango de valores reales
    prediccion_real = prediccion_norm * (s_max - s_min) + s_min
    
    # 3. Tomamos la salida esperada SIN ESCALAR (original)
    esperado_real = salidas_test_originales[i]
    
    # 4. Calculamos el Error Absoluto (la distancia real en unidades)
    distancia_absoluta = abs(esperado_real - prediccion_real)
    error_absoluto_total += distancia_absoluta
    error_cuadratico_total += (esperado_real - prediccion_real) ** 2
    
    # 5. Calculamos el porcentaje de error (solo para tener la comparación)
    if esperado_real != 0:
        porcentaje_error = (distancia_absoluta / abs(esperado_real)) * 100
    else:
        porcentaje_error = abs(prediccion_real) * 100
        
    porcentaje_errores = np.append(porcentaje_errores, porcentaje_error)
    
    # Opcional: Descomenta la siguiente línea para ver el detalle de cada predicción
    print(f"Ejemplo {i} | Esperado: {esperado_real:.4f} | Predicción: {prediccion_real:.4f} | Error: {distancia_absoluta:.4f} unidades | Error: {porcentaje_error:.2f}%")

# Calculamos las métricas finales
mse = error_cuadratico_total / len(entradas_test)
mae = error_absoluto_total / len(entradas_test)
error_porcentual_promedio = np.mean(porcentaje_errores)

print(f"Error Cuadrático Medio (MSE) en prueba: {mse:.4f}")
print(f"Error Absoluto Medio (MAE) en prueba: {mae:.4f} unidades")
print(f"Porcentaje de error promedio (MAPE): {error_porcentual_promedio:.2f}%")