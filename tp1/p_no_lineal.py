import numpy as np
import random
import math

#PERCEPTRON NO LINEAL

#Configuracion de Entrenamiento / Testing

PORCENTAJE_ENTRENAMIENTO = 70
PORCENTAJE_PRUEBA = 100 - PORCENTAJE_ENTRENAMIENTO

# Extraccion de datos:

# Leemos el archivo de entradas y convertimos cada línea en un np.array de flotantes
def entradas_del_archivo():
    path_entradas = 'entradas.txt'
    entradas = []
    with open(path_entradas, 'r') as f:
        for linea in f:
            # Limpiamos espacios y dividimos la línea por columnas
            valores = linea.strip().split()
            if valores:
                # Convertimos cada valor a float y creamos el np.array
                array_data = np.array([float(v) for v in valores])
                array_data = np.append(array_data, 1.0)
                entradas.append(array_data)
    return entradas

# Leemos el archivo de salidas esperadas y convertimos cada línea en un flotante
def salidas_del_archivo():
    path_salidas = 'salidas.txt'
    with open(path_salidas, 'r') as f:
        # Usamos una lista de comprensión para extraer los datos
        salidas = [float(linea.strip()) for linea in f if linea.strip()]
    return salidas

#Funcion para escalar las salidas entre 1 y 0 (Logisitica)
def escalar_lista(lista, min_val, max_val):
    # Usamos una comprensión de lista para aplicar la fórmula a cada elemento
    # Esto funciona tanto para listas de Python como para np.arrays
    return [(x - min_val) / (max_val - min_val) for x in lista]

#Funcion para desescalar las salidas entre 1 y 0 (Logisitica)
def desescalar(valor_norm, min_val, max_val):
    # Revierte la fórmula anterior
    return valor_norm * (max_val - min_val) + min_val

#Mezclar ejemplos
pares =  list(zip(entradas_del_archivo(), salidas_del_archivo()))
random.shuffle(pares)
LE_mezclado, LS_mezclado = zip(*pares)
TOTAL_ENTRADAS = list(LE_mezclado)
TOTAL_SALIDAS = list(LS_mezclado)

#Separar en conjuntos de Entrenamiento / Testing:

LIMITE =  math.floor(len(TOTAL_SALIDAS) * (PORCENTAJE_ENTRENAMIENTO/100))
ENTRADAS_ENTRENAMIENTO = TOTAL_ENTRADAS[:LIMITE]
SALIDAS_ENTRENAMIENTO_SIN_ESCALAR = TOTAL_SALIDAS[:LIMITE]

ENTRADAS_PRUEBA = TOTAL_ENTRADAS[-(len(TOTAL_ENTRADAS) - LIMITE):]
SALIDAS_PRUEBA_SIN_ESCALAR= TOTAL_SALIDAS[-(len(TOTAL_ENTRADAS) - LIMITE):]

#Escalar salidas del conjunto de entrenamiento
minimo = min(SALIDAS_ENTRENAMIENTO_SIN_ESCALAR)   
maximo = max(SALIDAS_ENTRENAMIENTO_SIN_ESCALAR)


SALIDAS_ENTRENAMIENTO = escalar_lista(SALIDAS_ENTRENAMIENTO_SIN_ESCALAR, minimo, maximo)
SALIDAS_PRUEBA = escalar_lista(SALIDAS_PRUEBA_SIN_ESCALAR, minimo, maximo)

#Cantidad de entradas.
P = len(SALIDAS_ENTRENAMIENTO)
#Tope de iteraciones.
COTA = 1000
#Tasa de aprendizaje
N = 0.1
#Beta para definir la "potencia" de la función g (Sigmoide).
B = 1
#Numero de iteración
i = 0
#Pesos iniciales (al azar)
pesos = np.array([1, 1, 1, 1])
#Vector de pesos con error minimo registrado
w_min = pesos
#Inicializar error (se sobreescribe en la primera iteracion)
error = 1
#Numero de error (?)
error_min = 2*P

def g(exitacion, b):
    #Funcion logistica.
    #np.exp = Exponencial natural.
    return 1/(1 + np.exp(-2*b*exitacion))

def g_der(exitacion, b):
    # Función logística derivada
    ex = g(exitacion, b)
    return 2*b*ex*(1-ex)

#Funcion de activacion(Usa)
def activacion(exitacion):
    return g(exitacion, B)

def aproximacionDeError(entradas, salidas, pesos, p):
    error = 0
    for i in range(p):
        exitacion = np.inner(entradas[i], pesos)
        obtenido = activacion(exitacion)
        resultado = (salidas[i] - obtenido) ** 2
        error = error + resultado
    return error/2

print("ENTRENAMIENTO")
print("Cantidad de ejemplos usados: "+str(P)+" ("+str(PORCENTAJE_ENTRENAMIENTO)+"%)")

while(error > 0 and i < COTA):

    #Selecciona un indice al azar.
    index_random = random.randint(0, P-1)

    #Se toma un ejemplo al azar.
    ejemplo = ENTRADAS_ENTRENAMIENTO[index_random]

    #Se calcula la exitación del ejemplo tomado
    exitacion = np.inner(ejemplo, pesos)

    #Se calcula la activacion de la exitacion calculada
    obtenido = activacion(exitacion)

    #Tomamos la salida esperada del ejemplo tomado
    esperado = SALIDAS_ENTRENAMIENTO[index_random]

    #Calculamos el delta w para saber que tanto "mover" w.
    dw = N * (esperado - obtenido) * g_der(exitacion, B) * ejemplo

    #Calculamos el nuevo vector de pesos.
    pesos = pesos + dw

    #Calculamos el error para saber que tan buena es nuestra entrada.
    error = aproximacionDeError(ENTRADAS_ENTRENAMIENTO, SALIDAS_ENTRENAMIENTO, pesos, P)

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
  print(pesos)
else:
  print("Limite de iteraciones alcanzado")
  print("Mejor solucion encontrada: W = "+ str(w_min))
  print("Error minimo encontrado: "+str(error_min))

print("GENERALIZACION")
print("Cantidad de ejemplos usados: "+str(len(ENTRADAS_PRUEBA))+" ("+str(PORCENTAJE_PRUEBA)+"%)")

#Iterar sobre el conjunto de prueba comparando la prediccion con el valor real.
porcentaje_errores = np.array([])
for i in range(len(ENTRADAS_PRUEBA)):
    exitacion = np.inner(ENTRADAS_PRUEBA[i], w_min)
    prediccion = activacion(exitacion)
    esperado = SALIDAS_PRUEBA[i]
    porcentaje_error = abs(prediccion - esperado) / abs(esperado) * 100
    porcentaje_errores = np.append(porcentaje_errores, porcentaje_error)
    #print("Ejemplo: "+str(i))
    #print("Salida obtenida: "+str(prediccion)+" - Salida esperada: "+str(esperado)+" - Porcentaje de error: "+str(porcentaje_error)+"%")
print("Porcentaje de error promedio: "+str(np.mean(porcentaje_errores))+"%")