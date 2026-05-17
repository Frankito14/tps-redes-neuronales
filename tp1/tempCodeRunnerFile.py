import numpy as np
import math
import random

#PERCEPTRON LINEAL

PORCENTAJE_ENTRENAMIENTO = 70
PORCENTAJE_PRUEBA = 100 - PORCENTAJE_ENTRENAMIENTO

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

TOTAL_EJEMPLOS = entradas


# Leemos el archivo de salidas esperadas y convertimos cada línea en un flotante
with open(path_salidas, 'r') as f:
    # Usamos una lista de comprensión para extraer los datos
    salidas = [float(linea.strip()) for linea in f if linea.strip()]

# Convertimos la lista completa en un solo np.array
TOTAL_SALIDAS = (np.array(salidas)).copy()


LIMITE =  math.floor(len(TOTAL_SALIDAS) * (PORCENTAJE_ENTRENAMIENTO/100))
ENTRADAS = TOTAL_EJEMPLOS[:LIMITE]
ESPERADO = TOTAL_SALIDAS[:LIMITE]

PRUEBA_ENTRADAS = TOTAL_EJEMPLOS[-(len(TOTAL_EJEMPLOS) -LIMITE):]
PRUEBA_ESPERADO = TOTAL_SALIDAS[-(len(TOTAL_EJEMPLOS) -LIMITE):]

#Cantidad de entradas.
P = len(ESPERADO)
#Tope de iteraciones.
COTA = 300
#Tasa de aprendizaje
N = 0.03#
#Numero de iteración
i = 0
#Pesos iniciales (al azar)
pesos = np.array([1, 1, 1, 1])
#Vector de pesos con error minimo registrado
w_min = pesos
#Inicializar error (se sobreescribe en la primera iteracion)
error = 1
#Numero de error (?)

def aproximacionDeError(entradas, esperado, pesos, p):#
    error = 0
    for i in range(p):
        exitacion = entradas[i] @ pesos
        obtenido = exitacion
        resultado = (esperado[i] - obtenido) ** 2
        error = error + resultado
    return error/2

error_min = aproximacionDeError(ENTRADAS, ESPERADO, pesos, P)

print("ENTRENAMIENTO")
print("Cantidad de ejemplos usados: "+str(P)+" ("+str(PORCENTAJE_ENTRENAMIENTO)+"%)")

while(error > 0 and i < COTA):

    #Selecciona un indice al azar.
    index_random = random.randint(0, P-1)

    #Se toma un ejemplo al azar.
    ejemplo = ENTRADAS[index_random]

    #Se calcula la exitación del ejemplo tomado
    exitacion = ejemplo@pesos
    #Se calcula la salida con la exitacion obtenida
    obtenido = exitacion

    #Tomamos la salida esperada del ejemplo tomado
    esperado = ESPERADO[index_random]

    #Calculamos el delta w para saber que tanto "mover" w. 
    dw = N * (esperado - obtenido) * ejemplo
    
    #Calculamos el nuevo vector de pesos.
    pesos = pesos + dw

    #Calculamos el error para saber que tan buena es nuestra entrada.
    error = aproximacionDeError(ENTRADAS, ESPERADO, pesos, P)

    #Debug
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
print("Cantidad de ejemplos usados: "+str(len(PRUEBA_ENTRADAS))+" ("+str(PORCENTAJE_PRUEBA)+"%)")