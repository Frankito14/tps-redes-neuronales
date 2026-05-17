import numpy as np
import random

#PERCEPTRÓN SIMPLE ESCALONADO

#Función lógica ’XOR’
#Entradas: x = {{−1, 1}, {1, −1}, {−1, −1}, {1, 1}}
#Salida esperada: y = {1, 1, −1, -1}.

#Conjunto de entrenamiento (4 ejemplos)
ENTRADAS = [np.array([-1, 1, 1]), np.array([1, -1, 1]), np.array([-1, -1, 1]), np.array([1, 1, 1])]
#Salidas esperadas para cada ejemplo.
ESPERADO = np.array([1, 1, -1, -1])
#Dimensiones de las entradas.
P = len(ENTRADAS)
#Tope de iteraciones.
COTA = 60
#Tasa de aprendizaje
N = 0.5
#Numero de iteración
i = 0
#Pesos iniciales (al azar)
pesos = np.array([0, 0, 1])
#Vector de pesos con error minimo registrado
w_min = pesos
#Inicializar error (se sobreescribe en la primera iteracion)
error = 1
#Numero de error (?)
error_min = 2*P

#Funcion escalón de activación
def activacion(exitacion):
    if(exitacion < 0):
        return -1
    else:
        return 1

def aproximacionDeError(entradas, esperado, pesos, p):
    error = 0
    for i in range(P):
        exitacion = np.inner(entradas[i], pesos)
        obtenido = activacion(exitacion)
        resultado = (esperado[i] - obtenido) ** 2
        error = error + resultado
    return error/2

while(error > 0 and i < COTA):

    #Selecciona un indice al azar.
    index_random = random.randint(0, P-1)

    #Se toma un ejemplo al azar.
    ejemplo = ENTRADAS[index_random]

    #Se calcula la exitación del ejemplo tomado
    exitacion = np.inner(ejemplo, pesos)

    #Se calcula la activacion de la exitacion calculada
    obtenido = activacion(exitacion)

    #Tomamos la salida esperada del ejemplo tomado
    esperado = ESPERADO[index_random]

    #Calculamos el delta w para saber que tanto "mover" w.
    dw = N * (esperado - obtenido) * ejemplo

    #Calculamos el nuevo vector de pesos.
    pesos = pesos + dw

    #Calculamos el error para saber que tan buena es nuestra entrada.
    error = aproximacionDeError(ENTRADAS, ESPERADO, pesos, P)

    print("Iteracion: "+ str(i))
    print("W = "+ str(pesos))
    print("Error actual: "+str(error)+" - Error minimo: "+str(error_min))

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

