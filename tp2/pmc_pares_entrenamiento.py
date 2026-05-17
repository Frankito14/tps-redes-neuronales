import numpy as np
import random

# --- CONSTANTES ---
NUM = {
    "0": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0] + [1], 
    "1": [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0] + [1],
    "2": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1] + [1],
    "3": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0] + [1],
    "4": [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] + [1],
    "5": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0] + [1],
    "6": [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0] + [1],
    "7": [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0] + [1],
    "8": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0] + [1],
    "9": [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0] + [1],
}

# --- TESTEO ---
TESTEO_ENTRADAS = np.array([NUM["7"],NUM["8"],NUM["9"]])
TESTEO_ESPERADO = np.array([ -1, 1, -1])

# --- ENTRENAMIENTO ---
ENTRADAS = np.array([NUM["0"],NUM["1"],NUM["2"],NUM["3"],NUM["4"],NUM["5"],NUM["6"]])
ESPERADO = np.array([1, -1, 1, -1, 1, -1, 1])

# --- PARÁMETROS ---
N = 0.1  # Tasa de aprendizaje
BETA = 0.5 # Parámetro de sigmoidea
COTA = 50000 # El PMC requiere más iteraciones PSE
N_OCULTAS = len(ENTRADAS) # Cantidad de neuronas en capa oculta
N_COMPONENTES = 36 # Componentes de los ejemplos (5*7) + 1


# --- FUNCIONES ---
# Tanh porque necesitamos un valor ente 1 y -1 en las salidas.
def g(h): return np.tanh(BETA * h)
def g_derivada(h): return BETA * (1 - np.tanh(BETA * h)**2)
def error_cuadratico_medio(obtenido, esperado): return 0.5 * (esperado - obtenido)**2
def porcentaje_error_distancia(esp, obt): return (abs(esp - obt) / abs(esp)) * 100 if esp != 0 else (0.0 if obt == 0 else float('inf'))

# --- INICIALIZACIÓN ---
w_oculta = np.random.uniform(-0.5, 0.5, (N_OCULTAS, N_COMPONENTES)) # Entrada -> capa oculta
w_salida = np.random.uniform(-0.5, 0.5, N_OCULTAS + 1) # Pesos oculta -> salida (+1 para sesgo) -> [V1, V2, SESGO]

# --- ENTRENAMIENTO ---
for iteracion in range(COTA):
    # Tomar un ejemplo al azar 
    random_index = random.randint(0, len(ENTRADAS) - 1)
    ejemplo = ENTRADAS[random_index]
    esperado = ESPERADO[random_index]

    # Propagación hacia adelante (Feedforward) 

    # Capa oculta
    exitacion_oculta = np.dot(w_oculta, ejemplo) # Ejemplo [3] x Matriz [3][2] = Entrada [2]
    activacion_oculta = g(exitacion_oculta)

    # Agregar sesgo a la salida de la capa oculta
    entradas_salida = np.append(activacion_oculta, 1) 
    
    # Capa de salida
    exitacion_salida = np.dot(w_salida, entradas_salida)
    activacion_salida = g(exitacion_salida)
    obtenido = activacion_salida

    # Calcular error cuadratico medio
    error = error_cuadratico_medio(obtenido, esperado)

    # Retropropagación (Backpropagation) 
    # Delta de salida
    delta_salida = (esperado - obtenido) * g_derivada(exitacion_salida) 
    
    # Delta de capa oculta (retropropagando delta_salida)

    # w_salida[:-1] -> Quitamos el sesgo para calcular el delta de la capa oculta
    delta_oculto = g_derivada(exitacion_oculta) * (w_salida[:-1] * delta_salida)

    # delta_oculta = Error local [2] (Uno por cada neurona)

    # Actualizar pesos 
    # Actualizar pesos de capa de salida
    w_salida = w_salida + ( N * delta_salida * entradas_salida )

    # Actualizar pesos de capa oculta (por cada neurona)
    for j in range(N_OCULTAS):
        w_oculta[j] = w_oculta[j] + N * delta_oculto[j] * ejemplo

    #Muestra actualizacion del error cada 1000 iteraciones
    if iteracion % 1000 == 0:
        print(f"Iteracion #{iteracion} - Error: {error:.4f}")

# --- RESULTADOS ---
print("\nResultados entrenamiento (0 al 6):")
for numero, ejemplo in enumerate(ENTRADAS):
    exitacion_oculta = np.dot(w_oculta, ejemplo)
    activacion_oculta = np.append(g(exitacion_oculta), 1)
    entradas_salida = activacion_oculta
    exitacion_salida = np.dot(w_salida, entradas_salida)
    activacion_salida = g(exitacion_salida)
    obtenido = activacion_salida
    print(f"Entrada: {numero} -> Salida: {obtenido:.4f} -> Error relativo (Distancia): {porcentaje_error_distancia(ESPERADO[numero], obtenido):.4f}%")



# --- TESTEO ---
print("\nResultados testeo (7 al 9):")
for numero, ejemplo in enumerate(TESTEO_ENTRADAS):
    exitacion_oculta = np.dot(w_oculta, ejemplo)
    activacion_oculta = np.append(g(exitacion_oculta), 1)
    entradas_salida = activacion_oculta
    exitacion_salida = np.dot(w_salida, entradas_salida)
    activacion_salida = g(exitacion_salida)
    obtenido = activacion_salida
    print(f"Entrada: {numero + N_OCULTAS} -> Salida: {obtenido:.4f} -> Error relativo (Distancia): {porcentaje_error_distancia(TESTEO_ESPERADO[numero], obtenido):.4f}%")


