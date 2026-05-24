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

V_E = {
    "0": [1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    "1": [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
    "2": [-1, -1, 1, -1, -1, -1, -1, -1, -1, -1],
    "3": [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1],
    "4": [-1, -1, -1, -1, 1, -1, -1, -1, -1, -1],
    "5": [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1],
    "6": [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1],
    "7": [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1],
    "8": [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1],
    "9": [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1]
}

ENTRADAS = np.array([NUM["0"],NUM["1"],NUM["2"],NUM["3"],NUM["4"],NUM["5"],NUM["6"],NUM["7"],NUM["8"],NUM["9"]])
ESPERADO = np.array([V_E["0"],V_E["1"],V_E["2"],V_E["3"],V_E["4"],V_E["5"],V_E["6"],V_E["7"],V_E["8"],V_E["9"],])
N = 0.1  # Tasa de aprendizaje
BETA = 0.5 # Parámetro de sigmoidea
COTA = 50000 # El PMC requiere más iteraciones PSE

N_OCULTAS = 10 # Cantidad de neuronas en capa oculta
N_COMPONENTES = 36 # Componentes de los ejemplos (5*7) + 1
N_SALIDA = 10 # Cantidad de neuronas en capa de salida

# --- FUNCIONES ---
# Tanh porque necesitamos un valor ente 1 y -1 en las salidas.
def g(h): return np.tanh(BETA * h)
def g_derivada(h): return BETA * (1 - np.tanh(BETA * h)**2)
def error_cuadratico_medio(obtenido, esperado): return 0.5 * (esperado - obtenido)**2

def aplicar_ruido(arregloNumeros, probabilidad=0.02):
    arregloConRuido = np.copy(arregloNumeros)
    for i in range(35): #El sesgo no
        if np.random.rand() < probabilidad:
            # Invertimos el bit: 1 -> -1; -1 -> 1
            arregloConRuido[i] *= -1 
    return arregloConRuido

# --- ENTRADAS CON RUIDO ---

PROBABILIDAD = 0.02 #2%

NUM_RUIDO = {
    "0": aplicar_ruido(NUM["0"], PROBABILIDAD),
    "1": aplicar_ruido(NUM["1"], PROBABILIDAD),
    "2": aplicar_ruido(NUM["2"], PROBABILIDAD),
    "3": aplicar_ruido(NUM["3"], PROBABILIDAD),
    "4": aplicar_ruido(NUM["4"], PROBABILIDAD),
    "5": aplicar_ruido(NUM["5"], PROBABILIDAD),
    "6": aplicar_ruido(NUM["6"], PROBABILIDAD),
    "7": aplicar_ruido(NUM["7"], PROBABILIDAD),
    "8": aplicar_ruido(NUM["8"], PROBABILIDAD),
    "9": aplicar_ruido(NUM["9"], PROBABILIDAD),
}

ENTRADAS_RUIDOSAS = np.array([NUM_RUIDO["0"],NUM_RUIDO["1"],NUM_RUIDO["2"],NUM_RUIDO["3"],NUM_RUIDO["4"],NUM_RUIDO["5"],NUM_RUIDO["6"],NUM_RUIDO["7"],NUM_RUIDO["8"],NUM_RUIDO["9"]])

# --- INICIALIZACIÓN ---
w_oculta = np.random.uniform(-0.5, 0.5, (N_OCULTAS, N_COMPONENTES)) # Entrada -> capa oculta | [10][35]
w_salida = np.random.uniform(-0.5, 0.5, (N_SALIDA, N_OCULTAS + 1)) # Pesos oculta -> salida | [10][11]

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

    # w_salida[:, :-1] -> Quitamos el sesgo (ultima columna) de TODAS las filas (neuronas)
    # ya no podemos hacer (w_salida[:-1] * delta_salida) porque delta_salida es un array, no un escalar, toca hacer la sumatoria.
    suma_errores_retropropagados = np.dot(delta_salida, w_salida[:, :-1]) # E wi * delta_i
    delta_oculto = g_derivada(exitacion_oculta) * suma_errores_retropropagados

    # delta_oculta = Error local [10] (Uno por cada neurona)

    # Actualizar pesos 
    # Actualizar pesos de capa de salida
    # No se usa el ejemplo, sino sus propias entradas (activacion/salida de capa oculta)
    for i in range(N_SALIDA):
        w_salida[i] += N * delta_salida[i] * entradas_salida

    # Actualizar pesos de capa oculta (por cada neurona)
    for j in range(N_OCULTAS):
        w_oculta[j] += N * delta_oculto[j] * ejemplo

    #Muestra actualizacion del error cada 1000 iteraciones
    if iteracion % 1000 == 0:
        print(f"Iteracion #{iteracion} - Error: {error}")

# --- RESULTADOS ---
print("\nResultados finales (Numeros limpios):")
for numero, ejemplo in enumerate(ENTRADAS):
    exitacion_oculta = np.dot(w_oculta, ejemplo)
    activacion_oculta = np.append(g(exitacion_oculta), 1)
    entradas_salida = activacion_oculta
    exitacion_salida = np.dot(w_salida, entradas_salida)
    activacion_salida = g(exitacion_salida)
    obtenido = activacion_salida
    prediccion = np.argmax(obtenido) # Indice del valor maximo
    print(f"Entrada: {numero} - Predicción: {prediccion} - Salida: {obtenido}")

# --- TESTEO DE RUIDO ---
print("\nResultados finales (Numeros con ruido):")
for numero, ejemplo in enumerate(ENTRADAS_RUIDOSAS):
    exitacion_oculta = np.dot(w_oculta, ejemplo)
    activacion_oculta = np.append(g(exitacion_oculta), 1)
    entradas_salida = activacion_oculta
    exitacion_salida = np.dot(w_salida, entradas_salida)
    activacion_salida = g(exitacion_salida)
    obtenido = activacion_salida
    prediccion = np.argmax(obtenido) # Indice del valor maximo
    print(f"Entrada (Ruido): {numero} - Predicción: {prediccion} - Salida: {obtenido}")


