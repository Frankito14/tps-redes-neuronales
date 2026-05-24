import numpy as np
import random

# --- CONSTANTES ---
ENTRADAS = np.array([[-1, 1, 1], [1, -1, 1], [-1, -1, 1], [1, 1, 1]])
ESPERADO = np.array([1, 1, -1, -1])
N = 0.1  # Tasa de aprendizaje
BETA = 0.5 # Parámetro de sigmoidea
COTA = 10000 # El PMC requiere más iteraciones PSE
N_OCULTAS = 2 # Cantidad de neuronas en capa oculta

# --- FUNCIONES ---
# Tanh porque necesitamos un valor ente 1 y -1 en las salidas.
def g(h): return np.tanh(BETA * h)
def g_derivada(h): return BETA * (1 - np.tanh(BETA * h)**2)
def error_cuadratico_medio(obtenido, esperado): return 0.5 * (esperado - obtenido)**2

# --- INICIALIZACIÓN ---
w_oculta = np.random.uniform(-0.5, 0.5, (N_OCULTAS, 3)) # [2][3] -> 2 = N_OCULTAS, 3 = Dimension de Ejemplos 
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
print("\nResultados finales:")
for ejemplo in ENTRADAS:
    exitacion_oculta = np.dot(w_oculta, ejemplo)
    activacion_oculta = np.append(g(exitacion_oculta), 1)
    entradas_salida = activacion_oculta
    exitacion_salida = np.dot(w_salida, entradas_salida)
    activacion_salida = g(exitacion_salida)
    obtenido = activacion_salida
    print(f"Entrada: {ejemplo[:2]} -> Salida: {obtenido:.4f}")


# --- FORMULAS ECUACIONES ---
print("\n--- Ecuaciones de los Hiperplanos (Capa Oculta) ---")
for j in range(len(w_oculta)):
    w1 = w_oculta[j][0]
    w2 = w_oculta[j][1]
    bias = w_oculta[j][2]
    print(f"Hiperplano Neurona Oculta {j+1}: {w1:.4f}*x + {w2:.4f}*y + {bias:.4f} = 0")
  