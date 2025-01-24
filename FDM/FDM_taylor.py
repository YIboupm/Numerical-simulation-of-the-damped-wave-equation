import numpy as np
import matplotlib.pyplot as plt

# Parámetros
L = 5             # Longitud de la cuerda
c = 2             # Velocidad de la onda, derivada de la ecuación de onda u_tt = c^2 * u_xx
T = 2.5           # Tiempo total de simulación
dx = 0.1          # Incremento espacial (delta x)
dt = 0.005        # Incremento temporal (delta t)

x = np.arange(0, L + dx, dx)   # Discretización del dominio espacial
t = np.arange(0, T + dt, dt)   # Discretización del dominio temporal
nx = len(x)                    # Número de puntos espaciales
nt = len(t)                    # Número de puntos temporales

# Cálculo del número de Courant para la estabilidad, debe ser <= 1
r = (c * dt / dx) ** 2

# Condición inicial: función de onda
def new_initial_wave_function(x, L):
    return 2 * np.sin(3 * np.pi * x / L)  # Función seno inicial de la onda

# Configuración inicial de la matriz u para almacenar los valores de la onda
u = np.zeros((nx, nt)) 

# Inicialización de la función de onda u en t = 0 utilizando la nueva condición inicial
u[:, 0] = new_initial_wave_function(x, L)

# Calculamos el primer paso de tiempo utilizando el desarrollo de Taylor para obtener u[:, 1]
# Suponemos que la velocidad inicial es cero (condición de reposo inicial)
for i in range(1, nx - 1):
    u[i, 1] = u[i, 0] + 0 + 0.5 * r * (u[i+1, 0] - 2 * u[i, 0] + u[i-1, 0])

# Método de Diferencias Finitas en el Tiempo para calcular la solución en los pasos de tiempo restantes
# Inicialización de la función de onda en el tiempo n-1 (paso anterior)
u_prev = np.copy(u[:, 0])  # u en el tiempo n-1

# Método de diferencias finitas en el tiempo para calcular la solución en los pasos de tiempo requeridos
for n in range(1, nt - 1):  # Notar que comenzamos en el segundo paso (n=1 ya está calculado)
    for i in range(1, nx - 1):
        u[i, n+1] = 2 * (1 - r) * u[i, n] + r * (u[i+1, n] + u[i-1, n]) - u_prev[i]
    u_prev = np.copy(u[:, n])  # Actualización del valor anterior

# Visualización de la evolución de la onda en diferentes pasos de tiempo
time_steps_to_display = [0, 0.5, 1, 1.5, 2, 2.5]
plt.figure(figsize=(12, 8))

# Gráfico de la forma de onda en cada uno de los pasos de tiempo especificados
for time_step in time_steps_to_display:
    n = int(time_step / dt)  # Conversión del tiempo a pasos de tiempo
    plt.plot(x, u[:, n], label=f't = {time_step:.1f}s')
    
plt.title('Evolución de la Onda en Diferentes Pasos de Tiempo')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.grid(True)
plt.show()
