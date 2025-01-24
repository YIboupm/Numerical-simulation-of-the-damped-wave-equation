import numpy as np
import matplotlib.pyplot as plt

# Parámetros
L = 5             # Longitud de la cuerda
c = 2             # Velocidad de la onda
beta = beta = (6 * c * np.pi) / L          # Coeficiente de amortiguamiento
T = 2.5           # Tiempo total de simulación
dx = 0.04          # Incremento espacial (delta x)
dt = 0.0005        # Incremento temporal (delta t)

x = np.arange(0, L + dx, dx)   # Discretización del dominio espacial
t = np.arange(0, T + dt, dt)   # Discretización del dominio temporal
nx = len(x)                    # Número de puntos espaciales
nt = len(t)                    # Número de puntos temporales

# Cálculo del número de Courant para la estabilidad
r = (c * dt / dx)**2

# Condición inicial: función de onda
def initial_wave_function(x, L):
    return 2 * np.sin(3 * np.pi * x / L)
    center = L / 2  # Center of the domain
    width = 2      # Width of the rectangular pulse
    amplitude = 2   # Amplitude of the rectangular pulse
    return np.where((x >= center - width / 2) & (x <= center + width / 2), amplitude, 0)
    
    
      # Función seno inicial de la onda

# Configuración inicial de la matriz u para almacenar los valores de la onda
u = np.zeros((nx, nt))

# Inicialización de la función de onda u en t = 0 utilizando la nueva condición inicial
u[:, 0] = initial_wave_function(x, L)

# Calculamos el primer paso de tiempo utilizando el desarrollo de Taylor
# Suponemos que la velocidad inicial es cero
for i in range(1, nx - 1):
    u[i, 1] = u[i, 0] + 0 + 0.5 * r * (u[i+1, 0] - 2 * u[i, 0] + u[i-1, 0])

# Método de Diferencias Finitas para resolver la ecuación de onda con amortiguamiento
for n in range(1, nt - 1):
    for i in range(1, nx - 1):
        u[i, n+1] = (1 / (1 + beta * dt / 2)) * (
            r * (u[i+1, n] + u[i-1, n])
            + 2 * (1 - r) * u[i, n]
            - (1 - beta * dt / 2) * u[i, n-1]
        )

# Visualización
time_steps_to_display = [0, 0.5, 1, 1.5, 2, 2.5]
fig, axes = plt.subplots(len(time_steps_to_display), 1, figsize=(10, 15))

# 计算 u(x, t) 的全局最小值和最大值
u_min = np.min(u)
u_max = np.max(u)

for ax, time_step in zip(axes, time_steps_to_display):
    n = int(round(time_step / dt))
    ax.plot(x, u[:, n], label=f't = {time_step:.1f}s')
    ax.set_title(f'Evolución de la Onda en t = {time_step:.1f}s')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.grid(True)
    ax.legend()
    ax.set_ylim([u_min, u_max])  # 固定纵轴范围

plt.tight_layout()
plt.show()

