import numpy as np
import matplotlib.pyplot as plt

# Parámetros
L = 5             
c = 2             
beta = 2          
T = 2.5           
dx = 0.04          
dt = 0.0005        

x = np.arange(0, L + dx, dx)  
t = np.arange(0, T + dt, dt)  
nx, nt = len(x), len(t)

# Cálculo del número de Courant
r = (c * dt / dx)**2
if r > 1:
    raise ValueError(f"El número de Courant r = {r} no cumple la condición de estabilidad (r <= 1).")

# Función inicial
def initial_wave_function(x, L):
    center = L / 2
    width = 2
    amplitude = 2
    return np.where((x >= center - width / 2) & (x <= center + width / 2), amplitude, 0)
    return 2*np.sinc(3 * np.pi * x / L)
    

# Configuración inicial
u = np.zeros((nx, nt))
u[:, 0] = initial_wave_function(x, L)

# Primer paso de tiempo
u[1:-1, 1] = u[1:-1, 0] + 0 + 0.5 * r * (u[2:, 0] - 2 * u[1:-1, 0] + u[:-2, 0])

# Método de Diferencias Finitas (vectorizado)
for n in range(1, nt - 1):
    u[1:-1, n+1] = (1 / (1 + beta * dt / 2)) * (
        r * (u[2:, n] + u[:-2, n])
        + 2 * (1 - r) * u[1:-1, n]
        - (1 - beta * dt / 2) * u[1:-1, n-1]
    )

# Visualización
time_steps_to_display = [0, 0.5, 1, 1.5, 2, 2.5]
fig, axes = plt.subplots(len(time_steps_to_display), 1, figsize=(10, 15))

for ax, time_step in zip(axes, time_steps_to_display):
    n = int(round(time_step / dt))
    ax.plot(x, u[:, n], label=f't = {time_step:.1f}s')
    ax.set_title(f'Evolución de la Onda en t = {time_step:.1f}s')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
