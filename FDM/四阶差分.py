import numpy as np
import matplotlib.pyplot as plt

# Parámetros
L = 5             
c = 2             
beta = 10         
T = 2.5           
dx = 0.01          
dt = 0.0001        

x = np.arange(0, L + dx, dx)  
t = np.arange(0, T + dt, dt)  
nx, nt = len(x), len(t)

# Verificación del número de Courant
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
# Opcional: velocidad inicial g(x). Aquí g(x)=0
def initial_velocity(x):
    return np.zeros_like(x)

f = initial_wave_function(x, L)
g = initial_velocity(x)

# Arreglo de soluciones
u = np.zeros((nx, nt))
u[:, 0] = f

# Condiciones de frontera
u[0, :] = 0
u[-1, :] = 0

# Función auxiliar para calcular la segunda derivada espacial con precisión de cuarto orden
def four_point_derivative_2nd_order(u_n, dx):
    # u_n 是当前时间层 n 时刻的 u(x), 大小为 nx
    # 返回四阶精度的空间二阶导数组 u_xx
    u_xx = np.zeros_like(u_n)
    # Cuarta orden: U_xx_j ≈ (-u[j-2] +16u[j-1] -30u[j] +16u[j+1] - u[j+2])/(12 dx^2)
    u_xx[2:-2] = (-u_n[0:-4] + 16*u_n[1:-3] -30*u_n[2:-2] +16*u_n[3:-1] - u_n[4:])/(12*dx**2)
    return u_xx

# Paso inicial: usando la expansión de Taylor para obtener u[:,1]
# U_j^1 = U_j^0 + dt*g(x_j) + (dt^2/2)*(C^2 U_xx_j^0 - beta*g(x_j))
u_xx_0 = four_point_derivative_2nd_order(u[:,0], dx)
u[2:-2, 1] = u[2:-2, 0] + dt*g[2:-2] + 0.5*dt**2*(c**2*u_xx_0[2:-2] - beta*g[2:-2])
# Mantener frontera en 0
u[0,1] = 0
u[-1,1] = 0

# A partir de n=1 hacia adelante
# Fórmula:
# U_j^{n+1} = [2U_j^n - U_j^{n-1} + dt^2*C^2*U_xx_j^n + (beta*dt/2)*U_j^{n-1}] / (1 + beta*dt/2)
for n in range(1, nt-1):
    u_xx = four_point_derivative_2nd_order(u[:,n], dx)
    u[2:-2, n+1] = (2*u[2:-2, n] 
                    - u[2:-2, n-1] 
                    + dt**2*c**2*u_xx[2:-2] 
                    + (beta*dt/2)*u[2:-2, n-1])/(1 + beta*dt/2)
    
    # Condiciones de frontera
    u[0, n+1] = 0
    u[-1, n+1] = 0

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