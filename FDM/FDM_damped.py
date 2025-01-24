import numpy as np
import matplotlib.pyplot as plt

# 参数
L = 5                  # 绳子的长度
c = 2                  # 波速
beta = 2               # 阻尼系数
T = 2.5                # 总时间
dx = 0.1               # 空间步长
dt = (dx / c) / np.sqrt(1 + (beta * dx / (2 * c)))  # 时间步长，为了稳定性进行调整

x = np.arange(0, L + dx, dx)  # 空间网格
t = np.arange(0, T + dt, dt)  # 时间网格
nx = len(x)                    # 空间点数量
nt = len(t)                    # 时间点数量

# CFL 和阻尼数
r = (c * dt / dx) ** 2
s = beta * dt

# 初始条件：波函数
def new_initial_wave_function(x, L):
    return 2 * np.sin(3 * np.pi * x / L)  # 初始波形

# 初始条件：速度
def initial_velocity_function(x, L):
    return np.zeros_like(x)  # 假设初始速度为零

# 初始化波函数矩阵
u = np.zeros((nx, nt))

# 设置初始条件
u[:, 0] = new_initial_wave_function(x, L)  # t=0 的位置初始条件
velocity = initial_velocity_function(x, L)  # 初始速度

# 通过泰勒展开式计算第一个时间步 u[:, 1]
for i in range(1, nx - 1):
    u[i, 1] = (u[i, 0] + dt * velocity[i] 
               + 0.5 * r * (u[i+1, 0] - 2 * u[i, 0] + u[i-1, 0]) 
               - 0.5 * s * u[i, 0])

# 使用差分法计算后续时间步
u_prev = np.copy(u[:, 0])  # 保存前一个时间步的数据

for n in range(1, nt - 1):
    for i in range(1, nx - 1):
        u[i, n+1] = ((2 - 2 * r - s) * u[i, n] 
                     + r * (u[i+1, n] + u[i-1, n]) 
                     - (1 - s) * u_prev[i])
    u_prev = np.copy(u[:, n])  # 更新前一个时间步的数据

# 图像展示，与第一段代码保持一致
time_steps_to_display = [0, 0.5, 1, 1.5, 2, 2.5]
plt.figure(figsize=(12, 8))

# 绘制不同时间点的波形
for time_step in time_steps_to_display:
    n = int(time_step / dt)  # 将时间转换为时间步
    plt.plot(x, u[:, n], label=f't = {time_step:.1f}s')

plt.title('Evolución de la Onda en Diferentes Pasos de Tiempo con Amortiguamiento')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.grid(True)
plt.show()
