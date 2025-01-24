import numpy as np
import matplotlib.pyplot as plt

# 时间和空间的离散化参数
c = 2                  # 波速
L = 5                   # 域的长度
beta = 1                 # 阻尼系数
T = 2.5                  # 总时间
dx = 0.1                 # 空间步长
dt = (dx / c) / np.sqrt(1 + (beta * dx / (2 * c)))  # 时间步长，为了稳定性进行调整
x = np.arange(0, L + dx, dx)  # 空间网格
t = np.arange(0, T + dt, dt)  # 时间网格
r = (c * dt / dx)**2        # CFL数
s = beta * dt

def f(x):
    
    return 2 * np.sin(3 * np.pi * x / L)
    center = L / 2  # Center of the domain
    width = 2      # Width of the rectangular pulse
    amplitude = 2   # Amplitude of the rectangular pulse
    return np.where((x >= center - width / 2) & (x <= center + width / 2), amplitude, 0)

# 初始化解矩阵
U = np.zeros((len(t), len(x)))    

# 设置初始条件
U[0, :] = f(x)

# 使用g(x) = 0应用初始条件
U[1, 1:-1] = U[0, 1:-1]

# 应用边界条件，两端固定为0
U[:, 0] = 0   
U[:, -1] = 0

# 使用更新的参数解决波动方程
for n in range(1, len(t) - 1):
    U[n + 1, 1:-1] = (2 - 2*r - s) * U[n, 1:-1] + r * (U[n, :-2] + U[n, 2:]) - (1 - s) * U[n - 1, 1:-1]
    U[n + 1, 0] = 0
    U[n + 1, -1] = 0

# 创建一个图形并绘制指定时间点的波形
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
times = [0, 0.5, 1, 1.5, 2, 2.5]
for ax, time in zip(axs.flat, times):
    ax.plot(x, U[int(time/dt), :], 'b', linewidth=2)
    ax.set_xlim(0, L)
    ax.set_ylim(-3, 3)
    ax.set_title(f"Time = {time:.1f} s")
    ax.grid(True)

plt.tight_layout()
plt.show()
