import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import eigh

# Parameters
L = 5       # Domain length
N = 200     # Number of elements (nodes - 1)
c = 5       # Wave speed
h = L / N   # Element size

# Time parameters
dt = 0.001
T = 2
nt = int(T / dt)

# Initial conditions
U = np.zeros(N+1)
V = np.zeros(N+1)

def initial_condition(x):
    return 2 * np.sin(3 * np.pi * x / L)

x_nodes = np.linspace(0, L, N+1)
U[:] = initial_condition(x_nodes)
U[0] = U[-1] = 0

# Assemble M, K (using a simple scheme)
M = np.zeros((N+1, N+1))
K = np.zeros((N+1, N+1))
for i in range(1, N):
    # Lumped mass
    M[i,i] += h
    # Stiffness
    K[i,i] = 2*(c**2)/h
    K[i,i-1] = -(c**2)/h
    K[i,i+1] = -(c**2)/h

M[0,0] = M[N,N] = h/2
K[0,0] = K[N,N] = 1

# ---------- 计算特征值频率，用于选取Rayleigh阻尼 ----------
M_internal = M[1:-1,1:-1]
K_internal = K[1:-1,1:-1]
eigvals, eigvecs = eigh(K_internal, M_internal)
# 提取正的特征值，对应模态频率
eigvals_positive = eigvals[eigvals > 0]
omega = np.sqrt(eigvals_positive)

# 第三模态频率对应omega[2] (因为下标从0开始)
omega_target = omega[2]

# 假设我们只想让第三模态达到近似临界阻尼 zeta=1
# 临界阻尼比zeta = 1 = (alpha/(2*omega_target) + beta*(omega_target)/2)
# 为确定alpha和beta，我们需要一个额外条件。这里为简化，我们只考虑alpha，让beta=0:
# zeta=1时: 1 = alpha/(2*omega_target) => alpha = 2*omega_target
alpha = 2 * omega_target
beta = 0.0  # 简化处理

C = alpha * M + beta * K

# System matrix for implicit Euler
A = M + dt*C + dt**2 * K
A_sparse = csr_matrix(A)

displacement_over_time = [U.copy()]

for n in range(1, nt+1):
    # Right-hand side vector B
    # B = M*U_n + dt*M*V_n + dt*C*U_n
    B = M @ U + dt*M @ V + dt*C @ U

    # Solve for U_{n+1}
    U_new = spsolve(A_sparse, B)

    # Update V_{n+1}
    V_new = (U_new - U)/dt

    # Apply boundary conditions
    U_new[0] = U_new[-1] = 0

    U, V = U_new, V_new
    displacement_over_time.append(U.copy())

# Plot results
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_nodes, initial_condition(x_nodes), '--', label='Initial')
time_steps_to_display = [0, int(nt*0.25), int(nt*0.5), int(nt*0.75), nt]
for step in time_steps_to_display:
    ax.plot(x_nodes, displacement_over_time[step], label=f't={step*dt:.2f}s')

ax.set_xlabel('x')
ax.set_ylabel('Displacement U(x,t)')
ax.set_title('Wave with Rayleigh Damping (Implicit Euler)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
