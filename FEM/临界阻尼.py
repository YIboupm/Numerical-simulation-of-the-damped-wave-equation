import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, inv

# Parameters
L = 5       # Domain length
N = 500     # Number of elements (nodes - 1)
c = 2       # Wave speed
h = L / N   # Element size

# Time parameters
dt = 0.0001  # Time step
T = 2        # Total simulation time
nt = int(T / dt)  # Number of time steps

# Initialize solution vectors
U = np.zeros(N+1)  # Displacement
V = np.zeros(N+1)  # Velocity

# Initial condition function
def initial_condition(x):
    return 2 * np.sin(3 * np.pi * x / L)

# Set initial condition
x_nodes = np.linspace(0, L, N+1)
U[:] = initial_condition(x_nodes)

# Apply boundary conditions
U[0] = U[-1] = 0

# Assemble mass (M) and stiffness (K) matrices
M = np.zeros((N+1, N+1))
K = np.zeros((N+1, N+1))
for i in range(1, N):
    # Mass matrix
    M[i, i] += 2 * h / 6
    M[i, i-1] += h / 6
    M[i-1, i] += h / 6

    # Stiffness matrix
    K[i, i] += 2 * (c**2) / h
    K[i, i-1] += -1 * (c**2) / h
    K[i, i+1] += -1 * (c**2) / h

# Apply boundary conditions to M and K
M[0, 0] = M[N, N] = h / 2
K[0, 0] = K[N, N] = 1

# Step 1: Compute eigenvalues and eigenvectors
K_internal = K[1:-1, 1:-1]
M_internal = M[1:-1, 1:-1]
eigvals, eigvecs = eigh(K_internal, M_internal)
eigvals_positive = eigvals[eigvals > 0]
if len(eigvals_positive) < 3:
    raise ValueError("Not enough positive eigenvalues to compute Rayleigh damping!")
omega = np.sqrt(eigvals_positive)

# Step 2: Select target frequencies (include the third mode)
omega1 = omega[0]  # 第一模态频率
omega2 = omega[2]  # 第三模态频率

# Step 3: Compute Rayleigh damping coefficients
xi_target = 1  # 目标阻尼比（临界阻尼）
alpha = 2 * xi_target * omega1 * omega2 / (omega1 + omega2)
beta = 2 * xi_target / (omega1 + omega2)

# Step 4: Construct Rayleigh damping matrix
C_rayleigh = alpha * M + beta * K

# Newmark method parameters
gamma = 0.5
beta_newmark = 0.25

# Modify system matrix for Newmark method
A = M + beta_newmark * dt**2 * K + gamma * dt * C_rayleigh
invA = inv(A)

# Time-stepping loop
displacement_over_time = [U.copy()]

for t in range(1, nt + 1):
    # Compute effective load vector
    B = M @ (U + dt * V + beta_newmark * dt**2 * V) + gamma * dt * C_rayleigh @ U

    # Solve for next displacement
    U_new = invA @ B

    # Compute next velocity
    V_new = gamma / (beta_newmark * dt) * (U_new - U) - (gamma / beta_newmark - 1) * V

    # Update solution
    U, V = U_new, V_new

    # Apply boundary conditions
    U[0] = U[-1] = 0

    # Store displacement
    displacement_over_time.append(U.copy())

# Plot the solution over time
x = x_nodes
plt.figure(figsize=(12, 6))

# Plot initial condition
plt.plot(x, initial_condition(x), label='Initial condition (t=0)', linestyle='--')

# Plot displacement over time
for i, U in enumerate(displacement_over_time):
    if i % (nt // 4) == 0:  # Plot 4 snapshots
        plt.plot(x, U, label=f'Displacement (t={i*dt:.2f}s)')

plt.xlabel('x')
plt.ylabel('Displacement U(x, t)')
plt.title('Finite Element Method Solution Over Time (Rayleigh Damping)')
plt.legend()
plt.grid(True)
plt.show()
