import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 5               # Length of the domain
N = 50              # Number of elements
c = 2               # Wave speed
gamma = 0.5         # Damping factor for the Newmark-beta method
beta = 0.25         # Parameter for the Newmark-beta method
h = L / N           # Size of each element
nodes = np.linspace(0, L, N+1)  # Node positions

# Initialize global matrices
M_global = np.zeros((N+1, N+1))
K_global = np.zeros((N+1, N+1))
C_global = np.zeros((N+1, N+1))

# Assemble global matrices
for e in range(N):
    indices = [e, e+1]
    # Element mass and stiffness matrices
    M_e = (h / 6) * np.array([[2, 1], [1, 2]])
    K_e = (4 / h) * np.array([[1, -1], [-1, 1]])  # c^2 = 4
    # Assemble into global matrices
    for i in range(2):
        for j in range(2):
            M_global[indices[i], indices[j]] += M_e[i, j]
            K_global[indices[i], indices[j]] += K_e[i, j]

# Apply damping
C_global = gamma * M_global

# Apply boundary conditions (remove first and last rows and columns)
M_reduced = M_global[1:-1, 1:-1]
K_reduced = K_global[1:-1, 1:-1]
C_reduced = C_global[1:-1, 1:-1]

# Initial conditions
u0 = 2 * np.sin(3 * np.pi * nodes[1:-1] / L)
v0 = np.zeros(N-1)
a0 = np.linalg.solve(M_reduced, -C_reduced @ v0 - K_reduced @ u0)

# Time integration parameters
delta_t = h / c
total_time = 2.0  # Adjust as needed to cover required time points
num_steps = int(total_time / delta_t)

# Newmark-beta parameters
u = u0.copy()
v = v0.copy()
a = a0.copy()

# Times to plot
plot_times = [0, 0.5, 1.0, 1.5, 2.0]
plot_steps = [int(time / delta_t) for time in plot_times]

# Time-stepping loop
for n in range(num_steps):
    # Predictor step
    u_pred = u + delta_t * v + (0.5 - beta) * delta_t**2 * a
    # Compute effective stiffness and force
    K_eff = K_reduced + (gamma / (beta * delta_t)) * C_reduced + (1 / (beta * delta_t**2)) * M_reduced
    F_eff = -C_reduced @ (v + (1 - gamma) * delta_t * a) - K_reduced @ u_pred
    # Solve for new displacement
    u_new = np.linalg.solve(K_eff, F_eff)
    # Update acceleration and velocity
    a_new = (u_new - u - delta_t * v) / (beta * delta_t**2)
    v_new = v + delta_t * ((1 - gamma) * a + gamma * a_new)
    # Update variables for next step
    u, v, a = u_new, v_new, a_new
    
    # Plot at specified times
    if n in plot_steps:
        plt.plot(nodes[1:-1], u, label=f'Time {n * delta_t:.1f}s')

# Final plot settings
plt.xlabel('Position')
plt.ylabel('Displacement')
plt.title('1D Wave Equation Solution at Specific Times')
plt.legend()
plt.show()
