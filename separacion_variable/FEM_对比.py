import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 5       # Length of the domain
N = 200     # Number of elements (nodes - 1)
c = 2       # Wave speed
beta = 2    # Damping coefficient
h = L / N   # Size of each element

# Time parameters
dt = 0.001   # Time step
T = 2.5      # Total time for simulation (adjusted to match other methods)
nt = int(T / dt)  # Number of time steps

# Spatial discretization
x = np.linspace(0, L, N+1)

# Initialize the solution vectors
U = np.zeros(N+1)        # Displacement at current time step
U_prev = np.zeros(N+1)   # Displacement at previous time step
V = np.zeros(N+1)        # Velocity at current time step
A = np.zeros(N+1)        # Acceleration at current time step

# Initial condition function
def initial_condition(x):
    return 2 * np.sin(3 * np.pi * x / L)

# Set the initial condition
U[:] = initial_condition(x)
U_prev[:] = U[:]

# Apply boundary conditions
U[0] = U[-1] = 0
U_prev[0] = U_prev[-1] = 0

# Assemble the mass (M) and stiffness (K) matrices
M = np.zeros((N+1, N+1))
K = np.zeros((N+1, N+1))
for i in range(N):
    M_local = (h / 6) * np.array([[2, 1],
                                  [1, 2]])
    K_local = (c**2 / h) * np.array([[1, -1],
                                     [-1, 1]])
    dof = [i, i+1]
    for a in range(2):
        for b in range(2):
            M[dof[a], dof[b]] += M_local[a, b]
            K[dof[a], dof[b]] += K_local[a, b]

# Damping matrix
C = beta * M

# Time-stepping parameters for Newmark-beta method
gamma = 0.5
beta_newmark = 0.25

# Precompute matrices
K_eff = K + (gamma / (beta_newmark * dt)) * C + (1 / (beta_newmark * dt**2)) * M
inv_K_eff = np.linalg.inv(K_eff)

# Initialize storage for displacements over time
displacement_over_time = [U.copy()]  # Store initial condition
time_points = [0, 0.5, 1.0, 1.5, 2.0, 2.5]  # Specific times to store
time_indices = [int(tp / dt) for tp in time_points]
stored_displacements = {tp: None for tp in time_points}

# Time-stepping loop
for step in range(1, nt + 1):
    t_current = step * dt

    # Compute effective force
    F_eff = (-C @ (V + (1 - gamma) * dt * A) - K @ (U + dt * V + ((0.5 - beta_newmark) * dt**2) * A))

    # Solve for acceleration
    A_new = inv_K_eff @ F_eff

    # Update displacement and velocity
    U_new = U + dt * V + ((0.5 - beta_newmark) * dt**2) * A + beta_newmark * dt**2 * A_new
    V_new = V + ((1 - gamma) * dt) * A + gamma * dt * A_new

    # Apply boundary conditions
    U_new[0] = U_new[-1] = 0
    V_new[0] = V_new[-1] = 0
    A_new[0] = A_new[-1] = 0

    # Update variables for next time step
    U, V, A = U_new, V_new, A_new

    # Store displacement at specific time points
    if step in time_indices:
        stored_displacements[t_current] = U.copy()

    # For comparison, you can store displacement at every time step if needed
    displacement_over_time.append(U.copy())

# Plotting the solution at specific times
plt.figure(figsize=(15, 10))
for idx, tp in enumerate(time_points):
    U_tp = stored_displacements.get(tp)
    if U_tp is not None:
        plt.subplot(3, 2, idx + 1)
        plt.plot(x, U_tp, label=f'U(x, t={tp}s)')
        plt.plot(x, initial_condition(x), '--', label='Initial condition', color='red')
        plt.title(f'Displacement at t = {tp:.1f} s')
        plt.xlabel('x')
        plt.ylabel('Displacement U(x, t)')
        plt.legend()
        plt.grid(True)

plt.tight_layout()
plt.show()
