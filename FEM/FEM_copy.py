import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
 
# Parameters
L = 5       # Length of the domain
N = 200     # Number of elements (nodes - 1)
c = 2       # Wave speed

h = L / N   # Size of each element

# Time parameters
dt = 0.001  # Smaller time step for stability
T = 2.5       # Total time for simulation
nt = int(T / dt)  # Number of time steps

# Initialize the solution vectors
U = np.zeros(N+1)  # Displacement
V = np.zeros(N+1)  # Velocity
#V[:] = 1.0
#V[:] = np.linspace(0, 1, N+1)

# Initial condition function (customize this as needed)
def initial_condition(x):
    return 2 * np.sin(3 * np.pi * x / L)

# Set the initial condition
U[:] = initial_condition(np.linspace(0, L, N+1))

# Boundary conditions: fixed ends
U[0] = U[-1] = 0

# Assemble the mass (M) and stiffness (K) matrices
M = np.zeros((N+1, N+1))
K = np.zeros((N+1, N+1))

# Interior nodes
for i in range(1, N):
    M[i,i] +=  h        # Lumped mass matrix - diagonal
    K[i,i] = 2*(c**2) / h      # Element stiffness matrix - diagonal
    K[i,i-1] = -1*(c**2)/ h    # Element stiffness matrix - off diagonal
    K[i,i+1] = -1*(c**2) / h   # Element stiffness matrix - off diagonal

# Boundary nodes - if not fixed, this part needs to be modified
M[0,0] = M[N,N] = h / 2
K[0,0] = K[N,N] = 1   # As this is a boundary node, if it is fixed then the diagonal can be set to 1 or a very large number



# System matrices for the implicit Euler method
A = M + dt**2 * K
invA = np.linalg.inv(A)

# Time-stepping loop
# Store the displacement at each time step for comparison with the initial condition
displacement_over_time = [U.copy()]  # Store initial condition

for t in range(1, nt + 1):
    # System vector for the implicit Euler method
    B = M @ U + dt * M @ V
    # Solve for the next time step
    U_new = invA @ B
    # Compute velocity
    V_new = (U_new - U) / dt

    # Update the solution vectors
    U, V = U_new, V_new

    # Apply boundary conditions
    U[0] = U[-1] = 0

    # Store the displacement
    displacement_over_time.append(U.copy())

# Plotting the solution over time alongside the initial condition for comparison
x = np.linspace(0, L, N+1)
plt.figure(figsize=(12, 6))

# Plot initial condition
plt.plot(x, initial_condition(x), label='Initial condition (t=0)', linestyle='--')

# Plot displacement over time
for i, U in enumerate(displacement_over_time):
    if i % (nt//5) == 0:
        plt.plot(x, U, label=f'Displacement (t={i*dt:.2f}s)')

plt.xlabel('x')
plt.ylabel('Displacement U(x, t)')
plt.title('Finite Element Method Solution Over Time')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.grid(True)
plt.show()
