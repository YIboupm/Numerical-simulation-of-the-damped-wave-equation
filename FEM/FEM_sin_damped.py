import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Revised code for stability with smaller time step and implicit time-stepping method

# Parameters
L = 5       # Length of the domain
N = 200      # Number of elements (nodes - 1)
c = 2       # Wave speed
#2*c*np.pi/L
xi = 2*c*np.pi/L   # Damping coefficient 2
h = L / N   # Size of each element

# Time parameters
dt = 0.001   # Smaller time step for stability
T = 2        # Total time for simulation
nt = int(T / dt)  # Number of time steps

# Initialize the solution vectors
U = np.zeros(N+1)  # Displacement
V = np.zeros(N+1)  # Velocity

# Initial condition function (customize this as needed)
def initial_condition(x):
    return 2 * np.sin(3 * np.pi * x / L)
    return 2*np.sinc(3 * np.pi * x / L)
    center = L / 2  # Center of the domain
    width = 2      # Width of the rectangular pulse
    amplitude = 2   # Amplitude of the rectangular pulse
    return np.where((x >= center - width / 2) & (x <= center + width / 2), amplitude, 0)
    
    
    return 2*np.sinc(3 * np.pi * x / L)
    return 2 * np.cos(3 * np.pi * x / L)
    return 2 * np.sin(3 * np.pi * x / L)
    
# Set the initial condition
U[:] = initial_condition(np.linspace(0, L, N+1))

# Boundary conditions: fixed ends
U[0] = U[-1] = 0

# Assemble the mass (M) and stiffness (K) matrices
M = np.zeros((N+1, N+1))
K = np.zeros((N+1, N+1))
for i in range(1, N):
    M[i,i] +=  h        # Lumped mass matrix - diagonal
    K[i,i] = 2*(c**2) / h      # Element stiffness matrix - diagonal
    K[i,i-1] = -1*(c**2)/ h    # Element stiffness matrix - off diagonal
    K[i,i+1] = -1*(c**2) / h   # Element stiffness matrix - off diagonal

# Boundary conditions: fixed ends
M[0,0] = M[N,N] = h / 2
K[0,0] = K[N,N] = 1 

# Damping matrix
C = xi * M

# System matrices for the implicit Euler method
A = M + dt**2 * K + dt * C
A_sparse = csr_matrix(A)
#invA = np.linalg.inv(A)

# Time-stepping loop
# Store the displacement at each time step for comparison with the initial condition
displacement_over_time = [U.copy()]  # Store initial condition

for t in range(1, nt + 1):
    # System vector for the implicit Euler method
    B = M @ U + dt * M @ V+C @ (dt*U)
    # Solve for the next time step
    U_new = spsolve(A_sparse, B)
    #U_new = invA @ B
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
time_steps_to_display = [0, int(nt*0.25), int(nt*0.5), int(nt*0.75), nt]
fig, axes = plt.subplots(len(time_steps_to_display), 1, figsize=(10, 15))

# Set unified y-axis and x-axis limits
y_min, y_max = np.min(displacement_over_time), np.max(displacement_over_time)

for idx, step in enumerate(time_steps_to_display):
    axes[idx].plot(x, displacement_over_time[step], label=f't={step*dt:.2f}s')
    axes[idx].set_title(f'Displazamiento en t={step*dt:.2f}s')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('U(x, t)')
    axes[idx].set_xlim([0, L])  # Unified x-axis
    axes[idx].set_ylim([y_min, y_max])  # Unified y-axis
    axes[idx].legend()
    axes[idx].grid(True)

plt.tight_layout()
plt.show()