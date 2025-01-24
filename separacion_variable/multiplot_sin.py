from cmath import cos
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# Constants
L = 5
C = 2
beta = 1
n_terms = 200  # Number of terms in the series approximation

# Function definitions
def f_vectorized(x):
    # Return the sine function
    return 2 * np.sin(3 * np.pi * x / L)

def g(x):
    return 0  # g(x) is defined to be 0 for all x

def calc_H1_H2_quad(n, L, C, beta):
    def integrand_f(x):
        return f_vectorized(x) * np.sin(n * np.pi * x / L)
    
    def integrand_g(x):
        return (g(x) + (beta / 2) * f_vectorized(x)) * np.sin(n * np.pi * x / L)
    
    H1, _ = quad(integrand_f, 0, L, limit=100)
    H2, _ = quad(integrand_g, 0, L, limit=100)
    K_n_squared = (C * n * np.pi / L)**2
    H2 = H2 / (math.sqrt(4 * K_n_squared - (beta**2)) / 2)
    
    return 2 * H1 / L, 2 * H2 / L

def U_xt_quad(x, t, n_terms, L, C, beta):
    U = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        H1, H2 = calc_H1_H2_quad(n, L, C, beta)
        K_n_squared = (C * n * np.pi / L)**2
        omega_n = np.sqrt(4 * K_n_squared - (beta**2)) / 2
        term = np.exp(-beta * t / 2) * (
            H1 * np.cos(omega_n * t) + H2 * np.sin(omega_n * t)
        ) * np.sin(n * np.pi * x / L)
        U += term
    return U

# Calculate waveform at specific times
x = np.linspace(0, L, 1000)
specific_times = [0, 0.5, 1.0, 1.5, 2.0]

# Create subplots with 1 row and 5 columns to match the number of times
fig, axs = plt.subplots(1, 5, figsize=(25, 5))  # Adjust the size for horizontal layout
fig.suptitle('Waveforms at Specific Times')

for i, t in enumerate(specific_times):
    U_01_quad = U_xt_quad(x, t, n_terms, L, C, beta)
    axs[i].plot(x, U_01_quad, label=f'U(x,t) at t={t:.1f}')
    axs[i].plot(x, f_vectorized(x), '--', label='f(x)', color='red', linewidth=2)
    axs[i].set_title(f't={t:.1f}')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout
plt.show()
