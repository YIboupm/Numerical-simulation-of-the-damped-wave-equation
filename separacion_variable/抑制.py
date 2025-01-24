from cmath import cos, sinh, cosh
from random import betavariate
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# Constants
L = 5
C = 2
beta=2
beta = (2 * C * np.pi) / L
n_terms = 200  # Number of terms in the series approximation
m = 200  # Sigma kernel parameter

# Function definitions
def f_vectorized(x):
    return 2*np.sin(3*np.pi*x/L)
    return 2 * np.sinc(3 * np.pi * x / L)

def lanczos_sigma(k, m):
    return np.sinc(k / m)

def g(x):
    return 0  # g(x) is defined to be 0 for all x

def calc_H1_H2_quad(n, L, C, beta):
    def integrand_f(x):
        return f_vectorized(x) * np.sin(n * np.pi * x / L)
    
    def integrand_g(x):
        return (g(x) + (beta / 2) * f_vectorized(x)) * np.sin(n * np.pi * x / L)
    
    H1, _ = quad(integrand_f, 0, L, limit=100)
    H2, _ = quad(integrand_g, 0, L, limit=100)
    
    K_n_squared = (C * n * np.pi / L) ** 2
    if math.isclose(beta, (2 * C * np.pi) / L, rel_tol=1e-9):  # Critical damping
        H2 = H2 / (beta / 2)
    elif beta**2 < 4 * K_n_squared:  # Small damping
        H2 = H2 / (math.sqrt(4 * K_n_squared - (beta**2)) / 2)
    else:  # Large damping
        H2 = H2 / (math.sqrt((beta**2) - 4 * K_n_squared) / 2)
    
    return 2 * H1 / L, 2 * H2 / L

def U_xt_quad(x, t, n_terms, L, C, beta, m):
    U = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        K_n_squared = (C * n * np.pi / L) ** 2
        H1, H2 = calc_H1_H2_quad(n, L, C, beta)
        sigma = lanczos_sigma(n, m)
        if math.isclose(beta, (2 * C * np.pi) / L, rel_tol=1e-9):  # Critical damping
            term = sigma * np.exp(-beta * t / 2) * (
                H1 + H2 * t
            ) * np.sin(n * np.pi * x / L)
        elif beta**2 < 4 * K_n_squared:  # Small damping
            omega_n = np.sqrt(4 * K_n_squared - (beta**2)) / 2
            term = sigma * np.exp(-beta * t / 2) * (
                H1 * np.cos(omega_n * t) + H2 * np.sin(omega_n * t)
            ) * np.sin(n * np.pi * x / L)
        else:  # Large damping
            omega_n = np.sqrt((beta**2) - 4 * K_n_squared) / 2
            term = sigma * np.exp(-beta * t / 2) * (
                H1 * np.cosh(omega_n * t) + H2 * np.sinh(omega_n * t)
            ) * np.sin(n * np.pi * x / L)
        
        U += term
    return U

# Calculate waveform at specific times
x = np.linspace(0, L, 1000)
specific_times = [0, 0.5, 1.0, 1.5, 2.0]

# Create subplots with 5 rows and 1 column for a vertical layout
fig, axs = plt.subplots(len(specific_times), 1, figsize=(8, 20))  # 8 width, 20 height
fig.suptitle('Formas de onda en momentos específicos con aproximación Sigma de Lanczos', fontsize=16)

for i, t in enumerate(specific_times):
    U_01_quad = U_xt_quad(x, t, n_terms, L, C, beta, m)
    axs[i].plot(x, U_01_quad, label=f'U(x,t) at t={t:.1f}')
    axs[i].plot(x, f_vectorized(x), '--', label='f(x)', color='red', linewidth=2)
    axs[i].set_title(f'Time = {t:.1f}', fontsize=12)
    axs[i].set_xlabel('x')
    axs[i].set_ylabel('U(x,t)')
    axs[i].legend()
    axs[i].grid(True)

# Adjust layout to prevent overlapping titles and labels
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
