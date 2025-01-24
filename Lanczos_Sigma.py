import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math

# Constants
L = 50  # Domain length
C = 500  # Wave speed
beta = 2  # Damping coefficient
n_terms = 200  # Number of terms in the Fourier series
m = 140  # Sigma approximation parameter

# Rectangular pulse function
def f_vectorized(x):
    center = L / 2  # Center of the domain
    width = 20      # Width of the rectangular pulse
    amplitude = 2   # Amplitude of the rectangular pulse
    return np.where((x >= center - width / 2) & (x <= center + width / 2), amplitude, 0)

# Fourier coefficient calculation
def calc_H1_H2_quad(n, L, C, beta):
    def integrand_f(x):
        return f_vectorized(x) * np.sin(n * np.pi * x / L)
    
    def integrand_g(x):
        return 0  # No initial velocity
    
    H1, _ = quad(integrand_f, 0, L)
    H2, _ = quad(integrand_g, 0, L)
    
    K_n_squared = (C * n * np.pi / L) ** 2
    if 4 * K_n_squared <= beta**2:
        raise ValueError(f"Invalid parameters: beta too large for n={n}")
    
    H2 = H2 / (math.sqrt(4 * K_n_squared - (beta**2)) / 2)
    return 2 * H1 / L, 2 * H2 / L

# Lanczos Sigma approximation
def lanczos_sigma(k, m):
    return np.sinc(k / m)

# Fourier series approximation of U(x, t) with Sigma approximation
def U_xt_sigma(x, t, n_terms, L, C, beta, m):
    U = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        H1, H2 = calc_H1_H2_quad(n, L, C, beta)
        K_n_squared = (C * n * np.pi / L) ** 2
        omega_n = np.sqrt(4 * K_n_squared - (beta**2)) / 2
        sigma = lanczos_sigma(n, m)
        term = sigma * np.exp(-beta / 2 * t) * (
            H1 * np.cos(omega_n * t) + H2 * np.sin(omega_n * t)
        ) * np.sin(n * np.pi * x / L)
        U += term
    return U

# Generate x values and compute the Fourier approximation
x = np.linspace(0, L, 1000)
t = 0  # Initial time
U_sigma = U_xt_sigma(x, t, n_terms, L, C, beta, m)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x, U_sigma, label=f'Aproximación Sigma (m={m}, n_terms={n_terms})')
plt.plot(x, f_vectorized(x), label='Pulso rectangular original', color='red', linewidth=2)
plt.axhline(2 + 0.18, color='blue', linestyle='--', label='Pico Gibbs teórico')
plt.axhline(0 - 0.18, color='blue', linestyle='--', label='Valle de Gibbs teórico')
plt.xlabel('x')
plt.ylabel('U(x, t)')
plt.title(f'Aproximación de Lanczos Sigma para el fenómeno de Gibbs (t={t})')
plt.legend()
plt.grid(True)
plt.show()
