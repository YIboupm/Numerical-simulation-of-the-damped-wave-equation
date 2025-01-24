#we have configured f(x) as Piecewise functions

from xml.etree.ElementInclude import LimitedRecursiveIncludeError
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import math
# Constants
L = 20
C = 20
beta = 3.5
n_terms = 100  # Number of terms in the series approximation

# Function definitions
def f_vectorized(x):
    # Vectorized form of f(x) to handle numpy arrays
    conditions = [
        #(x >= 0) & (x <= 0.0001),
        #(x > 0.0001) & (x <= 0.999),
        #(x > 0.999) & (x <= 1)
        #(x >= 0) & (x <=1)
    ]
    functions = [
        #lambda x: 20000 * x,
        #lambda x: 2 * np.cos(3 * np.pi * x / L),
        #lambda x: 20000 * x - 20000
    ]
    #return np.piecewise(x, conditions, functions)
    return 2 * np.cos(3 * np.pi * x / L)

def g(x):
    #return -2*np.sin(3*np.pi*x/L)
    return 0

# Update calc_H1_H2 function to use scipy.integrate.quad for integration
def calc_H1_H2_quad(n, L, C, beta):
    def integrand_f(x):
        return f_vectorized(x) * np.sin(n * np.pi * x / L)
    
    def integrand_g(x):
        return (g(x) + (beta/2) * f_vectorized(x)) * np.sin(n * np.pi * x / L)
    
    H1, _ = quad(integrand_f, 0, L,limit=100)
    H2, _ = quad(integrand_g, 0, L,limit=100)
    K_n_squared = (C * n * np.pi / L)**2
    H2 = H2 / (math.sqrt(4 * K_n_squared - (beta**2)) / 2)
    
    return 2 * H1 / L, 2 * H2 / L

# Update U(x, t) function to include time dependency using the updated calc_H1_H2 method
def U_xt_quad(x, t, n_terms, L, C, beta):
    U = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        H1, H2 = calc_H1_H2_quad(n, L, C, beta)
        K_n_squared = (C * n * np.pi / L)**2
        omega_n = np.sqrt(4 * K_n_squared - (beta**2))/2
        term = np.exp(-beta/2 * t) * (
            H1 * np.cos(omega_n * t) + H2 * np.sin(omega_n * t)
        ) * np.sin(n * np.pi * x / L)
        U += term
    return U
# Calculate waveform at t=0
x = np.linspace(0, L, 1000)
t = 0.2
U_01_quad = U_xt_quad(x, t, n_terms, L, C, beta)

x = np.linspace(0, L, 1000)
times = np.arange(0, 2.1, 0.1)  # Adjusted to include 0 to 2 seconds

# Create subplots with 5 rows and 5 columns
fig, axs = plt.subplots(5, 5, figsize=(20, 20))  # Adjust the size as needed
fig.suptitle('Comparison of Waveforms at Different Times')

# Flatten the array of axes for easy indexing
axs = axs.flatten()

for i, t in enumerate(times):
    U_01_quad = U_xt_quad(x, t, n_terms, L, C, beta)
    axs[i].plot(x, U_01_quad, label=f'U(x,t) at t={t:.1f}')
    axs[i].plot(x, f_vectorized(x), '--', label='f(x)', color='red', linewidth=2)
    axs[i].set_title(f't={t:.1f}')
    axs[i].legend()
    axs[i].grid(True)

# Hide any unused subplots
for i in range(len(times), len(axs)):
    axs[i].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout
plt.show()