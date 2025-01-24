import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 50  # Domain length
n_terms = 200  # Number of terms in Fourier series
m = 100  # Sigma approximation parameter

# Rectangular pulse function
def rectangular_pulse(x):
    center = L / 2
    width = 20
    amplitude = 2
    return np.where((x >= center - width / 2) & (x <= center + width / 2), amplitude, 0)

# Sigma kernel function
def lanczos_sigma(k, m):
    return np.sinc(k / m)

# Fourier coefficients calculation
def calculate_coefficients(n, L):
    # Fourier coefficients for a rectangular pulse
    if n == 0:
        return L / 2  # DC component
    else:
        # Fourier coefficient for cosine and sine
        a_n = (2 / L) * 2 * np.sin(n * np.pi * 10 / L) / (n * np.pi)  # Normalized coefficient
        b_n = 0  # No sine component for symmetric pulse
        return a_n, b_n

# Compute Fourier spectrum with and without Sigma kernel
freqs = np.arange(1, n_terms + 1)
original_spectrum = []
sigma_spectrum = []

for n in freqs:
    a_n, _ = calculate_coefficients(n, L)
    original_spectrum.append(np.abs(a_n))  # Original amplitude
    sigma_spectrum.append(np.abs(a_n) * lanczos_sigma(n, m))  # Sigma-weighted amplitude
plt.figure(figsize=(12, 6))
plt.plot(freqs, original_spectrum, label="Espectro original (sin sigma)", linewidth=2)
plt.plot(freqs, sigma_spectrum, label=f"Espectro Sigma (m={m})", linewidth=2, linestyle="--")
plt.xlabel("Frecuencia (n)")
plt.ylabel("Amplitud")
plt.title("Espectro de Fourier con y sin núcleo Sigma")
plt.legend()
plt.grid(True)
plt.show()
# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(freqs, original_spectrum, label="Espectro original (sin sigma)", linewidth=2)
plt.plot(freqs, sigma_spectrum, label=f"Espectro Sigma (m={m})", linewidth=2, linestyle="--")
plt.yscale("log")  # 使用对数坐标
plt.xlabel("Frecuencia (n)")
plt.ylabel("Amplitud (escala logarítmica)")
plt.title("Espectro de Fourier con y sin núcleo sigma (escala logarítmica)")
plt.legend()
plt.grid(True)
plt.show()

