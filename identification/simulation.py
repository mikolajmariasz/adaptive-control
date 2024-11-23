# simulation.py
import numpy as np
from scipy.signal import sawtooth

def generate_signals(N, frequency, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, 1, N)
    triangular_signal = sawtooth(2 * np.pi * frequency * t, 0.5)
    triangular_signal = (triangular_signal + 1)  # Range: 0 to 2
    z_k = np.random.normal(0, 0.05, N)  # Noise
    u_k = np.random.uniform(-1, 1, N) 
    return t, u_k, triangular_signal, z_k

def simulate_system(N, u_k, triangular_signal, z_k, Theta):
    y_k = np.zeros(N)
    for k in range(2, N):
        phiArr = np.array([u_k[k], u_k[k-1], u_k[k-2]])  # [u_k, u_{k-1}, u_{k-2}]
        y_k[k] = np.dot(phiArr, Theta[k]) + z_k[k]
    return y_k
