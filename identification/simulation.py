# simulation.py
import numpy as np

def simulate_system(N, u_k, z_k, Theta):
    y_k = np.zeros(N)
    for k in range(2, N):
        phiArr = np.array([u_k[k], u_k[k - 1], u_k[k - 2]])  # [u_k, u_{k-1}, u_{k-2}]
        y_k[k] = np.dot(phiArr, Theta[k]) + z_k[k]
    return y_k
