# rls.py
import numpy as np

def run_rls(y_k, Phi, lambda_forgetting, initial_P=np.eye(3)*1000):
    N = Phi.shape[0]
    estimated_Theta = np.zeros(3)
    P = initial_P.copy()
    estimated_Theta_history = np.zeros((N, 3))

    for k in range(2, N):
        Phi_k = Phi[k]
        y_pred = np.dot(Phi_k, estimated_Theta)
        error = y_k[k] - y_pred
        denominator = lambda_forgetting + Phi_k @ P @ Phi_k
        K = (P @ Phi_k) / denominator 
        estimated_Theta += K * error
        P = (P - np.outer(K, Phi_k) @ P) / lambda_forgetting
        estimated_Theta_history[k] = estimated_Theta

    return estimated_Theta_history
