# adaptive_control.py
import numpy as np

def adaptive_control(dynamicModel, z_k, num_samples=1000, beta=100, lambda_factor=0.999):
    true_b0 = dynamicModel[0] 
    true_b1 = dynamicModel[1]
    true_b2 = dynamicModel[2]

    b_est = np.zeros(3)
    b_est[0] = 0.1 
    P = beta * np.eye(3)
    b_history = [b_est.copy()]

    u = np.zeros(num_samples)
    y = np.zeros(num_samples)
    d_k = np.random.uniform(-1, 1, num_samples) 

    error_squared = np.zeros(num_samples)
    y_pred_history = np.zeros(num_samples)
    error_history = np.zeros(num_samples)

    for k in range(2, num_samples):
        if b_est[0] == 0:
            b_est[0] = 1e-6
        u[k] = (d_k[k] - b_est[1] * u[k - 1] - b_est[2] * u[k - 2]) / b_est[0]

        y[k] = (
            true_b0[k] * u[k]
            + true_b1[k] * u[k - 1]
            + true_b2[k] * u[k - 2]
            + z_k[k]
        )
        phi_k = np.array([u[k], u[k - 1], u[k - 2]]).reshape(-1, 1)

        y_pred = phi_k.T @ b_est
        error = y[k] - y_pred

        P_phi = P @ phi_k
        denominator = lambda_factor + phi_k.T @ P_phi
        P = (1 / lambda_factor) * (P - (P_phi @ P_phi.T) / denominator)

        b_est = b_est + (P @ phi_k).flatten() * error
        error_squared[k] = (y[k] - d_k[k]) ** 2
        error_history[k] = error_squared[k]

        b_history.append(b_est.copy())

        y_pred_history[k] = y_pred

    b_history = np.array(b_history)

    cumulative_mse = np.cumsum(error_squared) / np.arange(1, num_samples + 1)
    time_steps = np.arange(num_samples)

    return b_history, cumulative_mse, y, d_k, time_steps, error_history
