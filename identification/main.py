# main.py
import numpy as np
from simulation import generate_signals, simulate_system
from rls import run_rls, calc_input
from plotting import plot_mse, plot_theta_history, plot_outputs

def identification(model, samples, t, u_k, triangular_signal, z_k, silent = False):
    # Theta* transposed
    ThetaT = model.T

    # Simulation
    y_k = simulate_system(samples, u_k, triangular_signal, z_k, ThetaT)

    # Phi matrix initialization (u_(k), u_(k-1), u_(k-2))
    Phi = np.zeros((samples, 3))
    for k in range(2, samples):
        Phi[k] = np.array([u_k[k], u_k[k-1], u_k[k-2]])

    # Lambda values to consider in  optimization
    lambda_values = np.linspace(0.1, 1, 15)

    # Storing errors for estimated model parameters
    errors = []
    estimated_Theta = []

    # Run estimation for every lambda value
    for lambda_forgetting in lambda_values:
        estimated_Theta_history = run_rls(y_k, Phi, lambda_forgetting)
        # Calculate MSE
        mse = np.mean((ThetaT[2:samples, 1] - estimated_Theta_history[2:samples, 1]) ** 2)
        errors.append(mse)
        estimated_Theta.append(estimated_Theta_history)

    # Finding optimal Lambda
    optimal_lambda_idx = np.argmin(errors)
    optimal_lambda = lambda_values[optimal_lambda_idx]

    print(f"Optimal lambda value: {optimal_lambda:.4f}")
    print(f"Minimal MSE: {errors[optimal_lambda_idx]:.6f}")

    # Error vs Lambda
    not silent and plot_mse(lambda_values, errors, optimal_lambda)

    # Running RLS with optimal lambda
    estimated_Theta_history_optimal = run_rls(y_k, Phi, optimal_lambda)

    # Estimated Parameters vs Time
    not silent and plot_theta_history(t, ThetaT, estimated_Theta_history_optimal, optimal_lambda)

    return estimated_Theta_history_optimal

def adaptiveControl(model, samples, t, u_k, triangular_signal, z_k, d_k):
    # Theta* transposed
    ThetaT = model.T

    # find optimal estimated theta
    estimated_Theta_history_optimal = identification(model, samples, t, u_k, triangular_signal, z_k,True)

    input = calc_input(estimated_Theta_history_optimal, d_k, samples)
    # we calculate input for to get d_k, then symulating system using calculated output
    calculated_y_k = simulate_system(samples, input, triangular_signal, z_k, ThetaT)
    
    plot_outputs(d_k,calculated_y_k)


if __name__ == "__main__":
    # Parameters of experiment
    samples = 500
    frequency = 2
    seed = 42

    t, u_k, triangular_signal, z_k = generate_signals(samples, frequency, seed)
    # --- Identification of Static Model
    print("Identification of Static Model")
    staticModel = np.vstack([
        1.5 * np.ones(samples),      # Theta_0 = 1.5 (const)
        1.0 * np.ones(samples),      # Theta_1 = 1.0 (const)
        0.5 * np.ones(samples)       # Theta_2 = 0.5 (const)
    ])
    identification(staticModel, samples, t, u_k, triangular_signal, z_k)

    # --- Identification of Dynamic Model
    print("Identification of Dynamic Model")
    dynamicModel = np.vstack([
        1.5 * np.ones(samples),      # Theta_0 = 1.5 (const)
        triangular_signal,           # Theta_1 = triangle signal
        0.5 * np.ones(samples)       # Theta_2 = 0.5 (const)
    ])
    identification(dynamicModel, samples, t, u_k, triangular_signal, z_k)

    # --- Adapitve control
    print("Adapitve control")

    #expected output
    d_k = np.random.normal(1, 0.1, samples)

    dynamicModel = np.vstack([
        1.5 * np.ones(samples),      # Theta_0 = 1.5 (const)
        triangular_signal,           # Theta_1 = triangle signal
        0.5 * np.ones(samples)       # Theta_2 = 0.5 (const)
    ])
    adaptiveControl(dynamicModel, samples, t, u_k, triangular_signal, z_k, d_k)
