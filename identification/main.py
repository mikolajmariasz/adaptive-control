# main.py
import numpy as np
from signal_generation import generate_time_samples, generate_triangular_signal, generate_noise_signal, generate_uniform_signal
from identification import identification
from adaptive_control import adaptive_control
from plotting import plot_mse, plot_adaptive_control_results, plot_error_history

if __name__ == "__main__":
    # Parameters of experiment
    samples = 500
    frequency = 2
    seed = 42

    # Generate signals
    t = generate_time_samples(samples)
    triangular_signal = generate_triangular_signal(samples, frequency)
    z_k = generate_noise_signal(samples, 0.05, seed=seed)
    u_k = generate_uniform_signal(samples, seed=seed)

    # --- Identification of Static Model
    print("Identification of Static Model")
    staticModel = np.vstack([
        1.5 * np.ones(samples),  # Theta_0 = 1.5 (const)
        1.0 * np.ones(samples),  # Theta_1 = 1.0 (const)
        0.5 * np.ones(samples)   # Theta_2 = 0.5 (const)
    ])
    identification(staticModel, samples, t, u_k, z_k)

    # --- Identification of Dynamic Model
    print("Identification of Dynamic Model")
    dynamicModel = np.vstack([
        1.5 * np.ones(samples),      # Theta_0 = 1.5 (const)
        triangular_signal,           # Theta_1 = triangle signal
        0.5 * np.ones(samples)       # Theta_2 = 0.5 (const)
    ])
    identification(dynamicModel, samples, t, u_k, z_k)

    # --- Adaptive Control
    print("Running Adaptive Control")
    lambda_values = np.linspace(0.1, 1, 15)
    errors = []

    for lambda_factor in lambda_values:
        b_history, cumulative_mse, y_k, d_k, time_steps, error_history = adaptive_control(
            dynamicModel, z_k, num_samples=samples, lambda_factor=lambda_factor)
        mse = cumulative_mse[-1]  
        errors.append(mse)

    optimal_lambda_idx = np.argmin(errors)
    optimal_lambda = lambda_values[optimal_lambda_idx]

    print(f"Optymalna wartość lambda: {optimal_lambda:.4f}")
    print(f"Minimalny MSE: {errors[optimal_lambda_idx]:.6f}")

    plot_mse(lambda_values, errors, optimal_lambda,
             ylabel='Średni Błąd Kwadratowy (MSE) pomiędzy $y_k$ a $d_k$',
             title='Optymalizacja współczynnika zapominania (λ)')

    b_history, cumulative_mse, y_k, d_k, time_steps, error_history = adaptive_control(
        dynamicModel, z_k, num_samples=samples, lambda_factor=optimal_lambda)

    plot_adaptive_control_results(y_k, d_k, time_steps)

    plot_error_history(error_history, time_steps)
