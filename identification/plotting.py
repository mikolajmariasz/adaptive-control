# plotting.py
import matplotlib.pyplot as plt
import numpy as np

def plot_mse(lambda_values, errors, optimal_lambda):
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, errors, marker='o', label='Średni błąd kwadratowy MSE dla różnych $\lambda$')
    plt.axvline(optimal_lambda, color='red', linestyle='--', label=f'Optymalna $\lambda = {optimal_lambda:.4f}$')
    plt.title('Optymalizacja współczynnika zapominania dla zmiennych $\Theta$')
    plt.xlabel('$\lambda$')
    plt.ylabel('Średni błąd kwadratowy (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_theta_history(t, Theta, estimated_Theta_history, optimal_lambda):
    plt.figure(figsize=(12, 8))
    plt.plot(t, estimated_Theta_history[:, 0], label=r'$\hat{\Theta}_0$', color='C0')
    plt.plot(t, estimated_Theta_history[:, 1], label=r'$\hat{\Theta}_1$', color='C1')
    plt.plot(t, estimated_Theta_history[:, 2], label=r'$\hat{\Theta}_2$', color='C2')
    plt.plot(t, Theta[:, 0], 'C0--', label=r'$\Theta_0$ (rzeczywiste)')
    plt.plot(t, Theta[:, 1], 'C1--', label=r'$\Theta_1$ (rzeczywiste)')
    plt.plot(t, Theta[:, 2], 'C2--', label=r'$\Theta_2$ (rzeczywiste)')
    plt.title(f'$\Theta*$ z zapominaniem (λ = {optimal_lambda:.4f})')
    plt.xlabel('Czas')
    plt.ylabel('Estymowane wartości $\Theta$')
    plt.legend()
    plt.grid(True)
    plt.show()
