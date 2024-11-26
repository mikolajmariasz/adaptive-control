# plotting.py
import matplotlib.pyplot as plt
import numpy as np

colors = {
    "Theta_real": ['#69b3e7', '#ffcc80', '#6fcf97'],  # Lighter colors for actual
    "Theta_estimated": ['#145a86', '#cc6600', '#1f7a1f']  # Darker colors for estimated
}

def plot_mse(lambda_values, errors, optimal_lambda, ylabel='Średni Błąd Kwadratowy (MSE)', title='Optymalizacja współczynnika zapominania (λ)'):
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, errors, marker='o')
    plt.axvline(optimal_lambda, color='red', linestyle='--', label=f'Optymalna $\lambda = {optimal_lambda:.4f}$')
    plt.title(title)
    plt.xlabel('$\lambda$')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_theta_history(t, Theta, estimated_Theta_history, optimal_lambda):
    plt.figure(figsize=(12, 8))
    for i in range(Theta.shape[1]):
        plt.plot(t, Theta[:, i], label=rf'$\Theta_{i}^*$', 
                 color=colors["Theta_real"][i], linestyle='-')
        plt.plot(t, estimated_Theta_history[:, i], label=rf'$\hat{{\Theta}}_{i}$', 
                 color=colors["Theta_estimated"][i], linestyle='--')
    plt.title(f'Estymacja $\Theta$ (λ = {optimal_lambda:.4f})')
    plt.xlabel('Czas')
    plt.ylabel('$\Theta$')
    plt.legend(loc='lower right') 
    plt.grid(True)
    plt.show()


def plot_adaptive_control_results(y, d_k, time_steps):
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, d_k, label=r'Sygnał zadany $d_k$', linewidth=2)
    plt.plot(y, label='Wyjście systemu $y_k$', linewidth=1.5, linestyle='--')
    plt.axhline(y=0, color='r', linestyle='--', label='Środek $d_k$')
    plt.xlabel('Czas ')
    plt.ylabel('Wyjście systemu $y_k$')
    plt.title('Sterowanie adaptacyjne')
    plt.legend()
    plt.grid()
    plt.show()

# plotting.py
def plot_error_history(error_history, time_steps):
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, error_history, color='blue')
    plt.xlabel('Czas')
    plt.ylabel('Średni błąd kwadratowy (MSE)')
    plt.title('Historia błędu sterowania adaptacyjnego')
    plt.legend()
    plt.grid(True)
    plt.show()
