# plotting.py
import matplotlib.pyplot as plt
import numpy as np

colors = {
    "Theta_real": ['#69b3e7', '#ffcc80', '#6fcf97'],  # Jaśniejsze kolory dla rzeczywistych
    "Theta_estimated": ['#145a86', '#cc6600', '#1f7a1f']  # Ciemniejsze kolory dla estymowanych
}
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
    # Dla każdej zmiennej Theta rysujemy estymowane i rzeczywiste wartości w odpowiednich kolorach
    for i in range(Theta.shape[1]):
        plt.plot(t, Theta[:, i], label=rf'$\Theta_{i}$ (rzeczywiste)', 
                 color=colors["Theta_real"][i], linestyle='-')
        plt.plot(t, estimated_Theta_history[:, i], label=rf'$\hat{{\Theta}}_{i}$ (estymowane)', 
                 color=colors["Theta_estimated"][i], linestyle='--')
    plt.title(f'$\Theta*$ z zapominaniem (λ = {optimal_lambda:.4f})')
    plt.xlabel('Czas')
    plt.ylabel('Estymowane wartości $\Theta$')
    plt.legend()
    plt.grid(True)
    plt.show()
