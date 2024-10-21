import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import triang  # Poprawiony import


def generate_triangle_signal(length_per_period, num_periods):
    """
    Generates triangle signal.

    Args:
        length_per_period: Number of points in one period of signal
        num_periods: Number of periods in generated signal

    Returns:
        Triangle signal.
    """
    single_period = triang(length_per_period)
    return np.tile(single_period, num_periods) 

def add_noise(signal, noise_level):
    """
    Generates noise to the signal.

    Args:
        signal: Signal on which noise will be applied.
        noise_level (float, 0 to 1): Variance of noise that will be applied on signal.  

    Returns:
        Triangle signal with noise applied.
    """
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise

def moving_average_filter(signal, window_size):
    """
    Applies a moving average filter to signal.
    
    Args:
        signal: Signal to be filtered.
        windows_size: Size of the moving average window.

    Returns:
        Filtered signal.
    """
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def calculate_mse(original, estimated):
    """
    Calculates Mean Squared Error (MSE) between the original and estimated signals.

    Args:
        original: Original (true) signal.
        estimated: Estimated (filtered) signal.

    Returns:
        MSE value.
    """
    n = len(original)
    mse = np.sum((original - estimated) ** 2) / n
    return mse

def find_optimal_window(signal, noisy_signal, max_window):
    """
    Finds the optimal window size for the moving average filter by minimazing MSE.

    Args:
        signal: Original (true) signal.
        noisy_signal: Signal with noise applied.
        max_window: Maximum windows size to consider for optimization.

    Returns:
        Optimal window size and list of MSE values for each window size.
    """
    mse_values = []
    window_range = range(1, max_window + 1)
    
    for window_size in window_range:
        filtered_signal = moving_average_filter(noisy_signal, window_size)
        mse = calculate_mse(signal, filtered_signal)
        mse_values.append(mse)
    
    optimal_window = np.argmin(mse_values) + 1 
    return optimal_window, mse_values

# Signal parameters
length_per_period = 100  # Length of one period
num_periods = 5          # Number of periods
noise_level_1 = 0.1      # Low level of noise variance
noise_level_2 = 0.2      # Higher level of noise variance
max_window_size = 50     # Maximum size of windows to be considered

# Generating signals (original and two with different noises)
original_signal = generate_triangle_signal(length_per_period, num_periods)
noisy_signal_1 = add_noise(original_signal, noise_level_1)
noisy_signal_2 = add_noise(original_signal, noise_level_2)

# Finding optimal window size for each signal
optimal_window_1, mse_values_1 = find_optimal_window(original_signal, noisy_signal_1, max_window_size)
optimal_window_2, mse_values_2 = find_optimal_window(original_signal, noisy_signal_2, max_window_size)

# Estimating signals with optimal window sizes
filtered_signal_1 = moving_average_filter(noisy_signal_1, optimal_window_1)
filtered_signal_2 = moving_average_filter(noisy_signal_2, optimal_window_2)

# Plot 1: Original signal and noised signal (low noise)
plt.figure(figsize=(10, 6))
plt.plot(original_signal, '.', label='Oryginalny sygnał')
plt.plot(noisy_signal_1, '.', label='Zaszumiony sygnał (niski szum)', alpha=0.7)
plt.title('Oryginalny sygnał i zaszumiony sygnał (niski poziom szumu)')
plt.legend()
plt.show()

# Plot 2: Noised signal and estimated signal (low noise)
plt.figure(figsize=(10, 6))
plt.plot(noisy_signal_1, '.', label='Zaszumiony sygnał (niski szum)', alpha=0.7) 
plt.plot(filtered_signal_1, '--', label=f'Estymowany sygnał (okno={optimal_window_1})')
plt.title('Zaszumiony sygnał i estymowany sygnał (niski poziom szumu)')
plt.legend()
plt.show()

# Plot 3: MSE dependence on the window size (low noise)
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_window_size + 1), mse_values_1, label='MSE') 
plt.plot(optimal_window_1, mse_values_1[optimal_window_1 - 1], 'ro', label=f'Optymalny rozmiar okna ({optimal_window_1})') # Optimal window
plt.title('MSE w funkcji wielkości okna (niski poziom szumu)')
plt.xlabel('Rozmiar okna')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Plot 4: Original signal and noised signal (high noise)
plt.figure(figsize=(10, 6))
plt.plot(original_signal, '.', label='Oryginalny sygnał') 
plt.plot(noisy_signal_2, '.', label='Zaszumiony sygnał (wysoki szum)', alpha=0.7) 
plt.title('Oryginalny sygnał i zaszumiony sygnał (wysoki poziom szumu)')
plt.legend()
plt.show()

# Plot 5: Noised signal and estimated signal (high noise)
plt.figure(figsize=(10, 6))
plt.plot(noisy_signal_2, '.', label='Zaszumiony sygnał (wysoki szum)', alpha=0.7)
plt.plot(filtered_signal_2, '--', label=f'Estymowany sygnał (okno={optimal_window_2})')
plt.title('Zaszumiony sygnał i estymowany sygnał (wysoki poziom szumu)')
plt.legend()
plt.show()

# Plot 6: MSE dependence on the window size (high noise)
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_window_size + 1), mse_values_2, label='MSE')  # Wykres MSE
plt.plot(optimal_window_2, mse_values_2[optimal_window_2 - 1], 'ro', label=f'Optymalny rozmiar okna ({optimal_window_2})')  # Optimal window
plt.title('MSE w funkcji wielkości okna (wysoki poziom szumu)')
plt.xlabel('Rozmiar okna')
plt.ylabel('MSE')
plt.legend()
plt.show()
