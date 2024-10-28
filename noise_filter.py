import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth 


def generate_triangle_signal(frequency, sampling_rate, duration, amplitude=1.0):
    """
    Generates a triangle wave signal with a specified frequency, amplitude, and duration.

    Args:
        frequency: Frequency of the triangle wave (controls the period).
        sampling_rate: Number of samples per second.
        duration: Total duration of the signal in seconds.
        amplitude: Amplitude of the triangle wave.

    Returns:
        A triangle wave signal array.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    triangle_wave = amplitude * sawtooth(2 * np.pi * frequency * t, width=0.5)
    return t, triangle_wave

def add_noise(signal, noise_level):
    """
    Generates noise to the signal.

    Args:
        signal: Signal on which noise will be applied.
        noise_level (float, 0 to 1): Standard deviation of noise that will be applied on signal.  

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
    filtered_signal = np.zeros_like(signal)
    
    for i in range(len(signal)):
        start_index = max(0, i - window_size + 1)
        filtered_signal[i] = np.mean(signal[start_index:i + 1])
    
    return filtered_signal

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

def mse_vs_noise_variance(signal, frequency, sampling_rate, duration, window_size, noise_variances):
    """
    Calculates MSE for different noise variances.

    Args:
        signal: Original (true) signal.
        frequency: Frequency of the triangle wave.
        sampling_rate: Sampling rate in samples per second.
        duration: Total duration of the signal in seconds.
        window_size: Window size for moving average filter.
        noise_variances: List of noise variances to test.

    Returns:
        List of MSE values for each noise variance.
    """
    mse_values = []
    
    for noise_variance in noise_variances:
        noise_level = np.sqrt(noise_variance)  # Standard deviation
        noisy_signal = add_noise(signal, noise_level)
        filtered_signal = moving_average_filter(noisy_signal, window_size)
        mse = calculate_mse(signal, filtered_signal)
        mse_values.append(mse)
    
    return mse_values

# Signal parameters
frequency = 0.5          # Frequency in Hz
sampling_rate = 500      # Sampling rate in samples per second
duration = 6             # Total duration in seconds
amplitude = 1.0          # Amplitude of the triangle wave

noise_level_1 = 0.15     # Low level of noise
noise_level_2 = 0.3      # Higher level of noise
max_window_size = 100    # Maximum size of windows to be considered

# Generating signals (original and two with different noises)
t, original_signal = generate_triangle_signal(frequency, sampling_rate, duration, amplitude)
noisy_signal_1 = add_noise(original_signal, noise_level_1)
noisy_signal_2 = add_noise(original_signal, noise_level_2)

# Finding optimal window size for each signal
optimal_window_1, mse_values_1 = find_optimal_window(original_signal, noisy_signal_1, max_window_size)
optimal_window_2, mse_values_2 = find_optimal_window(original_signal, noisy_signal_2, max_window_size)

# Estimating signals with optimal window sizes
filtered_signal_1 = moving_average_filter(noisy_signal_1, optimal_window_1)
filtered_signal_2 = moving_average_filter(noisy_signal_2, optimal_window_2)

low_noise_mse = calculate_mse(original_signal, filtered_signal_1)
high_noise_mse = calculate_mse(original_signal, filtered_signal_2)
print("Low Noise MSE: ", low_noise_mse, ", High Noise MSE: ", high_noise_mse)

# Plot 1: Original signal
plt.figure(figsize=(10, 6))
plt.plot(t, original_signal, '.', label='Oryginalny sygnał', markersize=2)
plt.title('Oryginalny sygnał')
plt.ylim(-1.6, 1.6)
plt.grid(alpha=0.2)
plt.savefig("plots/original_signal.pdf")
plt.show()

# Plot 2: Original signal and noised signal (low noise)
plt.figure(figsize=(10, 6))
plt.plot(noisy_signal_1, '.', label='Zaszumiony sygnał (niski szum)', alpha=0.7, markersize=2)
plt.plot(original_signal, '.', label='Oryginalny sygnał', markersize=2)
plt.title('Oryginalny sygnał i zaszumiony sygnał (niski poziom szumu)')
plt.ylim(-1.6, 1.6)
plt.grid(alpha=0.2)
plt.legend()
plt.savefig("plots/original_noised_signal_01.pdf")
plt.show()

# Plot 3: Noised signal and estimated signal (low noise)
plt.figure(figsize=(10, 6))
plt.plot(noisy_signal_1, '.', label='Zaszumiony sygnał (niski szum)', alpha=0.7, markersize=2) 
plt.plot(filtered_signal_1, '.', label=f'Estymowany sygnał (okno={optimal_window_1})', markersize=2)
plt.title('Zaszumiony sygnał i estymowany sygnał (niski poziom szumu)')
plt.ylim(-1.6, 1.6)
plt.grid(alpha=0.2)
plt.legend()
plt.savefig("plots/noised_estimated_signal_01.pdf")
plt.show()

# Plot 4: MSE dependence on the window size (low noise)
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_window_size + 1), mse_values_1, label='MSE') 
plt.plot(optimal_window_1, mse_values_1[optimal_window_1 - 1], 'ro', label=f'Optymalny rozmiar okna ({optimal_window_1})') # Optimal window
plt.title('MSE w funkcji wielkości okna (niski poziom szumu)')
plt.xlabel('Rozmiar okna')
plt.ylabel('MSE')
plt.grid(alpha=0.2)
plt.legend()
plt.savefig("plots/mse_window01.pdf")
plt.show()

# Plot 5: Original signal and noised signal (high noise)
plt.figure(figsize=(10, 6))
plt.plot(noisy_signal_2, '.', label='Zaszumiony sygnał (wysoki szum)', alpha=0.7, markersize=2) 
plt.plot(original_signal, '.', label='Oryginalny sygnał', markersize=2) 
plt.title('Oryginalny sygnał i zaszumiony sygnał (wysoki poziom szumu)')
plt.ylim(-1.6, 1.6)
plt.grid(alpha=0.2)
plt.legend()
plt.savefig("plots/original_noised_signal_02.pdf")
plt.show()

# Plot 6: Noised signal and estimated signal (high noise)
plt.figure(figsize=(10, 6))
plt.plot(noisy_signal_2, '.', label='Zaszumiony sygnał (wysoki szum)', alpha=0.7, markersize=2)
plt.plot(filtered_signal_2, '.', label=f'Estymowany sygnał (okno={optimal_window_2})', markersize=2)
plt.title('Zaszumiony sygnał i estymowany sygnał (wysoki poziom szumu)')
plt.ylim(-1.6, 1.6)
plt.legend()
plt.grid(alpha=0.2)
plt.savefig("plots/noised_estimated_signal_02.pdf")
plt.show()

# Plot 7: MSE dependence on the window size (high noise)
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_window_size + 1), mse_values_2, label='MSE')  # Wykres MSE
plt.plot(optimal_window_2, mse_values_2[optimal_window_2 - 1], 'ro', label=f'Optymalny rozmiar okna ({optimal_window_2})')  # Optimal window
plt.title('MSE w funkcji wielkości okna (wysoki poziom szumu)')
plt.xlabel('Rozmiar okna')
plt.ylabel('MSE')
plt.grid(alpha=0.2)
plt.legend()
plt.savefig("plots/mse_window02.pdf")
plt.show()

# MSE vs variance
noise_variances = np.linspace(0.01, 0.5, 20)
optimal_window = optimal_window_1
mse_values = mse_vs_noise_variance(original_signal, frequency, sampling_rate, duration, optimal_window, noise_variances)

# Plot 8: MSE dependence on the variance
plt.figure(figsize=(10, 6))
plt.plot(noise_variances, mse_values, '-o')
plt.title('MSE w funkcji wariancji (dla okna 14)')
plt.xlabel('Wariancja szumu')
plt.ylabel('MSE')
plt.grid(alpha=0.2)
plt.legend()
plt.savefig("plots/mse_vs_variance.pdf")
plt.show()

optimal_windows = []
for variance in noise_variances:
    noisy_signal = add_noise(original_signal, np.sqrt(variance))
    optimal_window, mse_values = find_optimal_window(original_signal, noisy_signal, max_window_size)
    optimal_windows.append(optimal_window) 

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(noise_variances, optimal_windows, marker='o')
plt.title("Optymalne okno w funkcji wariancji szumu")
plt.xlabel("Wariancja szumu")
plt.ylabel("Optymalne okno")
plt.grid(alpha=0.2)
plt.savefig("plots/window_vs_variance.pdf")
plt.show()