import numpy as np
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