# signal_generation.py
import numpy as np
from scipy.signal import sawtooth

def generate_time_samples(N, duration=1.0):
    return np.linspace(0, duration, N)

def generate_triangular_signal(N, frequency, duration=1.0):
    t = generate_time_samples(N, duration)
    triangular_signal = sawtooth(2 * np.pi * frequency * t, 0.5)
    return (triangular_signal + 1)  # Range: 0 to 2

def generate_noise_signal(N, stddev=0.1, seed=42):
    np.random.seed(seed)
    return np.random.normal(0, stddev, N)

def generate_uniform_signal(N, low=-1, high=1, seed=42):
    np.random.seed(seed)
    return np.random.uniform(low, high, N)
