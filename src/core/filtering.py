import numpy as np
from scipy.signal import butter, filtfilt

def apply_butterworth(data, cutoff, fs, order=4):
    """Zero-phase low-pass filter to clean the 20Hz jitter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)
