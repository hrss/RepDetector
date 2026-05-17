import numpy as np
from collections import deque, Counter

def smooth_predictions(raw_preds, window_size=12):
    """Majority vote smoothing to prevent flickering predictions."""
    if not raw_preds:
        return []
    smoothed = []
    history = deque([raw_preds[0]] * window_size, maxlen=window_size)
    for p in raw_preds:
        history.append(p)
        vote = Counter(history).most_common(1)[0][0]
        smoothed.append(vote)
    return smoothed

def extract_features(window):
    """
    20 carefully selected features that force the tree to learn
    universal physics (Accel), technique nuances (Gyro), and 
    repetition cadence (FFT).
    """
    # --- 8 ACCELEROMETER FEATURES (The Universal Physics) ---

    # 1. Total Kinetic Energy
    acc_smv = np.sqrt(window[:, 0] ** 2 + window[:, 1] ** 2 + window[:, 2] ** 2)
    acc_smv_mean = np.mean(acc_smv)

    # 2. Vertical Displacement & Orientation (Z-Axis) -> Crucial for Squats/Deadlifts
    acc_z = window[:, 2]
    acc_z_mean = np.mean(acc_z)
    acc_z_range = np.max(acc_z) - np.min(acc_z)
    acc_z_std = np.std(acc_z)

    # 3. Forward/Back Displacement & Orientation (Y-Axis) -> Crucial for arm hanging vs rack
    acc_y = window[:, 1]
    acc_y_mean = np.mean(acc_y)
    acc_y_range = np.max(acc_y) - np.min(acc_y)

    # 4. Side-to-Side Displacement & Orientation (X-Axis)
    acc_x = window[:, 0]
    acc_x_mean = np.mean(acc_x)
    acc_x_range = np.max(acc_x) - np.min(acc_x)

    # --- 8 GYROSCOPE FEATURES (The Technique Nuances) ---

    # 5. Total Rotational Energy
    gyro_smv = np.sqrt(window[:, 3] ** 2 + window[:, 4] ** 2 + window[:, 5] ** 2)
    gyro_smv_mean = np.mean(gyro_smv)
    gyro_smv_std = np.std(gyro_smv)

    # 6. Specific Wrist Rotations (Peaks and Valleys of technique)
    gyro_x = window[:, 3]
    gyro_x_max = np.max(gyro_x)
    gyro_x_min = np.min(gyro_x)

    # 7. Specific Wrist Rotations (Peaks and Valleys of technique)
    gyro_y = window[:, 4]
    gyro_y_max = np.max(gyro_y)
    gyro_y_min = np.min(gyro_y)

    # 8. Specific Wrist Rotations (Peaks and Valleys of technique)
    gyro_z = window[:, 5]
    gyro_z_max = np.max(gyro_z)
    gyro_z_min = np.min(gyro_z)

    # --- 4 FFT FEATURES (The Repetition Cadence) ---
    # These help distinguish between fast/slow movements and identify periodic patterns
    
    # FFT for Accel SMV
    acc_fft = np.abs(np.fft.rfft(acc_smv - acc_smv_mean))
    acc_dom_freq_idx = np.argmax(acc_fft[1:]) + 1  # Skip DC component
    acc_max_power = acc_fft[acc_dom_freq_idx]

    # FFT for Gyro SMV
    gyro_fft = np.abs(np.fft.rfft(gyro_smv - gyro_smv_mean))
    gyro_dom_freq_idx = np.argmax(gyro_fft[1:]) + 1  # Skip DC component
    gyro_max_power = gyro_fft[gyro_dom_freq_idx]

    return np.array([
        # The 8 Accel Features
        acc_smv_mean, acc_z_mean, acc_z_range, acc_z_std,
        acc_y_mean, acc_y_range, acc_x_mean, acc_x_range,
        # The 8 Gyro Features
        gyro_smv_mean, gyro_smv_std,
        gyro_x_max, gyro_x_min, gyro_y_max, gyro_y_min, gyro_z_max, gyro_z_min,
        # The 4 FFT Features
        acc_dom_freq_idx, acc_max_power,
        gyro_dom_freq_idx, gyro_max_power
    ])
