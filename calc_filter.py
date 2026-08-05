from scipy.signal import butter
import numpy as np

def get_coeffs(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    print(f"Sample Rate: {fs} Hz, Cutoff: {cutoff} Hz")
    print(f"b = [{', '.join([f'{x:.6f}' for x in b])}]")
    print(f"a = [{', '.join([f'{x:.6f}' for x in a])}]")

if __name__ == "__main__":
    get_coeffs(3.0, 25.0)
