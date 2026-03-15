# import numpy as np
# import pywt

# def apply_dwt(signal, wavelet='db4', level=5):
#     """
#     Applies Discrete Wavelet Transform (DWT) to the input signal.
    
#     This function decomposes the EEG signal into different frequency sub-bands
#     using the specified wavelet. This allows for the isolation of seizure-specific
#     frequencies (e.g., spikes in specific bands).

#     Args:
#         signal (np.array): The input EEG signal (1D array).
#         wavelet (str): The wavelet family and order to use (default: 'db4').
#         level (int): The decomposition level (default: 5). 
#                      For 173.61 Hz sampling rate, level 5 is appropriate 
#                      to separate Delta, Theta, Alpha, Beta, and Gamma bands.

#     Returns:
#         list: A list of coefficients [cA_n, cD_n, cD_n-1, ..., cD_1].
#               cA_n: Approximation coefficients (lowest frequency).
#               cD_i: Detail coefficients (higher frequencies at level i).
#     """
#     coeffs = pywt.wavedec(signal, wavelet, level=level)
#     return coeffs

# def reconstruct_signal(coeffs, wavelet='db4'):
#     """
#     Reconstructs the signal from DWT coefficients.
#     Useful for visualizing specific frequency bands by zeroing out others.
#     """
#     return pywt.waverec(coeffs, wavelet)

# def get_band_energy(coeffs):
#     """
#     Calculates the energy of each wavelet sub-band.
#     Energy features are commonly used for epilepsy detection.
    
#     Args:
#         coeffs (list): DWT coefficients.
        
#     Returns:
#         np.array: Energy values for each sub-band.
#     """
#     energy = [np.sum(np.square(c)) for c in coeffs]
#     return np.array(energy)
"""
dwt.py
------
Discrete Wavelet Transform utilities for EEG feature extraction.
"""

import numpy as np
import pywt


def apply_dwt(signal, wavelet='db4', level=5):
    """
    Applies Discrete Wavelet Transform to decompose EEG into sub-bands.

    At 173.61 Hz, level-5 db4 maps to:
        cA5 → Delta  (0–2.7 Hz)
        cD5 → Theta  (2.7–5.4 Hz)
        cD4 → Alpha  (5.4–10.9 Hz)
        cD3 → Beta   (10.9–21.7 Hz)
        cD2 → Gamma  (21.7–43.4 Hz)
        cD1 → High   (43.4–86.8 Hz)

    Args:
        signal  : 1D np.array EEG signal
        wavelet : wavelet family (default 'db4')
        level   : decomposition level (default 5)

    Returns:
        list: [cA5, cD5, cD4, cD3, cD2, cD1]
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs


def reconstruct_signal(coeffs, wavelet='db4'):
    """Reconstruct signal from DWT coefficients."""
    return pywt.waverec(coeffs, wavelet)


def get_band_energy(coeffs):
    """
    Energy of each wavelet sub-band (used as ML features).

    Returns:
        np.array of shape (n_bands,)
    """
    return np.array([np.sum(np.square(c)) for c in coeffs])


def get_dwt_features(signal, wavelet='db4', level=5):
    """
    Extracts statistical features from each DWT sub-band.
    Features per band: mean, std, energy, max, min  → 5 × (level+1) total.

    Returns:
        np.array of shape (5 * (level+1),)
    """
    coeffs = apply_dwt(signal, wavelet, level)
    features = []
    for c in coeffs:
        features.extend([
            np.mean(c),
            np.std(c),
            np.sum(np.square(c)),   # energy
            np.max(np.abs(c)),
            np.min(np.abs(c)),
        ])
    return np.array(features)