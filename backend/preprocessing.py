# import numpy as np
# from scipy import signal
# from dwt import apply_dwt

# def butter_bandpass(lowcut, highcut, fs, order=5):
#     """
#     Designs a Butterworth band-pass filter.
#     """
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = signal.butter(order, [low, high], btype='band')
#     return b, a

# def apply_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=173.61, order=5):
#     """
#     Applies the Butterworth band-pass filter to the data.
    
#     Args:
#         data (np.array): Input EEG signal.
#         lowcut (float): Lower frequency cutoff (Hz).
#         highcut (float): Upper frequency cutoff (Hz).
#         fs (float): Sampling frequency (Hz).
#         order (int): Order of the filter.
        
#     Returns:
#         np.array: Filtered signal.
#     """
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = signal.lfilter(b, a, data)
#     return y

# def z_score_normalize(data, mean=None, std=None):
#     """
#     Applies Z-score normalization.
    
#     CRITICAL: To prevent zero-information leakage, the mean and std
#     MUST be calculated from the TRAINING set only and applied to the
#     test/validation sets.

#     Args:
#         data (np.array): Input data.
#         mean (float, optional): Mean of the training set. If None, calculates from data.
#         std (float, optional): Std dev of the training set. If None, calculates from data.
        
#     Returns:
#         np.array: Normalized data.
#         tuple: (mean, std) used for normalization.
#     """
#     if mean is None:
#         mean = np.mean(data)
#     if std is None:
#         std = np.std(data)
    
#     normalized_data = (data - mean) / (std + 1e-8) # Add epsilon to avoid division by zero
#     return normalized_data, (mean, std)

# def preprocess_pipeline(raw_signal, fs=173.61):
#     """
#     Full preprocessing pipeline:
#     1. Band-pass Filter (0.5-40Hz)
#     2. DWT Feature Extraction (optional step here, or kept separate)
    
#     Note: Z-score normalization should be applied AFTER splitting the dataset.
#     """
#     # 1. Band-pass Filter
#     filtered_signal = apply_bandpass_filter(raw_signal, fs=fs)
    
#     return filtered_signal
"""
preprocessing.py
----------------
EEG signal preprocessing: bandpass filtering and z-score normalisation.

Key design rules (industry standard):
  - Use zero-phase filtfilt (not lfilter) to avoid phase distortion.
  - Filter order = 2 (matches paper: Butterworth bandpass 0.5–50 Hz).
  - Z-score stats MUST be computed on TRAINING data only,
    then applied to validation / test / live data.
"""

import numpy as np
from scipy.signal import butter, filtfilt


# ── Filter ───────────────────────────────────────────────────────────────────

def butter_bandpass(lowcut=0.5, highcut=50.0, fs=173.61, order=2):
    """
    Designs a zero-phase Butterworth band-pass filter.
    Paper uses order=2, [0.5, 50] Hz (Study 2/4/6).
    """
    nyq  = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=173.61, order=2):
    """
    Applies zero-phase Butterworth bandpass filter to a 1-D signal.

    Args:
        data    : 1D np.array
        lowcut  : lower cutoff Hz (default 0.5)
        highcut : upper cutoff Hz (default 50.0)
        fs      : sampling frequency Hz (default 173.61)
        order   : filter order (default 2 — matches paper)

    Returns:
        1D np.array (filtered signal, same length)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)   # zero-phase — no distortion


# ── Normalisation ─────────────────────────────────────────────────────────────

def z_score_normalize(data, mean=None, std=None):
    """
    Z-score normalisation.

    CRITICAL: Always pass `mean` and `std` from the TRAINING set when
    processing validation, test, or live data.  Computing stats from the
    data itself (mean=None) is only safe for isolated verification runs.

    Args:
        data : np.array (any shape)
        mean : float, training mean   (None → compute from data)
        std  : float, training std    (None → compute from data)

    Returns:
        normalised : np.array same shape as data
        stats      : (mean, std) tuple
    """
    if mean is None:
        mean = float(np.mean(data))
    if std is None:
        std  = float(np.std(data))
    normalised = (data - mean) / (std + 1e-8)
    return normalised, (mean, std)


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def preprocess_pipeline(raw_signals, lowcut=0.5, highcut=50.0, fs=173.61):
    """
    Applies bandpass filter to every signal in the dataset.

    Handles:
        - 1D array  → single signal
        - 2D array  → (n_signals, signal_length), filters each row

    NOTE: Z-score is intentionally NOT applied here.
          Call z_score_normalize() separately after train/test split.

    Args:
        raw_signals : np.array, 1D or 2D
        lowcut      : lower cutoff Hz
        highcut     : upper cutoff Hz
        fs          : sampling frequency Hz

    Returns:
        np.array, same shape as input (filtered)
    """
    if raw_signals.ndim == 1:
        return apply_bandpass_filter(raw_signals, lowcut, highcut, fs)

    # 2D: apply filter row-by-row
    return np.array([
        apply_bandpass_filter(sig, lowcut, highcut, fs)
        for sig in raw_signals
    ])