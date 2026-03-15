# import os
# import numpy as np
# import pandas as pd

# DATASET_PATH = r"c:/Epilipsy project/Dataset_of_Eplipsy"

# # def load_data(data_path=DATASET_PATH):
# #     """
# #     Loads the Bonn EEG dataset from the specified directory.
# #     Relying on file naming conventions: S*.txt, F*.txt, N*.txt, O*.txt, Z*.txt
    
# #     Returns:
# #         dict: A dictionary containing the loaded data for each set.
# #               Keys: 'S', 'F', 'N', 'O', 'Z'
# #               Values: List of numpy arrays (signals).
# #     """
# #     data_sets = {'S': [], 'F': [], 'N': [], 'O': [], 'Z': []}
    
# #     # Ensure strict file sorting for reproducibility
# #     all_files = sorted(os.listdir(data_path))
    
# #     for filename in all_files:
# #         # Check against all prefixes
# #         for key in data_sets.keys():
# #             if filename.upper().startswith(key) and filename.lower().endswith('.txt'):
# #                 file_path = os.path.join(data_path, filename)
# #                 try:
# #                     # Files are single columns of text numbers
# #                     signal = np.loadtxt(file_path)
# #                     data_sets[key].append(signal)
# #                 except Exception as e:
# #                     print(f"Error loading {filename}: {e}")
# #                 break # Found the key, move to next file
                
# #     # Convert lists to numpy arrays for easier handling
# #     # shape will be (num_samples, signal_length)
# #     for key in data_sets:
# #         data_sets[key] = np.array(data_sets[key])
# #         print(f"Loaded set {key}: {data_sets[key].shape}")
        
# #     return data_sets
# def load_data(data_path=DATASET_PATH):
#     """
#     Loads the Bonn EEG dataset from subdirectories named S, F, N, O, Z.
#     Each subfolder contains files like S001.txt ... S100.txt
    
#     Returns:
#         dict: Keys: 'S', 'F', 'N', 'O', 'Z' | Values: numpy arrays (num_files, signal_length)
#     """
#     data_sets = {'S': [], 'F': [], 'N': [], 'O': [], 'Z': []}

#     for key in data_sets.keys():
#         folder_path = os.path.join(data_path, key)  # e.g., .../Dataset/S/

#         if not os.path.isdir(folder_path):
#             print(f"Warning: Folder '{folder_path}' not found. Skipping.")
#             continue

#         all_files = sorted(os.listdir(folder_path))  # Sort for reproducibility

#         for filename in all_files:
#             if filename.lower().endswith('.txt'):
#                 file_path = os.path.join(folder_path, filename)
#                 try:
#                     signal = np.loadtxt(file_path)
#                     data_sets[key].append(signal)
#                 except Exception as e:
#                     print(f"Error loading {filename}: {e}")

#         data_sets[key] = np.array(data_sets[key])
#         print(f"Loaded set '{key}': {data_sets[key].shape}")  # (100, signal_length)

#     return data_sets
# def prepare_data_for_training(data_sets, binary=True):
#     """
#     Prepares data for training. 
#     If binary=True, S (Seizure) is Class 1, others are Class 0.
    
#     Returns:
#         X: Combined features (signals)
#         y: Combined labels
#     """
#     X = []
#     y = []
    
#     for key, signals in data_sets.items():
#         if len(signals) == 0:
#             continue
            
#         X.append(signals)
        
#         # Label generation
#         if binary:
#             # Set S is seizure (1), others are non-seizure (0)
#             label = 1 if key == 'S' else 0
#         else:
#             # Multi-class mapping can be defined here if needed
#             # For now, let's map letters to ints? Or just keep it binary as per common use.
#             # Usually strict binary 'Seizure vs Non-Seizure' is the main goal.
#              label_map = {'Z': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4}
#              label = label_map.get(key, 0)

#         y.append(np.full(len(signals), label))
        
#     X = np.concatenate(X, axis=0)
#     y = np.concatenate(y, axis=0)
    
#     return X, y

# def segment_signals(X, y=None, window_size=174, overlap=0):
#     """
#     Segments the signals into smaller windows.
#     174 samples ~ 1 second at 173.61 Hz.
#     Args:
#         X: List of signals or 2D array (shape: samples x length)
#         y: List of labels (shape: samples). Optional.
#     """
#     new_X = []
#     new_y = []
    
#     # Check if X is a single signal (1D array)
#     # If so, wrap it in a list to treat uniformly
#     # But wait, common use case X is (Num_Signals, Time_Steps).
#     # If X is (Time_Steps,), we should wrap it.
#     if isinstance(X, np.ndarray) and X.ndim == 1:
#         X = [X]
#         if y is not None:
#              y = [y] # Should ideally not happen if y was passed correctly for 1 sample
    
#     for i in range(len(X)):
#         signal = X[i]
#         label = y[i] if y is not None else None
        
#         start = 0
#         step = window_size - overlap
#         if step < 1: step = 1

#         while start + window_size <= len(signal):
#             segment = signal[start:start+window_size]
#             new_X.append(segment)
#             if label is not None:
#                 new_y.append(label)
#             start += step
            
#     if y is None:
#         return np.array(new_X)
        
#     return np.array(new_X), np.array(new_y)
"""
data_loader.py
--------------
Loads the Bonn EEG dataset from subdirectories S, F, N, O, Z.
Each subfolder contains 100 .txt files (e.g. S001.txt ... S100.txt).
"""

import os
import numpy as np

DATASET_PATH = r"c:/Epilipsy project/Dataset_of_Eplipsy"


def load_data(data_path=DATASET_PATH):
    """
    Loads the Bonn EEG dataset from subdirectories named S, F, N, O, Z.

    Returns:
        dict: Keys 'S','F','N','O','Z' → numpy arrays (100, 4097)
    """
    data_sets = {'S': [], 'F': [], 'N': [], 'O': [], 'Z': []}

    for key in data_sets.keys():
        folder_path = os.path.join(data_path, key)

        if not os.path.isdir(folder_path):
            print(f"[WARNING] Folder '{folder_path}' not found. Skipping '{key}'.")
            continue

        all_files = sorted(os.listdir(folder_path))

        for filename in all_files:
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                try:
                    signal = np.loadtxt(file_path)
                    data_sets[key].append(signal)
                except Exception as e:
                    print(f"[ERROR] Loading {filename}: {e}")

        data_sets[key] = np.array(data_sets[key])
        print(f"[INFO] Loaded set '{key}': {data_sets[key].shape}")

    return data_sets


def prepare_data_for_training(data_sets, binary=True):
    """
    Prepares data for training.

    Binary mode  : S=1 (Seizure), F/N/O/Z=0 (Non-Seizure)
    Multiclass   : Z=0, O=1, N=2, F=3, S=4

    Returns:
        X : np.array  (total_samples, signal_length)
        y : np.array  (total_samples,)
    """
    X, y = [], []
    label_map = {'Z': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4}

    for key, signals in data_sets.items():
        if len(signals) == 0:
            continue
        X.append(signals)
        label = (1 if key == 'S' else 0) if binary else label_map[key]
        y.append(np.full(len(signals), label))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    print(f"[INFO] Total samples: {X.shape[0]} | Signal length: {X.shape[1]}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"[INFO] Class distribution: {dict(zip(unique.astype(int), counts))}")

    return X, y


def segment_signals(X, y=None, window_size=178, overlap=0):
    """
    Segments signals into fixed-length windows.
    178 samples = exactly 1 second at 173.61 Hz (matches paper Study 5/6).

    Args:
        X           : 2D array (n_signals, signal_length) or 1D single signal
        y           : 1D label array (n_signals,)  — optional
        window_size : int, samples per window (default 178)
        overlap     : int, samples to overlap between windows (default 0)

    Returns:
        np.array of segments, and optionally np.array of labels
    """
    new_X, new_y = [], []

    # Handle single 1D signal
    if isinstance(X, np.ndarray) and X.ndim == 1:
        X = X[np.newaxis, :]
        if y is not None:
            y = np.array([y])

    step = max(1, window_size - overlap)

    for i in range(len(X)):
        signal = X[i]
        label  = y[i] if y is not None else None
        start  = 0
        while start + window_size <= len(signal):
            new_X.append(signal[start:start + window_size])
            if label is not None:
                new_y.append(label)
            start += step

    if y is None:
        return np.array(new_X)
    return np.array(new_X), np.array(new_y)