"""
features.py
------------
Extracts features from preprocessed ERP epochs for classification.

Two feature families are implemented:
  1. Time-domain amplitude features: mean amplitude in the P300 window (250-450ms)
     per channel — the classic "does this look like a P300" feature.
  2. Downsampled waveform features: the epoch itself, downsampled, so the
     classifier can learn shape/timing information beyond a single window mean.

Combining both mirrors what's commonly done in real BCI/ERP classification
pipelines before moving to more complex approaches (e.g. xDAWN + Riemannian
geometry, or deep learning models like EEGNet).
"""

import numpy as np


def p300_window_features(epochs, times, window=(0.25, 0.45)):
    """Mean amplitude per channel within the P300 time window. Shape: (n_trials, n_channels)."""
    mask = (times >= window[0]) & (times <= window[1])
    return epochs[..., mask].mean(axis=-1)


def downsampled_waveform_features(epochs, factor=5):
    """Downsample the time axis by `factor` and flatten channels x time into one feature vector."""
    downsampled = epochs[..., ::factor]
    n_trials = downsampled.shape[0]
    return downsampled.reshape(n_trials, -1)


def extract_features(epochs, times):
    """Combine window-mean and downsampled-waveform features into one feature matrix."""
    f1 = p300_window_features(epochs, times)
    f2 = downsampled_waveform_features(epochs)
    return np.concatenate([f1, f2], axis=1)


if __name__ == "__main__":
    data = np.load("/home/claude/erp-ml-project/data/preprocessed_erp_dataset.npz", allow_pickle=True)
    epochs, labels, times = data["epochs"], data["labels"], data["times"]

    X = extract_features(epochs, times)
    y = labels

    np.savez("/home/claude/erp-ml-project/data/features.npz", X=X, y=y)
    print(f"Feature matrix shape: {X.shape}")
