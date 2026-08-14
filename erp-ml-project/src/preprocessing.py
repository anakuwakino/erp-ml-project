"""
preprocessing.py
-----------------
Standard EEG/ERP preprocessing steps applied per-trial:
  1. Band-pass filter (0.1-30 Hz) to remove slow drift and high-frequency noise
  2. Baseline correction using the pre-stimulus window (-0.2 to 0s)
  3. Simple amplitude-based artifact rejection (drop trials with extreme values)

These are the same core steps taught in Mike X Cohen's neural signal processing
course (filtering, baseline correction) applied here with scipy instead of
MATLAB's Signal Processing Toolbox.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(epochs, sfreq, low=0.1, high=30.0, order=4):
    """Zero-phase Butterworth band-pass filter applied along the time axis."""
    nyq = sfreq / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    filtered = filtfilt(b, a, epochs, axis=-1)
    return filtered


def baseline_correct(epochs, times, baseline=(None, 0.0)):
    """Subtract the mean of the baseline window from each trial/channel."""
    t0 = times[0] if baseline[0] is None else baseline[0]
    t1 = baseline[1]
    mask = (times >= t0) & (times <= t1)
    baseline_mean = epochs[..., mask].mean(axis=-1, keepdims=True)
    return epochs - baseline_mean


def reject_artifacts(epochs, labels, threshold_uv=100.0):
    """
    Drop trials where any channel exceeds +/- threshold_uv at any sample.
    Returns filtered epochs, labels, and a boolean mask of kept trials.
    """
    max_abs = np.max(np.abs(epochs), axis=(1, 2))
    keep = max_abs < threshold_uv
    return epochs[keep], labels[keep], keep


def preprocess(epochs, labels, times, sfreq):
    """Run the full preprocessing pipeline and return cleaned epochs/labels."""
    filtered = bandpass_filter(epochs, sfreq)
    baselined = baseline_correct(filtered, times)
    clean_epochs, clean_labels, keep_mask = reject_artifacts(baselined, labels)
    n_dropped = len(labels) - len(clean_labels)
    print(f"Preprocessing: dropped {n_dropped}/{len(labels)} trials as artifacts")
    return clean_epochs, clean_labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str,
        default="/home/claude/erp-ml-project/data/simulated_erp_dataset.npz",
        help="Path to input .npz (simulated_erp_dataset.npz or real_erp_dataset.npz)",
    )
    parser.add_argument(
        "--output", type=str,
        default="/home/claude/erp-ml-project/data/preprocessed_erp_dataset.npz",
    )
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    epochs, labels, times, sfreq = data["epochs"], data["labels"], data["times"], data["sfreq"]

    clean_epochs, clean_labels = preprocess(epochs, labels, times, sfreq)

    np.savez(
        args.output,
        epochs=clean_epochs, labels=clean_labels,
        ch_names=data["ch_names"], sfreq=sfreq, times=times,
    )
    print(f"Saved {clean_epochs.shape[0]} clean trials to {args.output}")
