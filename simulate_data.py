"""
simulate_data.py
-----------------
Simulates EEG data from a classic P300 oddball paradigm.

Why simulated data?
This project starts with simulated EEG so the full pipeline (preprocessing ->
feature extraction -> classification) can be built and tested end-to-end
without depending on a large external download. The simulator is physiologically
informed (1/f background noise + realistic P300 waveform + trial-to-trial
jitter) so the pipeline generalizes to real recordings.

To use REAL data instead, replace this module's output with data loaded via
MNE-Python from a public dataset such as:
  - ERP CORE (Kappenman et al., 2021) - https://osf.io/thsqg/
  - OpenNeuro P300 datasets - https://openneuro.org
The rest of the pipeline (preprocessing.py, features.py, classify.py) expects
the same data format produced here, so it will work unchanged.

Output format
--------------
epochs   : ndarray, shape (n_trials, n_channels, n_samples)
labels   : ndarray, shape (n_trials,)  -> 1 = target/oddball, 0 = standard
ch_names : list of channel names
sfreq    : sampling frequency (Hz)
times    : ndarray, shape (n_samples,) -> epoch time axis in seconds
"""

import numpy as np

CH_NAMES = ["Fz", "Cz", "Pz", "Oz", "P3", "P4"]
SFREQ = 250  # Hz
EPOCH_TMIN, EPOCH_TMAX = -0.2, 0.8  # seconds relative to stimulus onset


def _p300_waveform(times, latency=0.35, width=0.09, amplitude=1.0):
    """Gaussian-shaped P300 centered at `latency` seconds."""
    return amplitude * np.exp(-0.5 * ((times - latency) / width) ** 2)


def _pink_noise(n_samples, rng):
    """Generate 1/f ('pink') noise, closer to real EEG background than white noise."""
    white = rng.standard_normal(n_samples)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]  # avoid divide-by-zero at DC
    spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n_samples)
    return pink / np.std(pink)


def simulate_epochs(n_trials=300, target_ratio=0.2, seed=42):
    """
    Simulate an EEG oddball dataset.

    Parameters
    ----------
    n_trials : int
        Total number of trials (standards + targets).
    target_ratio : float
        Proportion of trials that are the rare "target" (oddball) stimulus.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    epochs, labels, ch_names, sfreq, times
    """
    rng = np.random.default_rng(seed)

    times = np.arange(EPOCH_TMIN, EPOCH_TMAX, 1 / SFREQ)
    n_samples = len(times)
    n_channels = len(CH_NAMES)

    labels = rng.random(n_trials) < target_ratio
    labels = labels.astype(int)

    epochs = np.zeros((n_trials, n_channels, n_samples))

    # Channel-specific P300 amplitude scaling (parietal channels show the
    # largest P300, consistent with the real scalp topography)
    channel_scale = {
        "Fz": 0.4, "Cz": 0.7, "Pz": 1.0, "Oz": 0.3, "P3": 0.8, "P4": 0.8,
    }

    for trial in range(n_trials):
        for ch_idx, ch in enumerate(CH_NAMES):
            background = _pink_noise(n_samples, rng) * 3.0  # microvolt-ish scale

            if labels[trial] == 1:
                # jitter latency/amplitude trial-to-trial, like real ERPs
                latency = rng.normal(0.35, 0.03)
                amplitude = rng.normal(5.0, 1.0) * channel_scale[ch]
                erp = _p300_waveform(times, latency=latency, amplitude=amplitude)
            else:
                erp = np.zeros(n_samples)

            epochs[trial, ch_idx, :] = background + erp

    return epochs, labels, CH_NAMES, SFREQ, times


if __name__ == "__main__":
    epochs, labels, ch_names, sfreq, times = simulate_epochs()
    np.savez(
        "/home/claude/erp-ml-project/data/simulated_erp_dataset.npz",
        epochs=epochs, labels=labels, ch_names=ch_names, sfreq=sfreq, times=times,
    )
    print(f"Simulated {epochs.shape[0]} trials, {epochs.shape[1]} channels, "
          f"{epochs.shape[2]} samples/trial")
    print(f"Targets: {labels.sum()} | Standards: {len(labels) - labels.sum()}")
