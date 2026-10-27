"""
load_real_data.py
------------------
Loads real EEG data via MNE-Python and converts it into the exact same
format produced by simulate_data.py, so the rest of the pipeline
(preprocessing.py, features.py, classify.py, visualize_erp.py) works
UNCHANGED on real data.

Designed around the ERP CORE P3 (oddball) dataset (Kappenman et al., 2021),
but works with any two-condition oddball-style EEG dataset as long as you
set EVENT_ID_TARGET / EVENT_ID_STANDARD correctly for your triggers.

-----------------------------------------------------------------------
SETUP (run locally, on a machine with internet — not in this sandbox)
-----------------------------------------------------------------------
1. Install dependencies:
     pip install mne mne-bids

2. Download ERP CORE P3 data for one subject:
     - Go to https://osf.io/thsqg/ (ERP CORE, OSF repository)
     - Download the "P3" task folder for one subject, e.g. sub-001
     - You want the raw EEG file, e.g. sub-001_P3.set (EEGLAB format)
       plus its accompanying .fdt file in the same folder
     - Place both files in: data/raw/sub-001_P3.set (+ .fdt)

   (Alternative: any OpenNeuro P300 dataset in BIDS format works too —
   see the mne-bids branch below.)

3. Run this script:
     python src/load_real_data.py --file data/raw/sub-001_P3.set

   This produces data/real_erp_dataset.npz in the same format as
   data/simulated_erp_dataset.npz. From there, run the existing pipeline:

     python src/preprocessing.py --input data/real_erp_dataset.npz
     python src/features.py
     python src/classify.py
     python src/visualize_erp.py

-----------------------------------------------------------------------
NOTES ON ERP CORE P3 TASK
-----------------------------------------------------------------------
In the ERP CORE P3 (active visual oddball) task, participants see a stream
of letters (A,B,C,D,E) and press a button whenever the current letter
matches a pre-specified target letter for that block. Event codes follow
the pattern described in the ERP CORE documentation (Kappenman et al.,
2021, NeuroImage) — check the accompanying events.tsv / task description
for exact trigger codes, since they can differ slightly by version.
As a starting point, ERP CORE typically codes:
  - stimulus onset events per letter (varies by identity + target/nontarget)
  - a common approach: events ending in a "target" flag vs "nontarget" flag

Adjust EVENT_ID_TARGET / EVENT_ID_STANDARD below to match the actual
event IDs printed when you run `mne.find_events()` or inspect `raw.annotations`
on your downloaded file — this varies per dataset version, so don't assume
the placeholders below are exactly right without checking.
"""

import argparse
import numpy as np

try:
    import mne
except ImportError:
    mne = None


# ---- CONFIGURE THESE FOR YOUR DATASET ----
EPOCH_TMIN, EPOCH_TMAX = -0.2, 0.8   # seconds, relative to stimulus onset
BASELINE = (None, 0.0)
L_FREQ, H_FREQ = 0.1, 30.0           # band-pass filter range (Hz)

# Placeholder event ID groups — REPLACE with your dataset's real trigger codes.
# Run: raw = mne.io.read_raw_eeglab(path); print(mne.events_from_annotations(raw))
# to see what's available, then group them into target vs standard.
EVENT_ID_TARGET = {"target": 1}
EVENT_ID_STANDARD = {"standard": 2}
# -------------------------------------------


def load_eeglab_file(filepath):
    """Load a .set/.fdt EEGLAB file (e.g. from ERP CORE) with MNE."""
    if mne is None:
        raise ImportError("mne is required: pip install mne mne-bids")
    raw = mne.io.read_raw_eeglab(filepath, preload=True)
    return raw


def load_bids_file(bids_root, subject, task):
    """
    Alternative loader for datasets organized in BIDS format (e.g. many
    OpenNeuro P300 datasets). Requires mne-bids.
    """
    from mne_bids import BIDSPath, read_raw_bids

    bids_path = BIDSPath(subject=subject, task=task, root=bids_root)
    raw = read_raw_bids(bids_path)
    raw.load_data()
    return raw


def make_epochs(raw):
    """Filter, epoch, and label the raw data. Returns MNE Epochs (target + standard combined)."""
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)

    events, event_id_map = mne.events_from_annotations(raw)
    print("Available event IDs found in file:", event_id_map)
    print("--> Check these against EVENT_ID_TARGET / EVENT_ID_STANDARD above.")

    combined_event_id = {**EVENT_ID_TARGET, **EVENT_ID_STANDARD}
    epochs = mne.Epochs(
        raw, events, event_id=combined_event_id,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=BASELINE,
        preload=True, verbose=False,
    )
    return epochs


def epochs_to_arrays(epochs):
    """Convert MNE Epochs to the (epochs, labels, ch_names, sfreq, times) format used by the pipeline."""
    data = epochs.get_data(units="uV")  # (n_trials, n_channels, n_samples)
    target_codes = set(epochs.event_id[k] for k in EVENT_ID_TARGET if k in epochs.event_id)
    labels = np.array([1 if e[2] in target_codes else 0 for e in epochs.events])

    ch_names = epochs.ch_names
    sfreq = epochs.info["sfreq"]
    times = epochs.times

    return data, labels, ch_names, sfreq, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to .set EEGLAB file (e.g. ERP CORE)")
    parser.add_argument("--bids_root", type=str, help="Path to BIDS dataset root (alternative to --file)")
    parser.add_argument("--subject", type=str, default="001")
    parser.add_argument("--task", type=str, default="P3")
    parser.add_argument("--out", type=str, default="/home/claude/erp-ml-project/data/real_erp_dataset.npz")
    args = parser.parse_args()

    if mne is None:
        print("MNE is not installed in this environment. Install it locally with:")
        print("  pip install mne mne-bids")
        return

    if args.file:
        raw = load_eeglab_file(args.file)
    elif args.bids_root:
        raw = load_bids_file(args.bids_root, args.subject, args.task)
    else:
        raise ValueError("Provide either --file (EEGLAB .set) or --bids_root (BIDS dataset)")

    epochs = make_epochs(raw)
    data, labels, ch_names, sfreq, times = epochs_to_arrays(epochs)

    np.savez(args.out, epochs=data, labels=labels, ch_names=ch_names, sfreq=sfreq, times=times)
    print(f"Saved {data.shape[0]} real trials ({labels.sum()} target, {len(labels)-labels.sum()} standard) to {args.out}")


if __name__ == "__main__":
    main()
