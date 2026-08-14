"""
visualize_erp.py
-----------------
Plots the grand-average ERP waveform (target vs. standard trials) at a
parietal channel (Pz), which is the classic figure used in ERP research to
visually confirm a P300 effect before any classification is attempted.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.load("/home/claude/erp-ml-project/data/preprocessed_erp_dataset.npz", allow_pickle=True)
    epochs, labels, times = data["epochs"], data["labels"], data["times"]
    ch_names = list(data["ch_names"])

    ch_idx = ch_names.index("Pz")

    target_avg = epochs[labels == 1, ch_idx, :].mean(axis=0)
    standard_avg = epochs[labels == 0, ch_idx, :].mean(axis=0)
    target_sem = epochs[labels == 1, ch_idx, :].std(axis=0) / np.sqrt((labels == 1).sum())
    standard_sem = epochs[labels == 0, ch_idx, :].std(axis=0) / np.sqrt((labels == 0).sum())

    plt.figure(figsize=(7, 4.5))
    plt.plot(times, target_avg, label="Target (oddball)", color="crimson")
    plt.fill_between(times, target_avg - target_sem, target_avg + target_sem, color="crimson", alpha=0.2)
    plt.plot(times, standard_avg, label="Standard", color="steelblue")
    plt.fill_between(times, standard_avg - standard_sem, standard_avg + standard_sem, color="steelblue", alpha=0.2)

    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.axhline(0, color="gray", linewidth=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title("Grand-Average ERP at Pz: Target vs. Standard")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/claude/erp-ml-project/results/grand_average_erp.png", dpi=150)
    print("Saved grand-average ERP plot to results/grand_average_erp.png")


if __name__ == "__main__":
    main()
