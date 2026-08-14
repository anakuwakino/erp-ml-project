"""
classify.py
-----------
Trains and evaluates a classifier to distinguish target (oddball/P300) vs.
standard trials from extracted ERP features.

Uses a Linear Discriminant Analysis (LDA) classifier — a standard, well-
established baseline in the BCI/ERP literature (e.g. P300 speller systems) —
alongside an SVM for comparison. Evaluation uses stratified k-fold cross-
validation, appropriate given the class imbalance (targets are the rare
oddball stimulus).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score
)


def evaluate_model(model, X, y, cv):
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return y_pred, y_proba


def main():
    data = np.load("/home/claude/erp-ml-project/data/features.npz")
    X, y = data["X"], data["y"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
    }

    results = {}
    for name, model in models.items():
        y_pred, y_proba = evaluate_model(model, X, y, cv)
        auc = roc_auc_score(y, y_proba)
        results[name] = {"y_pred": y_pred, "y_proba": y_proba, "auc": auc}

        print(f"\n=== {name} ===")
        print(f"ROC AUC: {auc:.3f}")
        print(classification_report(y, y_pred, target_names=["Standard", "Target"]))

    # --- Plot ROC curves ---
    plt.figure(figsize=(6, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y, res["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Target vs. Standard Trial Classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/claude/erp-ml-project/results/roc_curve.png", dpi=150)
    print("\nSaved ROC curve to results/roc_curve.png")

    # --- Plot confusion matrix for best model ---
    best_name = max(results, key=lambda k: results[k]["auc"])
    cm = confusion_matrix(y, results[best_name]["y_pred"])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Standard", "Target"])
    plt.yticks([0, 1], ["Standard", "Target"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("/home/claude/erp-ml-project/results/confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix to results/confusion_matrix.png (best model: {best_name})")


if __name__ == "__main__":
    main()
