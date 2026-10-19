"""
Fixes the confusion matrix to use cross-validation predictions instead of overfitted ones.
"""

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

SHORT_NAMES = ['6010', '6011', '6013', '6014', '6015', '6016']

def main():
    # Load embeddings
    print("Loading embeddings...", flush=True)
    results = torch.load('/home/aniello/Graphormer/archaea_embeddings.pt')
    X = results['embeddings'].numpy()
    y = results['labels'].numpy()

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated predictions (honest evaluation)
    print("Running cross-validated predictions...", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    y_pred_cv = cross_val_predict(clf, X_scaled, y, cv=cv)

    # Confusion matrix from CV predictions
    cm = confusion_matrix(y, y_pred_cv)

    print("\nClassification Report (5-fold CV):", flush=True)
    print(classification_report(y, y_pred_cv, target_names=SHORT_NAMES), flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=SHORT_NAMES,
        yticklabels=SHORT_NAMES,
        ax=ax
    )
    ax.set_title(
        'Confusion Matrix — Logistic Regression\n(5-fold Cross Validation, Archaea Classification)',
        fontsize=13, fontweight='bold', pad=12
    )
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)
    plt.tight_layout()

    output_path = '/home/aniello/Graphormer/plots/confusion_matrix_cv.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}", flush=True)

if __name__ == '__main__':
    main()
