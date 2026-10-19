"""
classify_bacteria.py
Trains and evaluates classifiers on Graphormer embeddings for bacteria dataset.

Usage:
    python classify_bacteria.py \
        --embeddings ~/thesis/embeddings/bacteria_embeddings.pt \
        --output_dir ~/thesis/plots/bacteria
"""

import torch
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Correct bacteria class names (sorted alphabetically = class index order)
CLASS_NAMES = [
    'Actinomycetota',
    'Alphaproteobacteria',
    'Bacillota - Bacilli',
    'Bacillota - Clostridia',
    'Bacteroidota',
    'Betaproteobacteria',
    'Epsilonproteobacteria',
    'Gammaproteobacteria - Enterobacteria',
    'Gammaproteobacteria - Others',
    'Other',
    'Tenericutes'
]

SHORT_NAMES = [
    'Actino',
    'Alpha',
    'Bacilli',
    'Clostridia',
    'Bacteroid',
    'Beta',
    'Epsilon',
    'Entero',
    'Gamma',
    'Other',
    'Teneri'
]


def plot_confusion_matrix(cm, title, output_path):
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='viridis',
        xticklabels=SHORT_NAMES,
        yticklabels=SHORT_NAMES,
        ax=ax
    )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}", flush=True)


def evaluate_classifier(name, clf, X, y, cv):
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print(f"\n{name}:", flush=True)
    print(f"  CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}", flush=True)
    print(f"  Per-fold:    {[f'{s:.3f}' for s in scores]}", flush=True)
    return scores.mean(), scores.std()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str,
                        default='/home/aniello/thesis/embeddings/bacteria_embeddings.pt')
    parser.add_argument('--output_dir', type=str,
                        default='/home/aniello/thesis/plots/bacteria')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    print("Loading embeddings...", flush=True)
    results = torch.load(args.embeddings)
    X = results['embeddings'].numpy()
    y = results['labels'].numpy()

    print(f"Embeddings: {X.shape}", flush=True)
    print(f"Classes: {np.unique(y)}", flush=True)
    print(f"\nClass distribution:", flush=True)
    for c in sorted(np.unique(y)):
        count = (y == c).sum()
        print(f"  Class {c} ({CLASS_NAMES[c]}): {count} graphs", flush=True)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (Linear)':        SVC(kernel='linear', C=1.0, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Neural Network':      MLPClassifier(
                                    hidden_layer_sizes=(512, 256),
                                    max_iter=500,
                                    random_state=42,
                                    early_stopping=True
                               ),
    }

    print("\n=== 5-FOLD CROSS VALIDATION RESULTS ===", flush=True)
    results_summary = {}
    for name, clf in classifiers.items():
        mean, std = evaluate_classifier(name, clf, X_scaled, y, cv)
        results_summary[name] = (mean, std)

    # Best classifier
    best_name = max(results_summary, key=lambda k: results_summary[k][0])
    best_mean, best_std = results_summary[best_name]
    print(f"\n=== BEST CLASSIFIER: {best_name} ===", flush=True)
    print(f"Accuracy: {best_mean:.3f} ± {best_std:.3f}", flush=True)
    print(f"Random baseline: {1/11:.3f}", flush=True)

    # Cross-validated confusion matrix
    print(f"\n=== CROSS-VALIDATED CONFUSION MATRIX ({best_name}) ===", flush=True)
    best_clf = classifiers[best_name]
    y_pred_cv = cross_val_predict(best_clf, X_scaled, y, cv=cv)
    print(classification_report(y, y_pred_cv, target_names=SHORT_NAMES), flush=True)

    cm = confusion_matrix(y, y_pred_cv)
    plot_confusion_matrix(
        cm,
        title=f'Confusion Matrix — {best_name}\n(5-fold CV, Bacteria Metabolic Pathway Classification)',
        output_path=os.path.join(args.output_dir, 'confusion_matrix_cv.png')
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(results_summary.keys())
    means = [results_summary[n][0] for n in names]
    stds  = [results_summary[n][1] for n in names]
    colors = ['#2ecc71' if n == best_name else '#3498db' for n in names]
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Classifier Comparison — Graphormer Embeddings\n(5-fold CV, Bacteria Dataset)',
                 fontsize=12, fontweight='bold')
    ax.axhline(y=1/11, color='red', linestyle='--', alpha=0.5,
               label=f'Random baseline (1/11 = {1/11:.3f})')
    ax.legend(fontsize=9)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'classifier_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: classifier_comparison.png", flush=True)
    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()