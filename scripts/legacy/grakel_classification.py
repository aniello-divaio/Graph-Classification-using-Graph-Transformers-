"""
grakel_classification.py
Runs GraKeL graph kernels on the bacteria metabolic pathway dataset
and compares with Graphormer embeddings.

Usage:
    python grakel_classification.py \
        --dataset ~/thesis/data/processed/bacteria_dataset_250.pt \
        --output_dir ~/thesis/plots/grakel
"""

import torch
import numpy as np
import argparse
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import grakel
from grakel import GraphKernel
from grakel.kernels import WeisfeilerLehman, VertexHistogram

CLASS_NAMES = [
    'Actinomycetota', 'Alphaproteobacteria', 'Bacillota - Bacilli',
    'Bacillota - Clostridia', 'Bacteroidota', 'Betaproteobacteria',
    'Epsilonproteobacteria', 'Gammaproteobacteria - Enterobacteria',
    'Gammaproteobacteria - Others', 'Other', 'Tenericutes'
]

SHORT_NAMES = [
    'Actino', 'Alpha', 'Bacilli', 'Clostridia', 'Bacteroid',
    'Beta', 'Epsilon', 'Entero', 'Gamma', 'Other', 'Teneri'
]


def pyg_to_grakel(dataset):
    """
    Convert PyTorch Geometric Data objects to GraKeL graph format.
    GraKeL expects: list of (edge_dict, node_label_dict) tuples
    """
    print("Converting graphs to GraKeL format...", flush=True)
    grakel_graphs = []
    labels = []

    for i, data in enumerate(dataset):
        if i % 500 == 0:
            print(f"  Converting {i+1}/{len(dataset)}...", flush=True)

        edge_index = data.edge_index.numpy()
        num_nodes = data.num_nodes

        # Build edge dict for GraKeL: {(src, dst): edge_label}
        edges = {}
        for j in range(edge_index.shape[1]):
            src = int(edge_index[0, j])
            dst = int(edge_index[1, j])
            edges[(src, dst)] = 1

        # Node labels — use node degree as label
        degree = np.zeros(num_nodes, dtype=int)
        for j in range(edge_index.shape[1]):
            degree[int(edge_index[0, j])] += 1
        # Bin degrees into categories to reduce label space
        degree_binned = np.clip(degree // 3, 0, 10)
        node_labels = {k: int(degree_binned[k]) for k in range(num_nodes)}

        grakel_graphs.append((edges, node_labels))
        labels.append(data.y.item())

    return grakel_graphs, np.array(labels)


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
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='/home/aniello/thesis/data/processed/bacteria_dataset_250.pt')
    parser.add_argument('--output_dir', type=str,
                        default='/home/aniello/thesis/plots/grakel')
    parser.add_argument('--n_iter', type=int, default=3,
                        help='Number of WL iterations (default: 3)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...", flush=True)
    dataset = torch.load(args.dataset)
    print(f"Loaded {len(dataset)} graphs", flush=True)

    # Convert to GraKeL format
    grakel_graphs, y = pyg_to_grakel(dataset)

    print(f"\nClass distribution:", flush=True)
    for c in sorted(np.unique(y)):
        count = (y == c).sum()
        print(f"  Class {c} ({CLASS_NAMES[c]}): {count} graphs", flush=True)

    # Compute WL Kernel matrix
    print(f"\nComputing Weisfeiler-Lehman kernel (n_iter={args.n_iter})...", flush=True)
    print("This may take a while...", flush=True)
    start = time.time()

    gk = WeisfeilerLehman(n_iter=args.n_iter, normalize=True,
                          base_graph_kernel=VertexHistogram)
    K = gk.fit_transform(grakel_graphs)

    elapsed = time.time() - start
    print(f"Kernel computed in {elapsed/60:.1f} minutes", flush=True)
    print(f"Kernel matrix shape: {K.shape}", flush=True)

    # Save kernel matrix for reuse
    np.save(os.path.join(args.output_dir, 'wl_kernel_matrix.npy'), K)
    print("Kernel matrix saved!", flush=True)

    # Classification with SVM
    print("\n=== SVM CLASSIFICATION WITH WL KERNEL ===", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(kernel='precomputed', random_state=42)

    scores = cross_val_score(svm, K, y, cv=cv, scoring='accuracy')
    print(f"\nWL Kernel + SVM:", flush=True)
    print(f"  CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}", flush=True)
    print(f"  Per-fold:    {[f'{s:.3f}' for s in scores]}", flush=True)
    print(f"  Random baseline: {1/11:.3f}", flush=True)

    # Confusion matrix
    y_pred_cv = cross_val_predict(svm, K, y, cv=cv)
    print(f"\nClassification Report:", flush=True)
    print(classification_report(y, y_pred_cv, target_names=SHORT_NAMES), flush=True)

    cm = confusion_matrix(y, y_pred_cv)
    plot_confusion_matrix(
        cm,
        title='Confusion Matrix — WL Kernel + SVM\n(5-fold CV, Bacteria Metabolic Pathway Classification)',
        output_path=os.path.join(args.output_dir, 'confusion_matrix_grakel.png')
    )

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ['Graphormer\n+ Neural Network', 'WL Kernel\n+ SVM']
    accuracies = [0.380, scores.mean()]
    stds = [0.011, scores.std()]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(methods, accuracies, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, width=0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Graphormer vs GraKeL\n(5-fold CV, Bacteria Classification)',
                 fontsize=12, fontweight='bold')
    ax.axhline(y=1/11, color='red', linestyle='--', alpha=0.5,
               label=f'Random baseline ({1/11:.3f})')
    ax.legend(fontsize=9)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=12,
                fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'graphormer_vs_grakel.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: graphormer_vs_grakel.png", flush=True)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
