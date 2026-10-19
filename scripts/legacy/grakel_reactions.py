import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from grakel.kernels import VertexHistogram, WeisfeilerLehman
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.svm import SVC

SHORT_NAMES = ['Actino', 'Alpha', 'Bacilli', 'Clostridia', 'Bacteroid',
               'Beta', 'Epsilon', 'Entero', 'Gamma', 'Other', 'Teneri']


def pyg_to_grakel(dataset):
    graphs, labels = [], []
    for i, data in enumerate(dataset):
        if i % 1000 == 0:
            print(f"  Converting {i+1}/{len(dataset)}...", flush=True)
        ei = data.edge_index.numpy()
        edges = {(int(s), int(d)): 1 for s, d in zip(ei[0], ei[1])}
        reaction_ids = data.x.view(-1).long().tolist()
        node_labels = {k: int(reaction_ids[k]) for k in range(data.num_nodes)}
        graphs.append((edges, node_labels))
        labels.append(int(data.y.item()))
    return graphs, np.array(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='~/thesis/data/processed/bacteria_dataset_v2_250_fixed.pt')
    p.add_argument('--output_dir', default='~/thesis/plots/grakel_reactions')
    p.add_argument('--n_iter', type=int, default=3)
    p.add_argument('--save_kernel', action='store_true')
    args = p.parse_args()
    out = os.path.expanduser(args.output_dir)
    os.makedirs(out, exist_ok=True)

    dataset = torch.load(os.path.expanduser(args.dataset))
    print(f"Loaded {len(dataset)} graphs", flush=True)
    const = sum(int(g.x.unique().numel() == 1) for g in dataset)
    print(f"Graphs with constant node features: {const} (should be ~0)", flush=True)
    if const > len(dataset) * 0.05:
        raise SystemExit("Reaction IDs missing: use bacteria_dataset_v2_250_fixed.pt")

    graphs, y = pyg_to_grakel(dataset)

    print(f"\nComputing WL kernel (n_iter={args.n_iter}, labels = reaction IDs)...", flush=True)
    t0 = time.time()
    gk = WeisfeilerLehman(n_iter=args.n_iter, normalize=True,
                          base_graph_kernel=VertexHistogram)
    K = gk.fit_transform(graphs)
    print(f"Kernel computed in {(time.time()-t0)/60:.1f} min, shape {K.shape}", flush=True)
    if args.save_kernel:
        np.save(os.path.join(out, 'wl_kernel_reactions.npy'), K)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm = SVC(kernel='precomputed', random_state=42)
    scores = cross_val_score(svm, K, y, cv=cv, scoring='accuracy')
    print("\n=== WL KERNEL (reaction labels) + SVM ===", flush=True)
    print(f"  CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}", flush=True)
    print(f"  Per-fold:    {[f'{s:.3f}' for s in scores]}", flush=True)
    print(f"  Random baseline: {1/11:.3f}", flush=True)

    y_pred = cross_val_predict(svm, K, y, cv=cv)
    print("\nClassification Report:", flush=True)
    print(classification_report(y, y_pred, target_names=SHORT_NAMES), flush=True)

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES, ax=ax)
    ax.set_title('Confusion Matrix — WL Kernel (reaction labels) + SVM\n'
                 '(5-fold CV, Bacteria Metabolic Pathway Classification)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    path = os.path.join(out, 'confusion_matrix_grakel_reactions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}\nDone!", flush=True)


if __name__ == '__main__':
    main()
