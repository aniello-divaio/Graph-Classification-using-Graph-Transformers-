"""
evaluate_all.py
Final evaluation of all configurations compared in the thesis, with one
common protocol (stratified 5-fold CV, shuffle, seed 42, same folds for all).

Graphormer (frozen) embeddings x 4 classifiers:
    degree      -> embeddings/bacteria_embeddings.pt          (node feature = out-degree)
    constant    -> embeddings/bacteria_embeddings_v2.pt       (all nodes identical*)
    reaction    -> embeddings/bacteria_embeddings_v2_fixed.pt (node feature = reaction ID)
    * produced by the old sample_graphs.py, which overwrote the reaction IDs
      with 1 for every graph with more than 250 nodes (7,362 of 7,644 graphs).
Weisfeiler-Lehman kernel + SVM on the 250-node sampled graphs:
    degree      -> initial WL label = out-degree // 3, capped at 10
    reaction    -> initial WL label = reaction ID

Classifiers on embeddings are wrapped in a Pipeline(StandardScaler, clf), so
the scaler is fitted on the training folds only.

Outputs (in --out_dir):
    all_results.csv                 one row per configuration (mean, std, folds)
    per_class_f1.csv                F1 per class for every configuration
    reports/<name>.txt              sklearn classification report
    confusion/<name>.npy            confusion matrix (rows = true class)

Usage:
    python scripts/evaluate_all.py --out_dir results/final
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SHORT_NAMES = ['Actino', 'Alpha', 'Bacilli', 'Clostridia', 'Bacteroid',
               'Beta', 'Epsilon', 'Entero', 'Gamma', 'Other', 'Teneri']
SEED = 42


def classifiers():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
        'Linear SVM': SVC(kernel='linear', C=1.0, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED),
        'MLP': MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=500,
                             early_stopping=True, random_state=SEED),
    }


def load(path):
    return torch.load(os.path.expanduser(path), weights_only=False)


# ------------------------------------------------------------------ CV loops
def cv_embeddings(clf, X, y, folds):
    """Returns per-fold accuracies and out-of-fold predictions."""
    y_pred = np.empty_like(y)
    accs = []
    for tr, te in folds:
        model = make_pipeline(StandardScaler(), clf)
        model.fit(X[tr], y[tr])
        y_pred[te] = model.predict(X[te])
        accs.append(accuracy_score(y[te], y_pred[te]))
    return np.array(accs), y_pred


def cv_kernel(K, y, folds):
    y_pred = np.empty_like(y)
    accs = []
    for tr, te in folds:
        svm = SVC(kernel='precomputed', random_state=SEED)
        svm.fit(K[np.ix_(tr, tr)], y[tr])
        y_pred[te] = svm.predict(K[np.ix_(te, tr)])
        accs.append(accuracy_score(y[te], y_pred[te]))
    return np.array(accs), y_pred


# ------------------------------------------------------------- WL kernel
def wl_kernel(dataset, labelling, n_iter):
    from grakel.kernels import VertexHistogram, WeisfeilerLehman
    graphs = []
    for data in dataset:
        ei = data.edge_index.numpy()
        edges = {(int(s), int(d)): 1 for s, d in zip(ei[0], ei[1])}
        if labelling == 'degree':
            deg = np.bincount(ei[0], minlength=data.num_nodes)
            lab = np.clip(deg // 3, 0, 10)
        else:
            lab = data.x.view(-1).long().numpy()
        graphs.append((edges, {k: int(lab[k]) for k in range(data.num_nodes)}))
    gk = WeisfeilerLehman(n_iter=n_iter, normalize=True,
                          base_graph_kernel=VertexHistogram)
    return gk.fit_transform(graphs)


# ------------------------------------------------------------------- output
def record(name, method, features, clf_name, accs, y, y_pred, out, rows, f1rows):
    rows.append({
        'name': name, 'method': method, 'node_features': features,
        'classifier': clf_name,
        'accuracy_mean': round(100 * accs.mean(), 2),
        'accuracy_std': round(100 * accs.std(), 2),
        'macro_f1': round(f1_score(y, y_pred, average='macro'), 3),
        'folds': ' '.join(f'{100 * a:.1f}' for a in accs),
    })
    f1 = f1_score(y, y_pred, average=None, labels=range(len(SHORT_NAMES)))
    f1rows.append({'name': name, **{c: round(v, 3) for c, v in zip(SHORT_NAMES, f1)}})
    np.save(os.path.join(out, 'confusion', f'{name}.npy'),
            confusion_matrix(y, y_pred, labels=range(len(SHORT_NAMES))))
    with open(os.path.join(out, 'reports', f'{name}.txt'), 'w') as fh:
        fh.write(classification_report(y, y_pred, target_names=SHORT_NAMES,
                                       digits=3))
    print(f'  {name:<32} {100 * accs.mean():5.1f} ± {100 * accs.std():.1f}',
          flush=True)


def write_csv(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--emb_degree', default='embeddings/bacteria_embeddings.pt')
    p.add_argument('--emb_constant', default='embeddings/bacteria_embeddings_v2.pt')
    p.add_argument('--emb_reaction', default='embeddings/bacteria_embeddings_v2_fixed.pt')
    p.add_argument('--dataset', default='data/processed/bacteria_dataset_v2_250_fixed.pt',
                   help='250-node sampled graphs WITH reaction IDs (for WL)')
    p.add_argument('--n_iter', type=int, default=3)
    p.add_argument('--out_dir', default='results/final')
    p.add_argument('--skip_wl', action='store_true')
    args = p.parse_args()

    out = os.path.expanduser(args.out_dir)
    for sub in ('', 'confusion', 'reports'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    rows, f1rows = [], []
    t0 = time.time()

    # Reference organism order = the sampled dataset; every embedding file is
    # re-ordered by organism code so that all methods share the same folds.
    dataset = load(args.dataset)
    species_ref = [str(g.species) for g in dataset]
    y_ref = np.array([int(g.y.item()) for g in dataset])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                 .split(np.zeros(len(y_ref)), y_ref))
    print(f'Reference: {len(species_ref)} organisms from {args.dataset}', flush=True)

    # --- Graphormer embeddings ---------------------------------------------
    for features, path in (('degree', args.emb_degree),
                           ('constant', args.emb_constant),
                           ('reaction', args.emb_reaction)):
        res = load(path)
        pos = {str(s): i for i, s in enumerate(res['species'])}
        missing = [s for s in species_ref if s not in pos]
        if missing:
            raise SystemExit(f'{path}: {len(missing)} organisms missing '
                             f'(e.g. {missing[:5]})')
        idx = [pos[s] for s in species_ref]
        X = res['embeddings'].numpy()[idx]
        y = res['labels'].numpy()[idx]
        if not np.array_equal(y, y_ref):
            raise SystemExit(f'{path}: labels disagree with the dataset')
        print(f'\nGraphormer, node features = {features}  ({X.shape})', flush=True)
        for clf_name, clf in classifiers().items():
            accs, y_pred = cv_embeddings(clf, X, y, folds)
            name = f'graphormer_{features}_{clf_name.lower().replace(" ", "_")}'
            record(name, 'Graphormer (frozen)', features, clf_name, accs, y,
                   y_pred, out, rows, f1rows)

    # --- Weisfeiler-Lehman kernel ------------------------------------------
    if not args.skip_wl:
        y = y_ref
        const = sum(int(g.x.unique().numel() == 1) for g in dataset)
        if const > len(dataset) * 0.05:
            raise SystemExit('Reaction IDs missing in the dataset (sampling bug)')
        for features in ('degree', 'reaction'):
            print(f'\nWL kernel, initial labels = {features}', flush=True)
            K = wl_kernel(dataset, features, args.n_iter)
            accs, y_pred = cv_kernel(K, y, folds)
            record(f'wl_{features}_svm', 'WL kernel', features, 'SVM', accs, y,
                   y_pred, out, rows, f1rows)

    write_csv(os.path.join(out, 'all_results.csv'), rows)
    write_csv(os.path.join(out, 'per_class_f1.csv'), f1rows)
    print(f'\nRandom baseline: {100 / 11:.1f}%   '
          f'Majority class: {100 * np.bincount(y_ref).max() / len(y_ref):.1f}%')
    print(f'Saved results to {out}  ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
