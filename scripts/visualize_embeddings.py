"""
visualize_embeddings.py
2D projections (t-SNE and UMAP) of the bacterial metabolic graphs, coloured
by taxonomic class, for three representations:

    graphormer_constant  frozen Graphormer embeddings, constant node features
    graphormer_reaction  frozen Graphormer embeddings, reaction-ID node features
    wl_reaction          WL kernel with reaction-ID labels, projected from the
                         kernel distance d(i,j) = sqrt(2 - 2 K(i,j))
                         (K is normalised, so K(i,i) = 1)

For each representation one figure with t-SNE (left) and UMAP (right) is saved
in --fig_dir as .pdf and .png. Projections are only a visual aid: distances in
a t-SNE/UMAP plot are not reliable, the quantitative comparison is in
evaluate_all.py.

Usage:
    python scripts/visualize_embeddings.py --fig_dir figures
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:
    umap = None
    print('umap-learn not installed: only t-SNE will be drawn '
          '(pip install umap-learn)')

SEED = 42
CLASSES = ['Actinomycetota', 'Alphaproteobacteria', 'Bacillota – Bacilli',
           'Bacillota – Clostridia', 'Bacteroidota', 'Betaproteobacteria',
           'Epsilonproteobacteria', 'Gammaproteob. – Enterobacteria',
           'Gammaproteob. – Others', 'Other', 'Tenericutes']
# 11 distinct qualitative colours (Tableau 10 + one), "Other" in grey
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#e377c2', '#bcbd22', '#17becf', '#9a9a9a', '#393b79']
TITLES = {
    'graphormer_constant': 'Graphormer (frozen) – constant node features',
    'graphormer_reaction': 'Graphormer (frozen) – reaction-ID node features',
    'wl_reaction': 'WL kernel – reaction-ID labels',
}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'savefig.dpi': 300})


def tsne(data, precomputed=False):
    kw = dict(n_components=2, perplexity=30, random_state=SEED,
              init='random' if precomputed else 'pca',
              metric='precomputed' if precomputed else 'euclidean')
    try:
        return TSNE(max_iter=1000, **kw).fit_transform(data)
    except TypeError:  # scikit-learn < 1.5 calls it n_iter
        return TSNE(n_iter=1000, **kw).fit_transform(data)


def run_umap(data, precomputed=False):
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        random_state=SEED,
                        metric='precomputed' if precomputed else 'euclidean')
    return reducer.fit_transform(data)


def wl_distance(dataset_path, n_iter=3):
    from grakel.kernels import VertexHistogram, WeisfeilerLehman
    dataset = torch.load(os.path.expanduser(dataset_path), weights_only=False)
    graphs, y = [], []
    for g in dataset:
        ei = g.edge_index.numpy()
        edges = {(int(s), int(d)): 1 for s, d in zip(ei[0], ei[1])}
        lab = g.x.view(-1).long().tolist()
        graphs.append((edges, {k: int(lab[k]) for k in range(g.num_nodes)}))
        y.append(int(g.y.item()))
    K = WeisfeilerLehman(n_iter=n_iter, normalize=True,
                         base_graph_kernel=VertexHistogram).fit_transform(graphs)
    D = np.sqrt(np.clip(2.0 - 2.0 * K, 0, None))
    np.fill_diagonal(D, 0.0)
    return D, np.array(y)


def plot(panels, y, title, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y))  # random drawing order: no class on top
    for ax, (name, xy) in zip(axes, panels):
        ax.scatter(xy[order, 0], xy[order, 1], s=3, linewidths=0, alpha=0.75,
                   c=[COLORS[c] for c in y[order]], rasterized=True)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#c3c2b7')
            s.set_linewidth(0.6)
    handles = [plt.Line2D([], [], marker='o', ls='', markersize=5, color=c,
                          label=n) for n, c in zip(CLASSES, COLORS)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.17))
    fig.suptitle(title, fontsize=10, y=1.0)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved', path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--emb_constant', default='embeddings/bacteria_embeddings_v2.pt')
    p.add_argument('--emb_reaction', default='embeddings/bacteria_embeddings_v2_fixed.pt')
    p.add_argument('--dataset', default='data/processed/bacteria_dataset_v2_250_fixed.pt')
    p.add_argument('--fig_dir', default='figures')
    p.add_argument('--skip_wl', action='store_true')
    args = p.parse_args()
    fig_dir = os.path.expanduser(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    for key, path in (('graphormer_constant', args.emb_constant),
                      ('graphormer_reaction', args.emb_reaction)):
        res = torch.load(os.path.expanduser(path), weights_only=False)
        X = StandardScaler().fit_transform(res['embeddings'].numpy())
        y = res['labels'].numpy()
        print(f'\n{key}: {X.shape}  t-SNE...', flush=True)
        panels = [('t-SNE', tsne(X))]
        if umap is not None:
            print('  UMAP...', flush=True)
            panels.append(('UMAP', run_umap(X)))
        plot(panels, y, TITLES[key], os.path.join(fig_dir, f'fig_projection_{key}'))

    if not args.skip_wl:
        print('\nwl_reaction: computing WL kernel...', flush=True)
        D, y = wl_distance(args.dataset)
        print('  t-SNE...', flush=True)
        panels = [('t-SNE', tsne(D, precomputed=True))]
        if umap is not None:
            print('  UMAP...', flush=True)
            panels.append(('UMAP', run_umap(D, precomputed=True)))
        plot(panels, y, TITLES['wl_reaction'],
             os.path.join(fig_dir, 'fig_projection_wl_reaction'))
    print('\nDone!')


if __name__ == '__main__':
    main()
