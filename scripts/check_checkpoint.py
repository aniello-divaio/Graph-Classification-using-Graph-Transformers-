"""
check_checkpoint.py
Why does the frozen Graphormer get worse when the nodes carry reaction IDs?

Graphormer turns the integer feature of each node into a vector by looking it
up in a table (atom_encoder, 4609 x 768). With one feature column the row used
for a node with value x is x + 1 (Graphormer's convert_to_single_emb adds an
offset of 1 to the first column). During pretraining on PCQM4Mv1 only the rows
of the atom features were updated; the others keep their random
initialisation, N(0, 0.02^2), whose expected norm is 0.02 * sqrt(768) = 0.55.

The script measures the norm of every row and checks which rows each of the
three node-feature configurations actually uses:
    constant     every node has value 1        -> row 2
    node degree  out-degree of the node        -> rows degree + 1
    reaction ID  reaction index of the node    -> rows ID + 1
(degrees and reaction IDs are read from the 250-node sampled dataset)

Outputs:
    results/checkpoint_norms.txt         summary used in the thesis
    figures/fig_checkpoint_norms.pdf/png norm of every row of atom_encoder

Usage:
    python scripts/check_checkpoint.py
"""

import argparse
import os

import numpy as np
import torch

INIT_NORM = 0.02 * np.sqrt(768)
TRAINED_THRESHOLD = 1.0  # rows above this norm were clearly updated


def find_table(state):
    keys = [k for k in state if k.endswith('atom_encoder.weight')]
    if len(keys) != 1:
        raise SystemExit(f'atom_encoder.weight not found (candidates: {keys})')
    return keys[0], state[keys[0]].float()


def summary(name, rows, norms):
    rows = np.unique(rows)
    rows = rows[rows < len(norms)]
    n = norms[rows]
    trained = int((n > TRAINED_THRESHOLD).sum())
    return (f'{name:<13} {len(rows):5d} rows used | mean norm {n.mean():.3f} '
            f'(min {n.min():.3f}, max {n.max():.3f}) | '
            f'{trained} trained ({100 * trained / len(rows):.1f}%)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=os.path.expanduser(
        '~/.cache/torch/hub/checkpoints/checkpoint_best_pcqm4mv1.pt'))
    p.add_argument('--dataset', default='data/processed/bacteria_dataset_v2_250_fixed.pt')
    p.add_argument('--out_txt', default='results/checkpoint_norms.txt')
    p.add_argument('--fig_dir', default='figures')
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt)
    key, table = find_table(state)
    norms = table.norm(dim=1).numpy()
    trained_rows = np.where(norms > TRAINED_THRESHOLD)[0]

    dataset = torch.load(args.dataset, weights_only=False)
    reaction_rows = np.unique(np.concatenate(
        [g.x.view(-1).long().numpy() for g in dataset])) + 1
    degree_rows = np.unique(np.concatenate(
        [np.bincount(g.edge_index[0].numpy(), minlength=g.num_nodes)
         for g in dataset])) + 1
    constant_rows = np.array([2])

    lines = [
        f'checkpoint: {args.checkpoint}',
        f'table: {key}, shape {tuple(table.shape)}',
        f'expected norm of an untrained row: 0.02 * sqrt(768) = {INIT_NORM:.3f}',
        f'rows with norm > {TRAINED_THRESHOLD}: {len(trained_rows)} '
        f'(indices {trained_rows.min()}-{trained_rows.max()}); '
        f'mean norm {norms[trained_rows].mean():.3f} '
        f'(min {norms[trained_rows].min():.3f}, max {norms[trained_rows].max():.3f})',
        f'rows with norm <= {TRAINED_THRESHOLD}: mean norm '
        f'{norms[norms <= TRAINED_THRESHOLD].mean():.3f}',
        '',
        'rows used by each node-feature configuration:',
        summary('constant', constant_rows, norms),
        summary('node degree', degree_rows, norms),
        summary('reaction ID', reaction_rows, norms),
        f'(row 2, constant feature: norm {norms[2]:.3f})',
    ]
    os.makedirs(os.path.dirname(args.out_txt) or '.', exist_ok=True)
    with open(args.out_txt, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nSaved {args.out_txt}')

    # figure: norm of every row, rows used by the reaction IDs highlighted
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'savefig.dpi': 300})
    PURPLE, TEAL, GREEN, INK2 = '#472d7b', '#25858e', '#6ece58', '#52514e'
    last = max(reaction_rows.max(), degree_rows.max(), trained_rows.max()) + 20
    idx = np.arange(last)
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.scatter(idx, norms[:last], s=2, color='#c9c8c3', lw=0, label='all rows')
    ax.scatter(reaction_rows, norms[reaction_rows], s=2, color=PURPLE, lw=0,
               label='rows used by reaction IDs')
    ax.scatter(degree_rows, norms[degree_rows], s=10, color=TEAL, lw=0,
               marker='s', label='rows used by node degree')
    ax.scatter([2], [norms[2]], s=40, color=GREEN, edgecolor='white', lw=0.8,
               zorder=5, label='row used by the constant feature')
    ax.axhline(INIT_NORM, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(last, INIT_NORM, ' random init.', color=INK2, fontsize=7.5,
            va='center', clip_on=False)
    ax.set_xlabel('Row of the node-feature embedding table (feature value + 1)')
    ax.set_ylabel('L2 norm')
    ax.set_xlim(-20, last)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, markerscale=2.5, loc='upper right')
    os.makedirs(args.fig_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(args.fig_dir, f'fig_checkpoint_norms.{ext}'),
                    bbox_inches='tight', facecolor='white')
    print('Saved fig_checkpoint_norms')


if __name__ == '__main__':
    main()
