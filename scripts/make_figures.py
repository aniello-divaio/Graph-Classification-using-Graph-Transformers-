"""
make_figures.py
Draws the thesis figures from the files written by evaluate_all.py.
No result is hard-coded: re-running evaluate_all.py and then this script
regenerates every figure.

Colours come from the viridis colormap (the same used for the confusion
matrices) and encode the information carried by the nodes, identically in
every figure: reaction ID -> purple, node degree -> teal, constant -> green.
Hatching repeats the same mapping for black-and-white printing.

Outputs (in --fig_dir), each as .pdf (for LaTeX) and .png:
    fig_overall_comparison     best classifier of each configuration
    fig_graphormer_classifiers Graphormer: 4 classifiers x 3 node features
    fig_per_class_f1           F1 per class
    fig_confusion_best         confusion matrices of the best WL and Graphormer
    annex_confusion_<name>     one confusion matrix per configuration

Usage:
    python scripts/make_figures.py --results results/final --fig_dir figures
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PURPLE, TEAL, GREEN = '#472d7b', '#25858e', '#6ece58'  # viridis 0.12 / 0.45 / 0.75
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
STYLE = {'reaction': (PURPLE, ''), 'degree': (TEAL, '///'),
         'constant': (GREEN, '\\\\\\')}
LABEL = {'reaction': 'Reaction ID', 'degree': 'Node degree',
         'constant': 'Constant'}
CLASSES = ['Actino', 'Alpha', 'Bacilli', 'Clostridia', 'Bacteroid', 'Beta',
           'Epsilon', 'Entero', 'Gamma', 'Other', 'Teneri']
LONG = {'Actino': 'Actinomycetota', 'Alpha': 'Alphaproteobacteria',
        'Bacilli': 'Bacillota – Bacilli', 'Clostridia': 'Bacillota – Clostridia',
        'Bacteroid': 'Bacteroidota', 'Beta': 'Betaproteobacteria',
        'Epsilon': 'Epsilonproteobacteria',
        'Entero': 'Gammaproteob. – Enterobacteria',
        'Gamma': 'Gammaproteob. – Others', 'Other': 'Other',
        'Teneri': 'Tenericutes'}
CLF_ORDER = ['Logistic Regression', 'Linear SVM', 'Random Forest', 'MLP']

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK, 'axes.linewidth': 0.6, 'xtick.color': INK2,
    'ytick.color': INK, 'hatch.linewidth': 0.6, 'hatch.color': 'white',
    'savefig.dpi': 300,
})


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def style(ax, grid_axis):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, fig_dir, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(fig_dir, f'{name}.{ext}'),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved', name)


def best(rows, method, features):
    cand = [r for r in rows if r['method'] == method
            and r['node_features'] == features]
    return max(cand, key=lambda r: float(r['accuracy_mean']))


def baselines(results_dir):
    """Random and majority-class baselines from a confusion matrix."""
    cm = np.load(os.path.join(results_dir, 'confusion',
                              os.listdir(os.path.join(results_dir, 'confusion'))[0]))
    support = cm.sum(axis=1)
    return 100 / len(support), 100 * support.max() / support.sum()


# ------------------------------------------------------------------ figures
def fig_overall(rows, rnd, maj, fig_dir):
    order = [('WL kernel', 'reaction'), ('WL kernel', 'degree'),
             ('Graphormer (frozen)', 'constant'),
             ('Graphormer (frozen)', 'degree'),
             ('Graphormer (frozen)', 'reaction')]
    order = [(m, f) for m, f in order
             if any(r['method'] == m and r['node_features'] == f for r in rows)]
    wl = [o for o in order if o[0] == 'WL kernel']
    gr = [o for o in order if o[0] != 'WL kernel']
    ys = [len(gr) + 1.0 + i for i in range(len(wl))][::-1] + \
         [i + 0.6 for i in range(len(gr))][::-1]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ticks = []
    for y, (m, f) in zip(ys, wl + gr):
        r = best(rows, m, f)
        mean, sd = float(r['accuracy_mean']), float(r['accuracy_std'])
        c, h = STYLE[f]
        ax.barh(y, mean, height=0.72, color=c, hatch=h, linewidth=0)
        ax.errorbar(mean, y, xerr=sd, color=INK, capsize=2.5, lw=0.8)
        ax.text(mean + sd + 1.2, y, f'{mean:.1f}%', va='center', fontsize=9)
        ticks.append(LABEL[f] if m == 'WL kernel'
                     else f'{LABEL[f]} ({r["classifier"]})')
    ax.set_yticks(ys, ticks)
    # place the method names left of the widest tick label (measured)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widest = max(t.get_window_extent(renderer).width
                 for t in ax.get_yticklabels())
    shift = -(widest * 72 / fig.dpi + 14)  # pixels -> points, plus padding
    groups = []
    if wl:
        groups.append((np.mean(ys[:len(wl)]), 'WL kernel\n+ SVM'))
    if gr:
        groups.append((np.mean(ys[len(wl):]), 'Graphormer\n(frozen)'))
    for yc, lab in groups:
        ax.annotate(lab, xy=(0, yc), xycoords=('axes fraction', 'data'),
                    xytext=(shift, 0), textcoords='offset points', ha='right',
                    va='center', fontweight='bold')
    top = max(ys) + 0.6
    lines = ((rnd, f'random {rnd:.1f}%', top + 0.45),
             (maj, f'majority class {maj:.1f}%', top))
    if abs(maj - rnd) < 2.5:  # baselines coincide: one line, one label
        lines = ((rnd, f'random = majority class {rnd:.1f}%', top),)
    for x, lab, yl in lines:
        ax.axvline(x, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.text(x + 0.8, yl, lab, color=INK2, fontsize=7.5, va='bottom')
    xmax = max(float(best(rows, m, f)['accuracy_mean']) +
               float(best(rows, m, f)['accuracy_std']) for m, f in wl + gr)
    ax.set_xlim(0, max(85, xmax + 12))
    ax.set_ylim(0, top + 0.9)
    ax.set_xlabel('Accuracy (%), mean ± std over 5 folds')
    style(ax, 'x')
    ax.tick_params(axis='y', length=0)
    save(fig, fig_dir, 'fig_overall_comparison')


def fig_classifiers(rows, rnd, maj, fig_dir):
    feats = [f for f in ('constant', 'degree', 'reaction')
             if any(r['method'] == 'Graphormer (frozen)'
                    and r['node_features'] == f for r in rows)]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    w, gap = 0.25, 0.02
    for k, f in enumerate(feats):
        c, h = STYLE[f]
        sub = {r['classifier']: r for r in rows
               if r['method'] == 'Graphormer (frozen)' and r['node_features'] == f}
        xs = [i + (k - (len(feats) - 1) / 2) * (w + gap)
              for i in range(len(CLF_ORDER))]
        means = [float(sub[cl]['accuracy_mean']) for cl in CLF_ORDER]
        sds = [float(sub[cl]['accuracy_std']) for cl in CLF_ORDER]
        ax.bar(xs, means, width=w, color=c, hatch=h, linewidth=0,
               label=LABEL[f])
        ax.errorbar(xs, means, yerr=sds, fmt='none', ecolor=INK, capsize=2,
                    lw=0.7)
        for x, v, s in zip(xs, means, sds):
            ax.text(x, v + s + 0.8, f'{v:.1f}', ha='center', fontsize=7)
    lines = ((rnd, 'random'), (maj, 'majority class'))
    if abs(maj - rnd) < 2.5:
        lines = ((rnd, 'random = majority'),)
    for yv, lab in lines:
        ax.axhline(yv, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.text(3.5, yv, lab, color=INK2, fontsize=7.5, va='center',
                clip_on=False)
    ax.set_xlim(-0.5, 3.45)
    ax.set_xticks(range(len(CLF_ORDER)),
                  ['Logistic\nRegression', 'Linear SVM', 'Random Forest', 'MLP'])
    ax.set_ylim(0, 60)
    ax.set_ylabel('Accuracy (%)')
    ax.legend(title='Node features', frameon=False, ncol=3, fontsize=8,
              title_fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.16))
    style(ax, 'y')
    ax.tick_params(axis='x', length=0)
    save(fig, fig_dir, 'fig_graphormer_classifiers')


def fig_per_class(rows, f1rows, fig_dir):
    f1 = {r['name']: {c: float(r[c]) for c in CLASSES} for r in f1rows}
    series = [  # (row name, colour, marker, filled, legend)
        (best(rows, 'WL kernel', 'reaction')['name'], PURPLE, 'o', True,
         'WL kernel – reaction ID'),
        (best(rows, 'WL kernel', 'degree')['name'], TEAL, 'o', True,
         'WL kernel – node degree'),
    ]
    g = best(rows, 'Graphormer (frozen)', 'reaction')
    series.append((g['name'], PURPLE, 's', False,
                   f'Graphormer (frozen) + {g["classifier"]} – reaction ID'))
    order = sorted(CLASSES, key=lambda c: f1[series[0][0]][c])
    offsets = {0: 0.14, 1: -0.14, 2: 0.0}
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    for i, c in enumerate(order):
        vals = [f1[s[0]][c] for s in series]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=1.2, zorder=1)
        for k, (name, col, mk, filled, _) in enumerate(series):
            if filled:
                ax.scatter(f1[name][c], i + offsets[k], s=46, marker=mk,
                           color=col, edgecolor='white', linewidth=1.2,
                           zorder=4 - k)
            else:
                ax.scatter(f1[name][c], i + offsets[k], s=46, marker=mk,
                           facecolor='white', edgecolor=col, linewidth=1.6,
                           zorder=3)
    ax.set_yticks(range(len(order)), [LONG[c] for c in order])
    ax.set_xlim(0, 1)
    ax.set_xlabel('F1-score (out-of-fold predictions, 5-fold CV)')
    handles = [Line2D([], [], marker=mk, ls='', markersize=7,
                      color=col if filled else 'white',
                      markerfacecolor=col if filled else 'white',
                      markeredgecolor=col, markeredgewidth=1.6, label=lab)
               for _, col, mk, filled, lab in series]
    ax.legend(handles=handles, frameon=False, ncol=2, fontsize=8,
              loc='upper center', bbox_to_anchor=(0.35, 1.13))
    style(ax, 'x')
    ax.tick_params(axis='y', length=0)
    save(fig, fig_dir, 'fig_per_class_f1')


def draw_cm(ax, cm, title):
    pct = 100 * cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap='viridis', vmin=0, vmax=100)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, f'{pct[i, j]:.0f}', ha='center', va='center',
                        fontsize=6, color=INK if pct[i, j] > 60 else 'white')
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha='right',
                  fontsize=7)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
    ax.set_xlabel('Predicted class', fontsize=8)
    ax.set_title(title, fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def fig_confusion(rows, results_dir, fig_dir):
    cm_path = lambda n: os.path.join(results_dir, 'confusion', f'{n}.npy')
    wl = max((r for r in rows if r['method'] == 'WL kernel'),
             key=lambda r: float(r['accuracy_mean']))
    gr = max((r for r in rows if r['method'] == 'Graphormer (frozen)'),
             key=lambda r: float(r['accuracy_mean']))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4))
    for ax, r, head in ((axes[0], wl, 'WL kernel + SVM'),
                        (axes[1], gr, f'Graphormer (frozen) + {gr["classifier"]}')):
        draw_cm(ax, np.load(cm_path(r['name'])),
                f'{head}\n{LABEL[r["node_features"]].lower()} – '
                f'{float(r["accuracy_mean"]):.1f}%')
    axes[0].set_ylabel('True class', fontsize=8)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    save(fig, fig_dir, 'fig_confusion_best')
    for r in rows:  # annex: one matrix per configuration (best classifier)
        if r is not best(rows, r['method'], r['node_features']):
            continue
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        draw_cm(ax, np.load(cm_path(r['name'])),
                f'{r["method"]} + {r["classifier"]}\n'
                f'{LABEL[r["node_features"]].lower()} – '
                f'{float(r["accuracy_mean"]):.1f}%')
        ax.set_ylabel('True class', fontsize=8)
        save(fig, fig_dir, f'annex_confusion_{r["name"]}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/final')
    p.add_argument('--fig_dir', default='figures')
    args = p.parse_args()
    res = os.path.expanduser(args.results)
    fig_dir = os.path.expanduser(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    rows = read_csv(os.path.join(res, 'all_results.csv'))
    f1rows = read_csv(os.path.join(res, 'per_class_f1.csv'))
    rnd, maj = baselines(res)
    fig_overall(rows, rnd, maj, fig_dir)
    if any(r['method'] == 'Graphormer (frozen)' for r in rows):
        fig_classifiers(rows, rnd, maj, fig_dir)
    if any(r['method'] == 'WL kernel' for r in rows):
        fig_per_class(rows, f1rows, fig_dir)
    fig_confusion(rows, res, fig_dir)


if __name__ == '__main__':
    main()
