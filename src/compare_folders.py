"""Side-by-side comparison of the datasets (real and generated), from the results CSVs.

Produces figures under results/figures:
  compare_growth.png        nodes and edges per snapshot
  compare_structure.png     density, reciprocity, modularity per snapshot
  compare_homophily.png     content homophily ratio per snapshot
  compare_degree_ccdf.png   out-degree distribution of the final snapshot
  compare_profiles.png      profile classification AUC by feature set (generated folders)

Usage: uv run python src/compare_folders.py [--prefix NAME] [folder ...]
       (default folders: real_snapshots, synthetic_progressive,
        synthetic_all_present; with --prefix the figures are named
        <NAME>_growth.png etc., so they do not overwrite the ones used by the internal report)
"""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from graph_io import load_edges, snapshots_folder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(RESULTS, 'figures')
DEFAULT_FOLDERS = ('real_snapshots', 'synthetic_progressive',
                   'synthetic_all_present')
LABELS = {'real_snapshots': 'real',
          'synthetic_progressive': 'progressive',
          'synthetic_all_present': 'all-present'}
ARGUMENTS = list(sys.argv[1:])
PREFIX = 'compare'
if ARGUMENTS and ARGUMENTS[0] == '--prefix':
    PREFIX = ARGUMENTS[1]
    ARGUMENTS = ARGUMENTS[2:]
FOLDERS = tuple(ARGUMENTS) if ARGUMENTS else DEFAULT_FOLDERS


def read_csv(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def numeric(rows, column):
    return [float(row[column]) if row[column] not in ('', 'nan') else float('nan') for row in rows]


def main():
    os.makedirs(FIGURES, exist_ok=True)
    structure = {folder: read_csv(f'structure_{folder}.csv') for folder in FOLDERS}
    content = {folder: read_csv(f'content_{folder}.csv') for folder in FOLDERS}

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for folder, rows in structure.items():
        if not rows:
            continue
        axis = np.arange(1, len(rows) + 1)
        axes[0].plot(axis, numeric(rows, 'nodes'), marker='o', label=LABELS.get(folder, folder))
        axes[1].plot(axis, numeric(rows, 'edges'), marker='o', label=LABELS.get(folder, folder))
    axes[0].set_title('nodes per snapshot')
    axes[1].set_title('edges per snapshot')
    for ax in axes:
        ax.set_xlabel('snapshot index')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, f'{PREFIX}_growth.png'))
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for folder, rows in structure.items():
        if not rows:
            continue
        axis = np.arange(1, len(rows) + 1)
        axes[0].plot(axis, numeric(rows, 'density'), marker='o', label=LABELS.get(folder, folder))
        axes[1].plot(axis, numeric(rows, 'reciprocity'), marker='o', label=LABELS.get(folder, folder))
        axes[2].plot(axis, numeric(rows, 'modularity'), marker='o', label=LABELS.get(folder, folder))
    for ax, title in zip(axes, ('density', 'reciprocity', 'modularity')):
        ax.set_title(title)
        ax.set_xlabel('snapshot index')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, f'{PREFIX}_structure.png'))
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(7, 4.5))
    for folder, rows in content.items():
        if not rows:
            continue
        ax.plot(np.arange(1, len(rows) + 1), numeric(rows, 'homophily_ratio'), marker='o',
                label=LABELS.get(folder, folder))
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('snapshot index')
    ax.set_ylabel('edge similarity / random pairs')
    ax.set_title('content homophily ratio')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, f'{PREFIX}_homophily.png'))
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(7, 4.5))
    for folder in FOLDERS:
        if not os.path.isdir(os.path.join(BASE, folder)):
            continue
        directories, names = snapshots_folder(os.path.join(BASE, folder))
        edges = load_edges(os.path.join(directories[-1], 'social_network.edg'))
        if not edges:
            continue
        degrees = {}
        for source, _ in edges:
            degrees[source] = degrees.get(source, 0) + 1
        values = np.sort(np.array(list(degrees.values())))
        ax.loglog(values, 1 - np.arange(len(values)) / len(values), label=LABELS.get(folder, folder))
    ax.set_xlabel('out-degree')
    ax.set_ylabel('P(out-degree > x)')
    ax.set_title('out-degree distribution, final snapshot')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, f'{PREFIX}_degree_ccdf.png'))
    plt.close(figure)

    generated = [folder for folder in FOLDERS if folder.startswith('generated')]
    rows = [(folder, read_csv(f'profiles_{folder}.csv')) for folder in generated]
    rows = [(folder, r) for folder, r in rows if r]
    if rows:
        attributes = sorted({row['attribute'] for _, r in rows for row in r})
        feature_sets = sorted({row['features'] for _, r in rows for row in r})
        figure, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.5), squeeze=False)
        for ax, (folder, r) in zip(axes[0], rows):
            width = 0.8 / len(feature_sets)
            for offset, feature in enumerate(feature_sets):
                values = [next((float(row['auc']) for row in r
                                if row['attribute'] == attribute and row['features'] == feature),
                               float('nan')) for attribute in attributes]
                ax.bar(np.arange(len(attributes)) + offset * width, values, width, label=feature)
            ax.set_xticks(np.arange(len(attributes)) + 0.4 - width / 2)
            ax.set_xticklabels(attributes, rotation=20, ha='right')
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
            ax.set_ylabel('AUC (one-vs-rest macro)')
            ax.set_title(LABELS.get(folder, folder))
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3, axis='y')
        figure.tight_layout()
        figure.savefig(os.path.join(FIGURES, f'{PREFIX}_profiles.png'))
        plt.close(figure)
    print(f'comparison figures written to {FIGURES}')


if __name__ == '__main__':
    main()
