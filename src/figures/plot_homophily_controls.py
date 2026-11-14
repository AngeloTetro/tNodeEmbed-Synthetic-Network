"""Content homophily of the real network under structural controls (bar chart).

Every bar is the ratio between the similarity of linked pairs and the similarity of a control
set of unlinked pairs with the same structure. A ratio of 1.0 means no homophily: the standard
measure (random pairs) is the weakest control, and the ratio falls towards 1 as the control gets
structurally closer.

Usage: uv run python -m src.figures.plot_homophily_controls
Output: results/figures/homophily_controls.png
"""

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.paths import ROOT, TABLES, FIGURES

BLUE = '#1f4e79'
RED = '#c0392b'

LABELS = {
    'casuale': 'random pairs (standard measure)',
    'grado': 'non-edges, same degree class',
    'grado stretto': 'non-edges, degree within 10%',
    'arrivo+grado': 'non-edges, same arrival step and degree',
    'testo': 'non-edges, same text volume',
    'grado+testo': 'non-edges, similar degree and text volume',
}
ORDER = ['casuale', 'grado', 'arrivo+grado', 'testo', 'grado+testo', 'grado stretto']


def main():
    with open(os.path.join(TABLES, 'homophily_control_real_snapshots.csv'),
              encoding='utf-8', newline='') as handle:
        rows = {row['control']: row for row in csv.DictReader(handle)}

    names = [LABELS[key] for key in ORDER]
    ratios = [float(rows[key]['ratio']) for key in ORDER]
    aucs = [float(rows[key]['auc_edges_vs_control']) for key in ORDER]

    figure, axis = plt.subplots(figsize=(8.2, 4.2))
    positions = range(len(names))
    colours = [RED if key == 'casuale' else BLUE for key in ORDER]
    axis.barh(list(positions), ratios, color=colours, alpha=0.85)
    axis.set_yticks(list(positions))
    axis.set_yticklabels(names, fontsize=9)
    axis.invert_yaxis()
    axis.axvline(1.0, color='#555555', linestyle='--', linewidth=1)
    axis.set_xlim(1.0, 2.5)
    axis.set_xlabel('similarity of linked pairs / similarity of the control set')
    for position, ratio, auc in zip(positions, ratios, aucs):
        axis.text(ratio + 0.02, position, f'{ratio:.2f}  (AUC {auc:.2f})',
                  va='center', fontsize=8.5)
    axis.set_title('Content homophily of the real network, under six controls', fontsize=10)
    axis.grid(alpha=0.25, axis='x')

    figure.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, 'homophily_controls.png')
    figure.savefig(path, dpi=200, bbox_inches='tight')
    print('written', path)


if __name__ == '__main__':
    main()
