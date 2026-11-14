"""The two temporal partitions of the synthetic dataset, drawn from users_by_birth.csv.

Left panel: share of the population present at each step. The all-present partition is flat at
100% from the first step; the progressive one follows the arrival curve measured on the real
network. Right panel: how many users are born at each step, which is the curve that generates
the cumulative one.

Usage: uv run python -m src.figures.plot_partitions
Output: results/figures/partitions_timeline.png
"""

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.paths import ROOT, BASE

FIGURES = os.path.join(ROOT, 'results', 'figures')
BLUE = '#1f4e79'
RED = '#c0392b'


def births(folder):
    path = os.path.join(BASE, folder, 'users_by_birth.csv')
    counts = {}
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            step = int(float(row['birth_snapshot']))
            counts[step] = counts.get(step, 0) + 1
    return counts


def main():
    progressive = births('synthetic_progressive_posts')
    steps = sorted(progressive)
    born = np.array([progressive[step] for step in steps], dtype=float)
    cumulative = np.cumsum(born) / born.sum()
    all_present = np.ones_like(born)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axis = axes[0]
    axis.plot(steps, 100 * all_present, marker='s', color='#7f8c8d', linewidth=2,
              label='all-present')
    axis.plot(steps, 100 * cumulative, marker='o', color=RED, linewidth=2,
              label='progressive')
    axis.set_title('share of the population present')
    axis.set_xlabel('step')
    axis.set_ylabel('percent of the 10,000 users')
    axis.set_ylim(0, 105)
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)

    axis = axes[1]
    axis.bar(steps, born, color=BLUE, alpha=0.85)
    axis.set_title('users entering at each step (progressive)')
    axis.set_xlabel('step')
    axis.set_ylabel('new users')
    axis.grid(alpha=0.3, axis='y')

    figure.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, 'partitions_timeline.png')
    figure.savefig(path, dpi=200, bbox_inches='tight')
    print('written', path)
    print('cumulative shares:', ', '.join(f'{100 * value:.1f}' for value in cumulative))
    print('births:', ', '.join(str(int(value)) for value in born))


if __name__ == '__main__':
    main()
