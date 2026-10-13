"""Schematic of the pipeline (a block diagram for the paper's method section).

Usage: uv run python src/plot_pipeline.py
Output: results/figures/paper_pipeline.png
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(ROOT, 'results', 'figures')

BOXES = {
    'real': (0.02, 0.62, 0.20, 0.24, 'Real dataset\n(9 snapshots)\n25,714 users | 824,045 edges\n173,475 posts'),
    'content': (0.27, 0.62, 0.20, 0.24, 'Content representation\none vector per post\n(MiniLM), mean per user'),
    'tnode': (0.52, 0.62, 0.20, 0.24, 'tNodeEmbed\nno node2vec\nProcrustes + LSTM\npair head [hu, hv, hu*hv]'),
    'fit': (0.27, 0.30, 0.20, 0.22, 'Fit on the real\n(transitions 1-5)\nvalidation 6, test untouched'),
    'v8': (0.52, 0.42, 0.20, 0.16, 'V8: pairwise model\nlogistic + sampling\nwith feedback'),
    'v9': (0.52, 0.22, 0.20, 0.16, 'V9: arrival events\nk per newcomer, target\nchoice, reciprocity'),
    'scenario': (0.02, 0.30, 0.20, 0.22, 'Synthetic scenario\n100,456 posts -> 9 steps\narrival curve + activity\n(measured on the real)'),
    'network': (0.77, 0.42, 0.21, 0.24, 'Generated synthetic\nnetwork\n(synthetic to synthetic)'),
    'eval': (0.77, 0.12, 0.21, 0.22, 'Evaluation\nstructure, evolution,\ncontent, controls\n(real vs generated)'),
}
ARROWS = [
    ('real', 'content'), ('content', 'tnode'), ('tnode', 'fit'), ('real', 'fit'),
    ('fit', 'v8'), ('fit', 'v9'), ('scenario', 'v9'), ('scenario', 'v8'),
    ('v8', 'network'), ('v9', 'network'), ('network', 'eval'), ('real', 'eval'),
]


def main():
    os.makedirs(FIGURES, exist_ok=True)
    figure, ax = plt.subplots(figsize=(13, 6.2))
    for key, (x, y, width, height, text) in BOXES.items():
        colour = {'v9': '#d8e8f8', 'v8': '#f2f2f2', 'network': '#dff0d8',
                  'eval': '#fdf2d0'}.get(key, '#ffffff')
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.012',
                                    linewidth=1.2, edgecolor='#333333', facecolor=colour))
        ax.text(x + width / 2, y + height / 2, text, ha='center', va='center', fontsize=8.5,
                linespacing=1.5)
    for start, end in ARROWS:
        x1, y1, w1, h1, _ = BOXES[start]
        x2, y2, w2, h2, _ = BOXES[end]
        arrow = FancyArrowPatch((x1 + w1 / 2, y1 + (h1 if y2 > y1 else 0)),
                                (x2 + w2 / 2, y2 + (0 if y2 > y1 else h2)),
                                arrowstyle='-|>', mutation_scale=12, linewidth=1.1,
                                color='#555555', connectionstyle='arc3,rad=0.0')
        ax.add_patch(arrow)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.92)
    ax.axis('off')
    ax.set_title('From the real network to the synthetic one: training, generation, evaluation',
                 fontsize=11)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, 'paper_pipeline.png'), dpi=200)
    plt.close(figure)
    print(f'written {os.path.join(FIGURES, "paper_pipeline.png")}')


if __name__ == '__main__':
    main()
