"""Temporal analysis of a dataset folder (real or generated), plus its figures.

Per consecutive snapshot pair: new nodes and edges, edge persistence (Jaccard between the
edge sets), community stability (NMI between the Louvain partitions), and the densification
exponent (E ~ N^alpha fitted on log E vs log N). Also tracks the most connected nodes over
time.

Usage: uv run python -m src.analysis.analyze_temporal <folder_name>
Output: results/temporal_<folder>.csv, results/figures/temporal_<folder>_*.png
"""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from src.paths import ROOT, BASE, TABLES, FIGURES

from src.lib.graph_io import load_network

SEED = 0


def communities_of(edges):
    graph = nx.Graph()
    graph.add_edges_from(edges)
    assignment = {}
    for index, community in enumerate(nx.community.louvain_communities(graph, seed=SEED)):
        for node in community:
            assignment[node] = index
    return assignment


def main():
    folder_name = sys.argv[1] if len(sys.argv) > 1 else 'real_snapshots'
    folder = os.path.join(BASE, folder_name)
    os.makedirs(FIGURES, exist_ok=True)
    snapshots = load_network(folder)

    assignments = [communities_of(edges) for _, edges, _ in snapshots]
    rows = []
    for index in range(1, len(snapshots)):
        name, edges, nodes = snapshots[index]
        previous_name, previous_edges, previous_nodes = snapshots[index - 1]
        new_edges = edges - previous_edges
        removed_edges = previous_edges - edges
        new_nodes = len(nodes - previous_nodes)
        union = edges | previous_edges
        common = [n for n in nodes if n in previous_nodes and n in assignments[index - 1]]
        nmi = normalized_mutual_info_score(
            [assignments[index - 1][n] for n in common],
            [assignments[index][n] for n in common]) if common else float('nan')
        rows.append({
            'snapshot': name,
            'nodes': len(nodes),
            'new_nodes': new_nodes,
            'edges': len(edges),
            'new_edges': len(new_edges),
            'removed_edges': len(removed_edges),
            'edge_jaccard': len(edges & previous_edges) / len(union) if union else float('nan'),
            'community_nmi': nmi,
        })
        print(f'{name:>20}: +{len(new_edges):>7} edges ({new_nodes:>5} new nodes) '
              f'| removed {len(removed_edges):>6} | jaccard {rows[-1]["edge_jaccard"]:.3f} '
              f'| community NMI {nmi:.3f}')

    with open(os.path.join(TABLES, f'temporal_{folder_name}.csv'), 'w', encoding='utf-8',
              newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    node_counts = np.array([len(nodes) for _, _, nodes in snapshots], dtype=float)
    edge_counts = np.array([len(edges) for _, edges, _ in snapshots], dtype=float)
    slope = np.polyfit(np.log(node_counts), np.log(edge_counts), 1)[0]

    labels = [name for name, _, _ in snapshots]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(labels[1:], [row['new_edges'] for row in rows], marker='o')
    axes[0].set_title('new edges per snapshot')
    axes[1].plot(labels[1:], [row['edge_jaccard'] for row in rows], marker='o', label='edge Jaccard')
    axes[1].plot(labels[1:], [row['community_nmi'] for row in rows], marker='s', label='community NMI')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('persistence and stability')
    axes[1].legend()
    axes[2].loglog(node_counts, edge_counts, marker='o')
    axes[2].plot(node_counts, np.exp(np.polyval(np.polyfit(np.log(node_counts), np.log(edge_counts), 1),
                                                 np.log(node_counts))), linestyle='--',
                 label=f'slope {slope:.2f}')
    axes[2].set_xlabel('nodes')
    axes[2].set_ylabel('edges')
    axes[2].set_title('densification (E ~ N^alpha)')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    figure.suptitle(folder_name)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, f'temporal_{folder_name}_summary.png'))
    plt.close(figure)
    print(f'densification slope alpha = {slope:.3f} | written results/tables/temporal_{folder_name}.csv')


if __name__ == '__main__':
    main()
