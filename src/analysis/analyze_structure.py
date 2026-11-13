"""Structural analysis of a dataset folder (real or generated), plus its figures.

Light metrics for every snapshot; heavy metrics (transitivity, clustering, diameter) only
for the last snapshot, where the graph is largest.

Usage: uv run python -m src.analysis.analyze_structure <folder_name>
Output: results/structure_<folder>.csv, results/structure_heavy_<folder>.csv,
        results/figures/structure_<folder>_*.png
"""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.paths import ROOT, BASE, TABLES, FIGURES

from src.lib.graph_io import load_network

SEED = 0


def reciprocity(edges):
    if not edges:
        return float('nan')
    return sum(1 for a, b in edges if (b, a) in edges) / len(edges)


def light_metrics(name, edges, nodes, previous_nodes):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    out_degree = np.array([d for _, d in graph.out_degree()], dtype=float)
    in_degree = np.array([d for _, d in graph.in_degree()], dtype=float)
    weak = list(nx.weakly_connected_components(graph))
    strong = list(nx.strongly_connected_components(graph))
    communities = nx.community.louvain_communities(graph.to_undirected(), seed=SEED)
    return {
        'snapshot': name,
        'nodes': len(nodes),
        'new_nodes': len(nodes - previous_nodes),
        'edges': len(edges),
        'density': len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else float('nan'),
        'out_degree_mean': out_degree.mean() if len(out_degree) else float('nan'),
        'out_degree_median': float(np.median(out_degree)) if len(out_degree) else float('nan'),
        'out_degree_max': int(out_degree.max()) if len(out_degree) else 0,
        'in_degree_mean': in_degree.mean() if len(in_degree) else float('nan'),
        'reciprocity': reciprocity(edges),
        'weak_components': len(weak),
        'largest_weak_share': max(len(c) for c in weak) / len(nodes) if nodes else float('nan'),
        'strong_components': len(strong),
        'max_k_core': max(nx.core_number(graph.to_undirected()).values()) if nodes else 0,
        'communities': len(communities),
        'modularity': nx.community.modularity(graph.to_undirected(), communities) if nodes else float('nan'),
    }


def heavy_metrics(name, edges, nodes):
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    undirected = graph.to_undirected()
    largest = max(nx.connected_components(undirected), key=len) if nodes else set()
    subgraph = undirected.subgraph(largest)
    row = {
        'snapshot': name,
        'transitivity': nx.transitivity(graph),
        'avg_clustering': nx.average_clustering(undirected),
        'largest_component_nodes': len(largest),
        'diameter_approx': nx.approximation.diameter(subgraph, seed=SEED) if len(largest) > 1 else 0,
        'avg_path_length_approx': nx.average_shortest_path_length(subgraph) if 1 < len(largest) <= 4000
        else float('nan'),
    }
    return row


def main():
    folder_name = sys.argv[1] if len(sys.argv) > 1 else 'real_snapshots'
    folder = os.path.join(BASE, folder_name)
    os.makedirs(FIGURES, exist_ok=True)
    snapshots = load_network(folder)

    rows, previous_nodes = [], set()
    for name, edges, nodes in snapshots:
        row = light_metrics(name, edges, nodes, previous_nodes)
        row['new_edges'] = len(edges) - (rows[-1]['edges'] if rows else 0)
        rows.append(row)
        previous_nodes = nodes
        print(f'{name:>20}: nodes {row["nodes"]:>6} edges {row["edges"]:>7} '
              f'density {row["density"]:.5f} reciprocity {row["reciprocity"]:.3f} '
              f'communities {row["communities"]:>3} modularity {row["modularity"]:.3f}')

    with open(os.path.join(TABLES, f'structure_{folder_name}.csv'), 'w', encoding='utf-8',
              newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    name, edges, nodes = snapshots[-1]
    heavy = heavy_metrics(name, edges, nodes)
    print(f'heavy metrics on {name}: {heavy}')
    with open(os.path.join(TABLES, f'structure_heavy_{folder_name}.csv'), 'w', encoding='utf-8',
              newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(heavy))
        writer.writeheader()
        writer.writerow(heavy)

    labels = [row['snapshot'] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(labels, [row['nodes'] for row in rows], marker='o')
    axes[0].set_title('nodes per snapshot')
    axes[1].plot(labels, [row['edges'] for row in rows], marker='o', color='tab:orange')
    axes[1].set_title('edges per snapshot')
    axes[2].plot(labels, [row['modularity'] for row in rows], marker='o', label='modularity')
    axes[2].plot(labels, [row['reciprocity'] for row in rows], marker='s', label='reciprocity')
    axes[2].set_title('modularity and reciprocity')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    figure.suptitle(folder_name)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, f'structure_{folder_name}_summary.png'))
    plt.close(figure)

    degrees = {}
    for a, _ in edges:
        degrees[a] = degrees.get(a, 0) + 1
    values = np.sort(np.array(list(degrees.values())))
    ccdf = 1 - np.arange(len(values)) / len(values)
    figure, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(values, ccdf)
    ax.set_xlabel('out-degree')
    ax.set_ylabel('P(out-degree > x)')
    ax.set_title(f'out-degree distribution (final snapshot) - {folder_name}')
    ax.grid(alpha=0.3, which='both')
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, f'structure_{folder_name}_degree_ccdf.png'))
    plt.close(figure)
    print(f'written results/tables/structure_{folder_name}.csv and two figures')


if __name__ == '__main__':
    main()
