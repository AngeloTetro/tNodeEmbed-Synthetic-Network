"""Content analysis of a dataset folder (real or generated), plus its figures.

Measures how much the content explains the edges:
  - mean cosine similarity between the content of the two endpoints of an edge, against the
    similarity of random pairs (a homophily ratio);
  - similarity inside vs across Louvain communities;
  - when the folder has user profiles (synthetic only), how often an edge joins users with
    the same attribute value, compared with the random expectation (a lift).

Usage: uv run python -m src.analysis.analyze_content <folder_name>
Output: results/content_<folder>.csv, results/figures/content_<folder>_*.png
"""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from src.paths import ROOT, BASE, TABLES, FIGURES

from src.lib.graph_io import cumulative_docs, load_attributes, snapshots_folder

ATTRIBUTES = ('political_leaning', 'profession', 'interests', 'state_of_origin', 'age_interval',
              'gender', 'ethnicity', 'religion', 'nationality')
SEED = 0
rng = np.random.default_rng(SEED)


def main():
    folder_name = sys.argv[1] if len(sys.argv) > 1 else 'real_snapshots'
    folder = os.path.join(BASE, folder_name)
    os.makedirs(FIGURES, exist_ok=True)
    directories, names = snapshots_folder(folder)
    snapshots = [(name, directory) for name, directory in zip(names, directories)]

    rows, community_rows = [], []
    attributes = load_attributes(os.path.join(directories[-1], 'posts_incremental.tsv'), ATTRIBUTES)
    doc_names, doc_sets = cumulative_docs(folder)
    for position, (name, directory) in enumerate(snapshots):
        edges = set()
        edge_path = os.path.join(directory, 'social_network.edg')
        if os.path.exists(edge_path):
            with open(edge_path, encoding='utf-8-sig', newline='') as f:
                for line in f:
                    parts = line.rstrip('\r\n').split('\t')
                    if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                        edges.add((parts[0].strip(), parts[1].strip()))
        docs = doc_sets[position]
        if not edges or not docs:
            continue
        users = sorted(docs)
        index = {u: i for i, u in enumerate(users)}
        vectorizer = TfidfVectorizer(min_df=5, sublinear_tf=True, lowercase=True,
                                     strip_accents='unicode', max_features=200000)
        matrix = normalize(TruncatedSVD(n_components=128, random_state=SEED).fit_transform(
            vectorizer.fit_transform([' '.join(docs[u]) for u in users])))
        pairs = [(a, b) for a, b in edges if a in index and b in index]
        random_pairs = [(users[i], users[j]) for i, j in
                        zip(rng.integers(0, len(users), len(pairs)), rng.integers(0, len(users), len(pairs)))]

        def similarity(items):
            i = np.array([index[a] for a, b in items])
            j = np.array([index[b] for a, b in items])
            return np.einsum('ij,ij->i', matrix[i], matrix[j])

        edge_similarity = float(similarity(pairs).mean())
        random_similarity = float(similarity(random_pairs).mean())
        graph = nx.Graph()
        graph.add_edges_from(pairs)
        communities = nx.community.louvain_communities(graph, seed=SEED)
        assignment = {node: c for c, community in enumerate(communities) for node in community}
        inside = [(a, b) for a, b in pairs if assignment.get(a) == assignment.get(b)]
        across = [(a, b) for a, b in pairs if assignment.get(a) != assignment.get(b)]
        rows.append({
            'snapshot': name,
            'edges_used': len(pairs),
            'edge_content_similarity': edge_similarity,
            'random_content_similarity': random_similarity,
            'homophily_ratio': edge_similarity / random_similarity if random_similarity else float('nan'),
            'similarity_inside_communities': float(similarity(inside).mean()) if inside else float('nan'),
            'similarity_across_communities': float(similarity(across).mean()) if across else float('nan'),
        })
        print(f'{name:>20}: content similarity edges {edge_similarity:.4f} vs random '
              f'{random_similarity:.4f} (ratio {rows[-1]["homophily_ratio"]:.2f})')

        for attribute in ATTRIBUTES:
            table = attributes[attribute]
            known = [(a, b) for a, b in pairs if a in table and b in table]
            if len(known) < 100:
                continue
            same = sum(1 for a, b in known if table[a] == table[b]) / len(known)
            pool = sorted(table)
            baseline = [(a, b) for a, b in zip(rng.choice(pool, len(known), replace=True),
                                               rng.choice(pool, len(known), replace=True)) if a != b]
            expected = sum(1 for a, b in baseline if table[a] == table[b]) / len(baseline)
            covered = [u for u in users if u in table]
            labels_attribute = np.array([table[u] for u in covered])
            labels_community = np.array([assignment.get(u, -1) for u in covered])
            community_rows.append({
                'snapshot': name,
                'attribute': attribute,
                'edges_same_value': same,
                'random_same_value': expected,
                'lift': same / expected if expected else float('nan'),
                'ari_vs_communities': adjusted_rand_score(labels_attribute, labels_community),
                'nmi_vs_communities': normalized_mutual_info_score(labels_attribute, labels_community),
                'users_covered': len(covered),
            })
        print(f'  attributes measured on {name}')

    with open(os.path.join(TABLES, f'content_{folder_name}.csv'), 'w', encoding='utf-8',
              newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if community_rows:
        with open(os.path.join(TABLES, f'content_attributes_{folder_name}.csv'), 'w',
                  encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(community_rows[0]))
            writer.writeheader()
            writer.writerows(community_rows)

    labels = [row['snapshot'] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(labels, [row['edge_content_similarity'] for row in rows], marker='o', label='edges')
    axes[0].plot(labels, [row['random_content_similarity'] for row in rows], marker='s',
                 label='random pairs')
    axes[0].set_title('content similarity')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    if community_rows:
        last = [row for row in community_rows if row['snapshot'] == rows[-1]['snapshot']]
        axes[1].bar([row['attribute'] for row in last], [row['lift'] for row in last])
        axes[1].axhline(1.0, color='gray', linestyle='--', linewidth=1)
        axes[1].set_title('attribute homophily (lift)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(alpha=0.3, axis='y')
    figure.suptitle(folder_name)
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, f'content_{folder_name}_summary.png'))
    plt.close(figure)
    print(f'written results/tables/content_{folder_name}.csv')


if __name__ == '__main__':
    main()
