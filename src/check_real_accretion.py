"""Do new users connect to users from previous snapshots? Same check on every dataset.

For every transition:
  - new edges, and how many have BOTH endpoints already present (old->old);
  - the direction break-down: new->old, old->new, new->new (a node is "new" at t if it
    appears in the network for the first time in that snapshot);
  - the previous degree of the existing endpoint of a new edge (hub preference).

Usage: uv run python src/check_real_accretion.py [folder_name]
Output: results/accretion_<folder_name>.csv
"""

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
FOLDER_NAME = sys.argv[1] if len(sys.argv) > 1 else 'real_snapshots'
RESULTS = os.path.join(ROOT, 'results')


def norm(value):
    value = value.strip()
    if not value:
        return None
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def load_edges(path):
    edges = set()
    with open(path, encoding='utf-8-sig', newline='') as f:
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                source, target = norm(parts[0]), norm(parts[1])
                if source != target:
                    edges.add((source, target))
    return edges


def main():
    folder = os.path.join(BASE, FOLDER_NAME)
    folders = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
    edge_sets = [load_edges(os.path.join(folder, name, 'social_network.edg')) for name in folders]
    rows = []
    total = {'new_edges': 0, 'old_old': 0, 'new_old': 0, 'old_new': 0, 'new_new': 0}
    for index in range(1, len(folders)):
        previous_edges = edge_sets[index - 1]
        previous_nodes = {n for edge in previous_edges for n in edge}
        delta = edge_sets[index] - previous_edges
        degree = {}
        for a, b in previous_edges:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1

        old_old = [(a, b) for a, b in delta if a in previous_nodes and b in previous_nodes]
        new_old = [(a, b) for a, b in delta if a not in previous_nodes and b in previous_nodes]
        old_new = [(a, b) for a, b in delta if a in previous_nodes and b not in previous_nodes]
        new_new = [(a, b) for a, b in delta if a not in previous_nodes and b not in previous_nodes]
        existing_end_degree = [degree.get(b, 0) for _, b in new_old] + \
                              [degree.get(a, 0) for a, _ in old_new]
        mean_degree = sum(existing_end_degree) / len(existing_end_degree) if existing_end_degree else 0.0
        mean_all_degree = sum(degree.values()) / len(degree) if degree else 0.0

        rows.append({
            'snapshot': folders[index],
            'new_edges': len(delta),
            'old_old': len(old_old),
            'share_old_old': f'{len(old_old) / len(delta):.4%}' if delta else '',
            'new_to_old': len(new_old),
            'old_to_new': len(old_new),
            'new_to_new': len(new_new),
            'mean_prev_degree_of_existing_end': round(mean_degree, 1),
            'mean_prev_degree_all_nodes': round(mean_all_degree, 1),
        })
        total['new_edges'] += len(delta)
        total['old_old'] += len(old_old)
        total['new_old'] += len(new_old)
        total['old_new'] += len(old_new)
        total['new_new'] += len(new_new)
        print(f'{folders[index]:>20}: new {len(delta):>7} | old->old {len(old_old):>5} | '
              f'new->old {len(new_old):>6} | old->new {len(old_new):>6} | new->new {len(new_new):>5} '
              f'| prev degree of the existing end {mean_degree:>6.1f} (all nodes {mean_all_degree:.1f})')

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f'accretion_{FOLDER_NAME}.csv'), 'w', encoding='utf-8',
              newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTOTAL: new edges {total['new_edges']} | old->old {total['old_old']} | "
          f"new->old {total['new_old']} | old->new {total['old_new']} | new->new {total['new_new']}")
    print(f'written results/accretion_{FOLDER_NAME}.csv')


if __name__ == '__main__':
    main()
