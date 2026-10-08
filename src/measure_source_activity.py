"""Measure what decides the design of the edge-formation model: does the source of a new edge
have to be active (write a post) in that period?

For every real transition t (from t-1 to t) it reports:
    - new edges in the transition
    - share whose SOURCE wrote a post in that period
    - share whose TARGET wrote a post in that period
    - share with at least one active endpoint

If the source is active in ~100% of the new edges, "the source must be active" is a fact of the
data and belongs in the definition of the candidates; otherwise activity stays a variable that
the model weighs on its own.

Usage: uv run python src/measure_source_activity.py
Output: results/source_activity.csv
"""

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
REAL = os.path.join(BASE, 'real_snapshots')
RESULTS = os.path.join(ROOT, 'results')
REPORT = os.path.join(RESULTS, 'source_activity.csv')


def load_edges(path):
    edges = set()
    with open(path, encoding='utf-8-sig', newline='') as f:
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                edges.add((parts[0].strip(), parts[1].strip()))
    return edges


def active_users(directory):
    """Users that wrote at least one post in this snapshot (the per-period posts file)."""
    users = set()
    path = os.path.join(directory, 'posts_current_snapshot.tsv')
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = [h.strip() for h in next(reader)]
        author = header.index('account_id')
        for row in reader:
            if len(row) == len(header):
                users.add(row[author].strip())
    return users


def main():
    names = sorted(os.listdir(REAL))
    directories = [os.path.join(REAL, name) for name in names]
    edge_sets = [load_edges(os.path.join(d, 'social_network.edg')) for d in directories]
    active = [active_users(d) for d in directories]
    print(f'snapshots {len(directories)} | active users per snapshot '
          f'{[len(a) for a in active]}')

    rows = []
    for step in range(1, len(directories)):
        new = edge_sets[step] - edge_sets[step - 1]
        if not new:
            continue
        source_active = sum(1 for a, _ in new if a in active[step])
        target_active = sum(1 for _, b in new if b in active[step])
        either = sum(1 for a, b in new if a in active[step] or b in active[step])
        row = {'snapshot': names[step], 'new_edges': len(new),
               'source_active': source_active,
               'source_active_share': round(source_active / len(new), 4),
               'target_active_share': round(target_active / len(new), 4),
               'either_active_share': round(either / len(new), 4)}
        rows.append(row)
        print(f'  {names[step]}: new edges {len(new)} | source active '
              f'{row["source_active_share"]:.3f} | target active '
              f'{row["target_active_share"]:.3f} | either {row["either_active_share"]:.3f}')

    os.makedirs(RESULTS, exist_ok=True)
    with open(REPORT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    total = sum(row['new_edges'] for row in rows)
    weighted = sum(row['source_active'] for row in rows) / total
    print(f'overall: source active in {weighted:.4f} of the new edges '
          f'({sum(row["source_active"] for row in rows)}/{total})')
    print(f'written {REPORT}')


if __name__ == '__main__':
    main()
