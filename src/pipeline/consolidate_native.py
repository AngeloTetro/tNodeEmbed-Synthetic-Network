"""Report on the native synthetic batches, and consolidation of a generated folder onto them.

Measured fact, and it is the reason a "back to the 4 native snapshots" delivery has to be done
carefully: the native files are NOT time slices. Each of batch_0..batch_2 covers the whole
post-id range (min ~0, max ~100.450, median ~50.000) and shares about 9.500 of its ~9.700 users
with the other two, while batch_3 holds two rows. The batch index therefore carries no temporal
order at all, which is why the 9-step partition imposes the time axis itself
(see build_synthetic_snapshots.py and docs/pipeline.md Â§2).

The consolidation below is by USER MEMBERSHIP, not by time: for every native file, the edges of
the generated network whose two endpoints are both users of that file. With 95% of the users
shared, the four files are nearly the same network, and that is the honest result: the native
split cannot be reproduced as four periods, because it is not one.

Usage: uv run python -m src.pipeline.consolidate_native <folder_name> [--step N]
Output: results/native_batches.csv,
        <folder>/native_4_batch_<i>.edg, <folder>/native_4.csv

The files are written in the ROOT of the dataset folder, not in a subdirectory: every
subdirectory of a dataset folder is read as a snapshot by graph_io.snapshots_folder.
"""

import argparse
import collections
import csv
import os

from src.paths import ROOT, BASE, TABLES

from src.lib.graph_io import load_network, norm

SOURCE = os.path.join(BASE, 'synthetic_snapshots')
N_BATCHES = 4


def read_batch(index):
    """(users, post ids, first post id per user) of one native file."""
    path = os.path.join(SOURCE, f'batch_{index}_synthetic.tsv')
    users, ids, first = set(), [], {}
    if not os.path.exists(path):
        return users, ids, first
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = [h.strip() for h in next(reader)]
        id_col, user_col = header.index('id'), header.index('account_id')
        for row in reader:
            if len(row) != len(header):
                continue
            user = norm(row[user_col])
            users.add(user)
            try:
                value = int(float(row[id_col]))
            except ValueError:
                continue
            ids.append(value)
            if user not in first or value < first[user]:
                first[user] = value
    return users, ids, first


def median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--step', type=int, default=9)
    args = parser.parse_args()

    batches = [read_batch(index) for index in range(N_BATCHES)]
    folder = os.path.join(BASE, args.folder)
    snapshots = load_network(folder)
    wanted = f'snapshot_{args.step}'
    name, edges, nodes = next((item for item in snapshots if item[0] == wanted), snapshots[-1])
    print(f'{args.folder}: snapshot {name} (step {args.step}) with {len(nodes)} users and '
          f'{len(edges)} edges')

    rows = []
    for index, (users, ids, first) in enumerate(batches):
        kept = {(a, b) for a, b in edges if a in users and b in users}
        kept_nodes = {n for edge in kept for n in edge}
        overlaps = [len(users & other) for other_index, (other, _, _) in enumerate(batches)
                    if other_index != index]
        rows.append({
            'batch': f'batch_{index}',
            'posts': len(ids),
            'users': len(users),
            'post_id_min': min(ids) if ids else '',
            'post_id_median': median(ids) if ids else '',
            'post_id_max': max(ids) if ids else '',
            'first_post_id_median': median(first.values()) if first else '',
            'overlap_with_others': ' | '.join(str(value) for value in overlaps),
            'users_also_in_generated': sum(1 for user in users if user in nodes),
            'edges_written': len(kept),
            'nodes_written': len(kept_nodes),
        })

    fields = list(rows[0])
    with open(os.path.join(TABLES, 'native_batches.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for index, (users, _, _) in enumerate(batches):
        kept = sorted((a, b) for a, b in edges if a in users and b in users)
        with open(os.path.join(folder, f'native_4_batch_{index}.edg'), 'w', encoding='utf-8',
                  newline='') as f:
            for source, target in kept:
                f.write(f'{source}\t{target}\r\n')
    with open(os.path.join(folder, 'native_4.csv'), 'w', encoding='utf-8',
              newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"  {row['batch']}: {row['posts']:>6} posts | {row['users']:>5} users | "
              f"ids {row['post_id_min']} to {row['post_id_max']} (median {row['post_id_median']}"
              f") | shared with the others: {row['overlap_with_others']} | "
              f"edges written {row['edges_written']:>7} of {len(edges)}")
    print()
    print('The four files are not temporal slices: each covers the whole id range and shares almost '
          'every user, so the files written are the same network restricted to the users of that '
          'batch (the only faithful consolidation).')
    print(f'written results/native_batches.csv and {args.folder}/native_4_*.edg')


if __name__ == '__main__':
    main()
