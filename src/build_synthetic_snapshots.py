"""Partition the synthetic dataset into 9 time steps (one snapshot = one time step).

The synthetic dataset has no time, so the time axis is imposed with explicit rules. Two things
come from the REAL dataset (measured there, never invented):

  arrival curve   the share of users entering at each snapshot (0.418, 0.142, ...)
  activity shape  WHEN users write: the aggregate posts-per-step curve and the empirical
                  distribution of posts per active user per period (bursts and silent steps)

What is ours, and declared: the order of arrival (users sorted by their first post id, assuming
the ids reflect the generation order), the order inside a step (by id) and the variant
"all" (control without accretion).

Variants:
  'all'          every user exists from the first snapshot
  'growth'       users enter progressively with the real arrival curve

Activity modes:
  'from-real'    per-user posts sampled from the real activity distribution and placed on the
                 real per-step curve  -> folders *_v2 (the definitive partitions)
  'uniform'      the older rule: each user's posts split into equal contiguous chunks
                 -> the historical folders (kept as a control)

Output per variant (same layout as the real dataset):
    <folder>/snapshot_<t>/posts_current_snapshot.tsv   posts written in that time step
    <folder>/snapshot_<t>/posts_incremental.tsv        posts up to that time step
    <folder>/users_by_birth.csv                        birth snapshot of every user

Usage:
    uv run python src/build_synthetic_snapshots.py --activity from-real
    uv run python src/build_synthetic_snapshots.py --activity uniform      # historical
"""

import argparse
import collections
import csv
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
SOURCE = os.path.join(BASE, 'synthetic_snapshots')
REAL = os.path.join(BASE, 'real_snapshots')
N_SNAPSHOTS = 9
SEED = 0
# share of users entering at each snapshot, measured on the real dataset
GROWTH_SHARE = (0.418, 0.142, 0.110, 0.072, 0.078, 0.051, 0.010, 0.077, 0.042)
FOLDERS = {
    'uniform': {'all': 'generated_snapshots', 'growth': 'generated_snapshots_growth'},
    'from-real': {'all': 'synthetic_all_present_posts',
                  'growth': 'synthetic_progressive_posts'},
}


def norm_id(value):
    value = value.strip()
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def read_posts():
    rows, header = [], None
    for path in sorted(glob.glob(os.path.join(SOURCE, '*.tsv'))):
        with open(path, encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            file_header = [h.strip() for h in next(reader)]
            if header is None:
                header = file_header
            assert header == file_header, f'header mismatch in {path}'
            for row in reader:
                if len(row) == len(header):
                    rows.append(row)
    return header, rows


def real_activity():
    """(posts per snapshot, posts per active user per period) measured on the real dataset."""
    curve, per_user = [], []
    for name in sorted(os.listdir(REAL)):
        counts = collections.Counter()
        path = os.path.join(REAL, name, 'posts_current_snapshot.tsv')
        with open(path, encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            header = [h.strip() for h in next(reader)]
            author = header.index('account_id')
            for row in reader:
                if len(row) == len(header):
                    counts[row[author].strip()] += 1
        curve.append(sum(counts.values()))
        per_user.extend(counts.values())
    return np.array(curve, dtype=float), np.array(per_user, dtype=int)


def sample_composition(total, span, empirical, rng):
    """How many posts at each of the `span` steps of a user's life, summing exactly to `total`.

    The counts come from the real's empirical distribution of posts per active user per period
    (so bursts and silent steps appear), then the remainder is spread uniformly and any excess is
    trimmed. We deliberately do NOT weight by the real's per-step volumes: those depend on the
    unequal length of the real snapshots (one year, three months, twenty days, three days), so
    they are not a measure of activity and they would concentrate every burst on the same steps.
    """
    if total <= 0:
        return [0] * span
    values = []
    while sum(values) < total and len(values) < span:
        values.append(int(rng.choice(empirical)))
    values.extend([0] * (span - len(values)))
    difference = total - sum(values)
    if difference > 0:
        extra = rng.multinomial(difference, np.full(span, 1.0 / span))
        values = [value + int(add) for value, add in zip(values, extra)]
    elif difference < 0:
        overflow = -difference
        while overflow > 0:
            index = int(rng.integers(0, span))
            take = min(overflow, values[index])
            values[index] -= take
            overflow -= take
    return values


def assign(header, rows, variant, activity, curve, empirical):
    user_col, id_col = header.index('account_id'), header.index('id')
    per_user = collections.defaultdict(list)
    for position, row in enumerate(rows):
        user = norm_id(row[user_col])
        try:
            post_id = int(float(row[id_col]))
        except ValueError:
            post_id = 10 ** 9 + position
        per_user[user].append((post_id, row))

    users = sorted(per_user, key=lambda u: min(post_id for post_id, _ in per_user[u]))
    if variant == 'growth':
        births = {}
        share_index, target = 0, GROWTH_SHARE[0] * len(users)
        for position, user in enumerate(users):
            if position >= target and share_index < N_SNAPSHOTS - 1:
                share_index += 1
                target += GROWTH_SHARE[share_index] * len(users)
            births[user] = share_index
    else:
        births = {user: 0 for user in users}

    periods = [[] for _ in range(N_SNAPSHOTS)]
    rng = np.random.default_rng(SEED)
    for user in users:
        ordered = sorted(per_user[user])
        n, birth = len(ordered), births[user]
        span = N_SNAPSHOTS - birth
        if activity == 'from-real':
            counts = sample_composition(n, span, empirical, rng)
            order = rng.permutation(span)          # bursts placed at random inside the user's life
            placement = [0] * span
            for value, step in zip(counts, order):
                placement[int(step)] = value
        else:
            placement = [0] * span
            for index in range(n):
                placement[min(span - 1, (index * span) // n)] += 1
        written = 0
        for step in range(span):
            take = placement[step]
            for post_id, row in ordered[written:written + take]:
                periods[birth + step].append(row)
            written += take
    return periods, births


def write(folder, header, periods, births):
    os.makedirs(folder, exist_ok=True)
    for t in range(N_SNAPSHOTS):
        directory = os.path.join(folder, f'snapshot_{t + 1}')
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'posts_current_snapshot.tsv'), 'w',
                  encoding='utf-8', newline='') as f:
            f.write('\t'.join(header) + '\r\n')
            for row in periods[t]:
                f.write('\t'.join(row) + '\r\n')
        with open(os.path.join(directory, 'posts_incremental.tsv'), 'w',
                  encoding='utf-8', newline='') as f:
            f.write('\t'.join(header) + '\r\n')
            for s in range(t + 1):
                for row in periods[s]:
                    f.write('\t'.join(row) + '\r\n')
    with open(os.path.join(folder, 'users_by_birth.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['account_id', 'birth_snapshot'])
        for user, birth in sorted(births.items()):
            writer.writerow([user, birth + 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activity', default='from-real', choices=('from-real', 'uniform'))
    args = parser.parse_args()

    header, rows = read_posts()
    print(f'posts {len(rows)} | columns {len(header)} | activity mode {args.activity}')
    curve, empirical = real_activity()
    print(f'real activity curve (posts per snapshot): {curve.astype(int).tolist()}')
    print(f'real posts per active user per period: mean {empirical.mean():.1f} '
          f'median {np.median(empirical):.0f} p90 {np.percentile(empirical, 90):.0f} '
          f'max {empirical.max()}')
    for variant in ('all', 'growth'):
        periods, births = assign(header, rows, variant, args.activity, curve, empirical)
        folder = os.path.join(BASE, FOLDERS[args.activity][variant])
        write(folder, header, periods, births)
        sizes = [len(p) for p in periods]
        unique = [len({norm_id(row[header.index("account_id")]) for row in p}) for p in periods]
        print(f'[{variant}/{args.activity}] -> {os.path.basename(folder)}: posts per snapshot '
              f'{sizes} | users per snapshot {unique}')


if __name__ == '__main__':
    main()
