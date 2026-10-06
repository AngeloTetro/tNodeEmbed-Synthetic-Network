"""Partition the synthetic dataset into 9 time steps (one snapshot = one time step).

The synthetic dataset has no time, so the time axis is imposed with explicit rules:
  - posts of each user are ordered by their post id (in the synthetic dataset ids are a
    dense sequence, i.e. the generation order) and distributed over the snapshots;
  - variant "all": every user exists from the first snapshot, posts split into 9
    contiguous chunks per user;
  - variant "growth": users enter progressively, following the real dataset's growth curve
    (the share of users entering at each snapshot), and each user's posts start from
    their birth snapshot.

Output per variant (same layout as the real dataset):
    <folder>/snapshot_<t>/posts_current_snapshot.tsv   posts written in that time step
    <folder>/snapshot_<t>/posts_incremental.tsv        posts up to that time step
"""

import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
SOURCE = os.path.join(BASE, 'synthetic_snapshots')
N_SNAPSHOTS = 9
# share of users entering at each snapshot, from the real dataset (10.755 -> 25.714 nodes)
GROWTH_SHARE = (0.418, 0.142, 0.110, 0.072, 0.078, 0.051, 0.010, 0.077, 0.042)


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


def assign(header, rows, variant):
    user_col, id_col = header.index('account_id'), header.index('id')
    per_user = {}
    for position, row in enumerate(rows):
        user = norm_id(row[user_col])
        try:
            post_id = int(float(row[id_col]))
        except ValueError:
            post_id = 10 ** 9 + position
        per_user.setdefault(user, []).append((post_id, row))

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
    for user in users:
        ordered = sorted(per_user[user])
        n, birth = len(ordered), births[user]
        span = N_SNAPSHOTS - birth
        for index, (post_id, row) in enumerate(ordered):
            snapshot = birth + min(span - 1, (index * span) // n)
            periods[snapshot].append(row)
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
    header, rows = read_posts()
    print(f'posts {len(rows)} | columns {len(header)}')
    for variant, folder_name in (('all', 'generated_snapshots'),
                                 ('growth', 'generated_snapshots_growth')):
        periods, births = assign(header, rows, variant)
        folder = os.path.join(BASE, folder_name)
        write(folder, header, periods, births)
        sizes = [len(p) for p in periods]
        unique = [len({norm_id(row[header.index("account_id")]) for row in p}) for p in periods]
        print(f'[{variant}] -> {folder_name}: posts per snapshot {sizes} | users per snapshot {unique}')


if __name__ == '__main__':
    main()
