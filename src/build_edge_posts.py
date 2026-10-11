"""Does the edge live on a post? Build the edge -> post map on the real dataset.

Convention (decided with the user): in an edge (left, right) the LEFT user owns the post, so an
edge is read as "a post written by the left user is the interaction that links the two". The
reciprocal edge (right, left) is a DIFFERENT post, written by the other user.

The .edg files carry only user ids, so the post attached to an edge is not recorded: what we can
do is check whether the reading is compatible with the data. For every transition (new edges
between two consecutive snapshots) we measure:

    - coverage: how many new edges have the source (and the target) writing a post in the window
        and, for comparison, by the cumulative posts file
    - how many new edges a single post has to explain (edges per post in the window)
    - reciprocity: how many new edges come with their reverse in the same window, and whether the
        two legs are explained by two different posts (one per owner) as the convention requires

Note on the two posts files (names are counter-intuitive):
    posts_current_snapshot.tsv = the posts written INSIDE the window
    posts_incremental.tsv      = all posts up to that snapshot (cumulative)

Usage: uv run python src/build_edge_posts.py
Output: results/edge_posts_report.csv, results/_edge_posts_run.txt
"""

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
REAL = os.path.join(BASE, 'real_snapshots')
RESULTS = os.path.join(ROOT, 'results')
REPORT = os.path.join(RESULTS, 'edge_posts_report.csv')
LOG = os.path.join(RESULTS, '_edge_posts_run.txt')


def log(message):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message, flush=True)


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


def load_posts_by_user(path):
    """{user: [post_id, ...]} from a posts file."""
    posts = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = [h.strip() for h in next(reader)]
        id_col = header.index('id')
        acc_col = header.index('account_id')
        for row in reader:
            if len(row) != len(header):
                continue
            user, post = norm(row[acc_col]), norm(row[id_col])
            if user and post:
                posts.setdefault(user, []).append(post)
    return posts


def main():
    open(LOG, 'w').close()
    names = sorted(d for d in os.listdir(REAL) if os.path.isdir(os.path.join(REAL, d)))
    log(f'# edges-as-posts | {len(names)} snapshots | ids in .edg are account_id (users)')

    directories = [os.path.join(REAL, name) for name in names]
    edge_sets = [load_edges(os.path.join(d, 'social_network.edg')) for d in directories]
    window = [load_posts_by_user(os.path.join(d, 'posts_current_snapshot.tsv')) for d in directories]
    cumulative = [load_posts_by_user(os.path.join(d, 'posts_incremental.tsv')) for d in directories]

    rows = []
    for step in range(1, len(directories)):
        new = edge_sets[step] - edge_sets[step - 1]
        if not new:
            continue
        w_posts, c_posts = window[step], cumulative[step]
        n_posts = sum(len(p) for p in w_posts.values())
        source_window = sum(1 for a, _ in new if a in w_posts)
        target_window = sum(1 for _, b in new if b in w_posts)
        source_cumulative = sum(1 for a, _ in new if a in c_posts)
        target_cumulative = sum(1 for _, b in new if b in c_posts)

        reverse = {(b, a) for a, b in new}
        reciprocated = sum(1 for edge in new if edge in reverse)

        row = {
            'snapshot': names[step],
            'new_edges': len(new),
            'posts_in_window': n_posts,
            'edges_per_post': round(len(new) / n_posts, 3) if n_posts else None,
            'source_window_share': round(source_window / len(new), 4),
            'target_window_share': round(target_window / len(new), 4),
            'source_cumulative_share': round(source_cumulative / len(new), 4),
            'target_cumulative_share': round(target_cumulative / len(new), 4),
            'reciprocated_edges': reciprocated,
            'reciprocated_share': round(reciprocated / len(new), 4),
        }
        rows.append(row)
        log(f'  {names[step]:24s} new {len(new):7d} | posts {n_posts:6d} '
            f'({row["edges_per_post"]} edges/post) | post of the source: window '
            f'{row["source_window_share"]:.3f} cumul {row["source_cumulative_share"]:.3f} | '
            f'post of the target: window {row["target_window_share"]:.3f} cumul '
            f'{row["target_cumulative_share"]:.3f} | reciprocated {row["reciprocated_share"]:.4f}')

    os.makedirs(RESULTS, exist_ok=True)
    with open(REPORT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    total_edges = sum(r['new_edges'] for r in rows)
    total_posts = sum(r['posts_in_window'] for r in rows)
    log(f'overall: {total_edges} new edges explained by {total_posts} posts in the windows '
        f'({total_edges / total_posts:.2f} edges per post)')
    log(f'written {REPORT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
