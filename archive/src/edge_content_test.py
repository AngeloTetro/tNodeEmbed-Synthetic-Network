"""Does the post carry the link? The content test at the edge level.

Convention (decided with the user): in an edge (left, right) the LEFT user owns the post, so the
edge is read as "a post written by the left user is the interaction that links the two". The
reciprocal edge (right, left) is a DIFFERENT post, written by the other user.

The project so far represented a user with the MEAN of their posts, which averages away exactly
the effect we are testing. Here every post keeps its own vector and we compare, on the same pairs
and the same protocol (positives = new edges of a transition, negatives paired on arrival and
previous degree), the discriminative power of:

    node_mean        cos(mean of the left user's posts, mean of the right user's posts)   [baseline]
    best_post_left   max over the left user's posts of cos(post, right user's mean)      [convention]
    best_post_right  max over the right user's posts of cos(left user's mean, post)
    best_post_post   max over pairs of posts of cos(post_left, post_right)
    mean_post_post   mean over pairs of posts of cos(post_left, post_right)
    topk_post_post   mean of the largest k values of cos(post_left, post_right)

A second measurement looks at reciprocity as the user described it: for a reciprocated couple the
two legs are two different posts, one per owner. We check whether the two posts are more similar
than the posts of two random, non-connected users.

Usage: uv run python src/edge_content_test.py [--encoder minilm] [--content period|cumulative]
                                             [--pairs 20000] [--topk 3]
Output: results/edge_content_report.csv, results/reciprocity_content.csv, results/_edge_content_run.txt
"""

import argparse
import csv
import os
import random
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from content_encoders import get_encoder
from transfer_to_synthetic import prepare_pairs
from tnodeembed_real import (BASE, N_SNAPSHOTS, TEST_STEPS, TRAIN_STEPS, VAL_STEP, load_docs,
                             load_edges, norm)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
CACHE = os.path.join(ROOT, 'models', 'real_posts')
REPORT = os.path.join(RESULTS, 'edge_content_report.csv')
RECIPROCITY = os.path.join(RESULTS, 'reciprocity_content.csv')
LOG = os.path.join(RESULTS, '_edge_content_run.txt')
SEED = 0
FEATURES = ('node_mean', 'best_post_left', 'best_post_right', 'best_post_post',
            'mean_post_post', 'topk_post_post')


def log(message):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message, flush=True)


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def snapshot_names():
    return sorted(d for d in os.listdir(os.path.join(BASE, 'real_snapshots'))
                  if os.path.isdir(os.path.join(BASE, 'real_snapshots', d)))


def load_post_vectors(encoder, docs, cache_tag):
    """{user: (n_posts, dim) matrix}, one row per post."""
    texts, keys = [], {}
    for user, user_docs in docs.items():
        for text in user_docs:
            if text not in keys:
                keys[text] = len(texts)
                texts.append(text)
    path = os.path.join(CACHE, f'{cache_tag}_{len(texts)}.npz')
    if os.path.exists(path):
        log(f'  encoded vectors from cache: {path}')
        with np.load(path) as data:
            encoded = data['vectors'].astype(np.float32)
    else:
        start = time.time()
        encoded = encoder.transform(texts)
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, vectors=encoded)
        log(f'  encoded {len(texts)} posts in {time.time() - start:.0f}s -> {path}')
    by_user = {}
    for user, user_docs in docs.items():
        rows = [keys[text] for text in user_docs]
        if rows:
            by_user[user] = encoded[rows]
    return by_user


def pair_feature(vectors, left_posts, right_posts, right_mean, left_mean, topk):
    """All the content features of a pair, from the individual posts of the two users."""
    scores = {'node_mean': float(left_mean @ right_mean)}
    if left_posts is not None:
        scores['best_post_left'] = float(np.max(left_posts @ right_mean))
    if right_posts is not None:
        scores['best_post_right'] = float(np.max(right_posts @ left_mean))
    if left_posts is not None and right_posts is not None:
        cross = left_posts @ right_posts.T
        flat = cross.reshape(-1)
        scores['best_post_post'] = float(flat.max())
        scores['mean_post_post'] = float(flat.mean())
        k = min(topk, flat.size)
        scores['topk_post_post'] = float(np.sort(flat)[-k:].mean())
    return scores


def summarize(labels, feature_values):
    out = {}
    for name in FEATURES:
        values = feature_values.get(name)
        if values is None:
            continue
        out[name] = roc_auc_score(labels, values)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='minilm')
    parser.add_argument('--content', default='period', choices=('period', 'cumulative'))
    parser.add_argument('--pairs', type=int, default=20000)
    parser.add_argument('--topk', type=int, default=3)
    args = parser.parse_args()

    open(LOG, 'w').close()
    random.seed(SEED)
    np.random.seed(SEED)
    names = snapshot_names()
    log(f'# edges-as-posts | content test | encoder {args.encoder} | content file '
        f'{"posts_current_snapshot (the window)" if args.content == "period" else "posts_incremental (cumulative)"}')

    file_name = 'posts_current_snapshot.tsv' if args.content == 'period' else 'posts_incremental.tsv'
    directories = [os.path.join(BASE, 'real_snapshots', name) for name in names]
    edge_sets = [load_edges(os.path.join(d, 'social_network.edg')) for d in directories]
    docs = [load_docs(os.path.join(d, file_name)) for d in directories]

    users = sorted({u for d in docs for u in d})
    index = {u: i for i, u in enumerate(users)}
    log(f'users {len(users)} | posts per snapshot {[sum(len(v) for v in d.values()) for d in docs]}')

    encoder = get_encoder(args.encoder)
    encoder.fit([text for t in TRAIN_STEPS for u in users for text in docs[t].get(u, [])])

    post_vectors, means, masks = [], [], []
    from content_encoders import encode_grouped
    for t in range(N_SNAPSHOTS):
        vectors = load_post_vectors(encoder, docs[t], f'{args.encoder}_{args.content}_t{t}')
        post_vectors.append(vectors)
        groups = [docs[t].get(u, []) for u in users]
        matrices = encode_grouped(encoder, groups) if any(groups) else np.zeros((len(users), 1))
        means.append(matrices)
        masks.append(np.array([bool(g) for g in groups]))
        log(f'  t={t} ({names[t]}): posts {sum(len(v) for v in vectors.values())} | '
            f'users with posts {int(masks[t].sum())}')

    build = prepare_pairs(edge_sets, users, index, means, masks, 'paired')
    all_rows = []
    reciprocity_rows = []
    for step in range(1, N_SNAPSHOTS):
        positives, negatives = build(step)
        pairs = positives + negatives
        labels = [1] * len(positives) + [0] * len(negatives)
        values = {name: [] for name in FEATURES}
        for a, b in pairs:
            left_posts = post_vectors[step].get(a)
            right_posts = post_vectors[step].get(b)
            scores = pair_feature(post_vectors, left_posts, right_posts,
                                  means[step][index[b]], means[step][index[a]], args.topk)
            for name in FEATURES:
                values[name].append(scores.get(name, 0.0))
        summary = summarize(labels, values)
        for name, auc in summary.items():
            all_rows.append({'step': step, 'snapshot': names[step], 'feature': name,
                             'auc': round(auc, 4), 'pairs': len(pairs)})
        text = ' | '.join(f'{name} {summary[name]:.4f}' for name in FEATURES if name in summary)
        log(f'  step {step} ({names[step]}): {text}')

        delta = {(a, b) for a, b in (edge_sets[step] - edge_sets[step - 1]) if a in index and b in index}
        reciprocal = [(a, b) for a, b in delta if (b, a) in delta]
        sample = random.sample(reciprocal, min(2000, len(reciprocal)))
        rec_values, ctrl_values = [], []
        for a, b in sample:
            left_posts, right_posts = post_vectors[step].get(a), post_vectors[step].get(b)
            if left_posts is None or right_posts is None:
                continue
            rec_values.append(float((left_posts @ right_posts.T).max()))
            c, d = random.choice(users), random.choice(users)
            c_posts, d_posts = post_vectors[step].get(c), post_vectors[step].get(d)
            if c != d and c_posts is not None and d_posts is not None:
                ctrl_values.append(float((c_posts @ d_posts.T).max()))
        if rec_values and ctrl_values:
            auc = roc_auc_score([1] * len(rec_values) + [0] * len(ctrl_values),
                                rec_values + ctrl_values)
            reciprocity_rows.append({'step': step, 'snapshot': names[step],
                                     'reciprocated_pairs': len(rec_values),
                                     'best_post_post_reciprocated': round(float(np.mean(rec_values)), 4),
                                     'best_post_post_random': round(float(np.mean(ctrl_values)), 4),
                                     'auc_vs_random': round(auc, 4)})
            log(f'    reciprocity: {len(rec_values)} couples | best post-post '
                f'{np.mean(rec_values):.4f} vs random {np.mean(ctrl_values):.4f} | AUC {auc:.4f}')

    os.makedirs(RESULTS, exist_ok=True)
    with open(REPORT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    if reciprocity_rows:
        with open(RECIPROCITY, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(reciprocity_rows[0]))
            writer.writeheader()
            writer.writerows(reciprocity_rows)

    log('')
    log('  mean AUC over the protocol steps (validation + test):')
    for name in FEATURES:
        aucs = [row['auc'] for row in all_rows if row['feature'] == name
                and row['step'] in (VAL_STEP,) + tuple(TEST_STEPS)]
        if aucs:
            log(f'    {name:16s} {np.mean(aucs):.4f}   (per step {aucs})')
    log(f'written {REPORT} and {RECIPROCITY}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
