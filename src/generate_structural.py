"""V8 -- the definitive edge-formation model: fitted on the real transitions, then sampled on the
synthetic network with feedback, step by step.

What is learned (fitted on the real training transitions; the coefficients are the scientific
result):

    content_score            the tNodeEmbed head score for the pair (the text signal)
    target/source_degree_pct percentile of the degree in the current population (transferable
                             across populations of different size)
    new_source/new_target    the endpoint has no edge yet
    reverse_edge             the opposite direction already exists (reciprocity)
    common_neighbours        log1p of common neighbours (triadic closure)
    source/target_active     the endpoint wrote a post in this period
    recency_target           steps since the target last received an edge
    step                     normalised time index

What is measured, not chosen (results/source_activity.csv): on the real, **at least one endpoint
is active in 100% of the new edges**, so candidates are the pairs involving at least one active
endpoint; the *source* is active in 68-81%, so that stays a variable.

Generation: every candidate pair becomes an edge with its own corrected probability (sampling, no
threshold); after every step the state is recomputed from the generated network (feedback), so
volume, degree distribution and hubs emerge instead of being imposed.

Global quantities are computed once per time step (the common-neighbour product is a sparse
matrix product); only the per-pair evaluation is done in row blocks.

Content modes (`--content`); only the synthetic side is affected, the model is always fitted on the
real content:
    real      the authentic user profiles;
    permuted  the same profiles reassigned to other users (one permutation for all the steps, same
              statistics, same supervision): the control that tells whether the text carries the
              link at all;
    off       the content channel zeroed (the ablation).

Usage:
    uv run python src/generate_structural.py --partition generated_snapshots_v2 --content real
    uv run python src/generate_structural.py --partition generated_snapshots_v2 --content permuted
    uv run python src/generate_structural.py --partition generated_snapshots_v2 --content off
Output: <partition>_<variant>/, results/v8_report.csv, results/v8_coefficients.csv,
        results/_v8_run.txt
"""

import argparse
import csv
import os
import traceback

import numpy as np
import scipy.sparse as sparse
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from apply_edge_rules import edges_to_files, metrics_from_folder
from transfer_to_synthetic import (BASE, DEVICE, N_SNAPSHOTS, SEED, node_embeddings,
                                   prepare_pairs, procrustes_chain, real_representations,
                                   synthetic_representations, train_model)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
REPORT = os.path.join(RESULTS, 'v8_report.csv')
COEFFICIENTS = os.path.join(RESULTS, 'v8_coefficients.csv')
LOG = os.path.join(RESULTS, '_v8_run.txt')
TRAIN_STEPS = (1, 2, 3, 4, 5)
VAL_STEP = 6
NEG_PER_POS = 10
MAX_POS = 30000
BLOCK = 256
PREVALENCE_SAMPLE = 500_000
FEATURES = ['content_score', 'target_degree_pct', 'source_degree_pct', 'new_source', 'new_target',
            'reverse_edge', 'common_neighbours', 'source_active', 'target_active',
            'recency_target', 'step']
FIELDS = ['variant', 'partition', 'content', 'operating_point', 'val_auc', 'val_ap', 'val_auc_gbm',
          'nodes', 'edges', 'density', 'reciprocity', 'mean_out_degree', 'max_out_degree']


def log(text):
    print(text, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def step_state(edges_i, edges_j, n, last_received, step, active):
    """Everything that is global for one time step: sparse adjacency, its product (common
    neighbours), degrees, percentiles, presence, activity."""
    edges_i = np.asarray(edges_i, dtype=np.int32)
    edges_j = np.asarray(edges_j, dtype=np.int32)
    rows = np.concatenate([edges_i, edges_j])
    columns = np.concatenate([edges_j, edges_i])
    matrix = sparse.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, columns)),
                               shape=(n, n))
    matrix.data[:] = 1.0
    product = (matrix @ matrix.T).tocsr()          # number of common neighbours (sharp sparse)
    out_degree = np.bincount(edges_i, minlength=n).astype(np.float32)
    in_degree = np.bincount(edges_j, minlength=n).astype(np.float32)
    degree = out_degree + in_degree
    present = degree > 0
    percentile = np.zeros(n, dtype=np.float32)
    if present.any():
        percentile[present] = (degree[present].argsort().argsort()
                               / max(1, int(present.sum()) - 1)).astype(np.float32)
    return {'matrix': matrix, 'product': product, 'degree': degree, 'percentile': percentile,
            'present': present, 'new': (~present).astype(np.float32), 'active': active,
            'last_received': last_received, 'step': step}


def features_for_pairs(state, scores, left, right):
    """Features for a list of pairs (used on the real, where the pairs are sampled)."""
    n = state['percentile'].size
    recency = np.clip(state['step'] - state['last_received'][right], 0, N_SNAPSHOTS)
    common = np.asarray(state['product'][left, right]).ravel()
    reverse = np.asarray(state['matrix'][left, right]).ravel()
    return np.column_stack([
        scores,
        state['percentile'][right],
        state['percentile'][left],
        state['new'][left],
        state['new'][right],
        reverse.astype(np.float32),
        np.log1p(common).astype(np.float32),
        state['active'][left].astype(np.float32),
        state['active'][right].astype(np.float32),
        recency.astype(np.float32),
        np.full(len(left), state['step'] / (N_SNAPSHOTS - 1), dtype=np.float32),
    ]).astype(np.float32)


def features_for_block(state, scores, start, row_index, column_index):
    """Features for the candidate pairs of one row block, read from the block slices."""
    block = slice(start, start + scores.shape[0])
    common = state['product'][block].toarray()
    reverse = state['matrix'][block].toarray()
    percentile = state['percentile']
    active = state['active']
    recency = np.clip(state['step'] - state['last_received'][column_index], 0, N_SNAPSHOTS)
    return np.column_stack([
        scores[row_index, column_index],
        percentile[column_index],
        percentile[row_index + start],
        state['new'][row_index + start],
        state['new'][column_index],
        reverse[row_index, column_index],
        np.log1p(common[row_index, column_index]),
        active[row_index + start].astype(np.float32),
        active[column_index].astype(np.float32),
        recency.astype(np.float32),
        np.full(row_index.size, state['step'] / (N_SNAPSHOTS - 1), dtype=np.float32),
    ]).astype(np.float32)


def content_scores_for(model, hidden, left, right):
    if len(left) == 0:
        return np.zeros(0, dtype=np.float32)
    scores = []
    with torch.no_grad():
        for start in range(0, len(left), 100_000):
            i = torch.from_numpy(np.asarray(left[start:start + 100_000], dtype=np.int64)).to(DEVICE)
            j = torch.from_numpy(np.asarray(right[start:start + 100_000], dtype=np.int64)).to(DEVICE)
            pair = torch.cat([hidden[i], hidden[j], hidden[i] * hidden[j]], dim=1)
            scores.append(model.head(pair).squeeze(1).cpu().numpy())
    return np.concatenate(scores)


# ---------------------------------------------------------------- real dataset
def load_real():
    encoder, users, index, edge_sets, matrices, masks = real_representations('minilm', 'period',
                                                                            'mean_of_posts')
    aligned = procrustes_chain(matrices, masks)
    build = prepare_pairs(edge_sets, users, index, matrices, masks, 'matched')
    model = train_model(aligned, build, index, matrices[0].shape[1])
    return encoder, users, index, edge_sets, masks, model, node_embeddings(model, matrices)


def real_arrays(edge_sets, index, n):
    """Index-pair arrays per snapshot and the 'last received' array per snapshot."""
    pairs, last_received = [], []
    latest = np.full(n, -1, dtype=np.int32)
    for edges in edge_sets:
        i = np.array([index[a] for a, b in edges if a in index and b in index], dtype=np.int32)
        j = np.array([index[b] for a, b in edges if a in index and b in index], dtype=np.int32)
        pairs.append((i, j))
        for target in j:
            latest[target] = len(pairs) - 1
        last_received.append(latest.copy())
    return pairs, last_received


def sample_negatives(allowed, active, linked_names, names, wanted, rng):
    negatives = []
    while len(negatives) < wanted:
        batch = 4 * (wanted - len(negatives))
        left = allowed[rng.integers(0, len(allowed), batch)]
        right = allowed[rng.integers(0, len(allowed), batch)]
        for i, j in zip(left, right):
            if i == j or not (active[i] or active[j]):
                continue
            if (names[i], names[j]) in linked_names:
                continue
            negatives.append((int(i), int(j)))
            if len(negatives) >= wanted:
                break
    return negatives


def training_matrix(model, embeddings, users, index, edge_sets, pairs, last_received, masks, step,
                    rng, with_content, n, max_pos=MAX_POS):
    i, j = pairs[step - 1]
    active = np.asarray(masks[step], dtype=bool)
    state = step_state(i, j, n, last_received[step - 1], step, active)
    delta = [(index[a], index[b]) for a, b in (edge_sets[step] - edge_sets[step - 1])
             if a in index and b in index]
    chosen = rng.choice(len(delta), min(max_pos, len(delta)), replace=False)
    positives = [delta[k] for k in chosen]
    # the candidate universe is the network at step t: the newcomers of this step are inside it,
    # otherwise the "is new" variable would separate positives from negatives by construction
    allowed = np.array(sorted({index[u] for edge in edge_sets[step] for u in edge if u in index}))
    negatives = sample_negatives(allowed, state['active'], edge_sets[step - 1], list(index),
                                 NEG_PER_POS * len(positives), rng)
    pairs_all = positives + negatives
    left = np.array([p[0] for p in pairs_all], dtype=np.int32)
    right = np.array([p[1] for p in pairs_all], dtype=np.int32)
    scores = (content_scores_for(model, embeddings[step], left, right) if with_content
              else np.zeros(len(pairs_all), dtype=np.float32))
    matrix = features_for_pairs(state, scores, left, right)
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    return matrix, labels, len(positives)


def candidate_rate(edge_sets, index, pairs, state, rng, n, step):
    """Empirical share of candidate pairs that become an edge (sampled, for the prior)."""
    names = list(index)
    linked_before = edge_sets[step - 1]        # pairs already linked BEFORE this transition
    new_edges = edge_sets[step] - edge_sets[step - 1]
    allowed = np.array(sorted({index[u] for edge in edge_sets[step] for u in edge if u in index}))
    hits = drawn = 0
    while drawn < PREVALENCE_SAMPLE:
        left = allowed[rng.integers(0, len(allowed), 400_000)]
        right = allowed[rng.integers(0, len(allowed), 400_000)]
        keep = (left != right) & (state['active'][left] | state['active'][right])
        left, right = left[keep], right[keep]
        for a, b in zip(left, right):
            if (names[a], names[b]) in linked_before:
                continue
            drawn += 1
            if (names[a], names[b]) in new_edges:
                hits += 1
    log(f'    [rate] universe {len(allowed)} | active {int(state["active"].sum())} | '
        f'new edges {len(new_edges)} | sampled {drawn} | hits {hits}')
    return hits / max(1, drawn)


# ---------------------------------------------------------------- generation on the synthetic
def generate(model, embeddings, users, masks, present_masks, scaler, classifier, shift, with_content,
             folder, partition_name, rng, steps=N_SNAPSHOTS):
    n = len(users)
    adj = torch.zeros((n, n), dtype=torch.bool, device=DEVICE)
    eye = torch.eye(n, dtype=torch.bool, device=DEVICE)
    last_received = np.full(n, -1, dtype=np.int32)
    for step in range(steps):
        active = np.asarray(masks[step], dtype=bool)
        present = np.asarray(present_masks[step], dtype=bool)
        index_pairs = torch.nonzero(adj, as_tuple=True)
        state = step_state(index_pairs[0].cpu().numpy(), index_pairs[1].cpu().numpy(), n,
                           last_received, step, active)
        hidden = embeddings[step]
        added = 0
        for start in range(0, n, BLOCK):
            stop = min(start + BLOCK, n)
            rows = np.arange(start, stop)
            left = hidden[start:stop].unsqueeze(1).expand(-1, n, -1)
            right = hidden.unsqueeze(0).expand(stop - start, -1, -1)
            with torch.no_grad():
                block = model.head(torch.cat([left, right, left * right], dim=2)).squeeze(2)
            scores = block.cpu().numpy()
            candidate = ((active[start:stop, None] | active[None, :])
                         & present[start:stop, None] & present[None, :]
                         & ~eye[start:stop, :].cpu().numpy()
                         & ~adj[start:stop, :].cpu().numpy())
            row_index, column_index = np.nonzero(candidate)
            if row_index.size == 0:
                continue
            feature = features_for_block(state, scores, start, row_index, column_index)
            if not with_content:
                feature[:, 0] = 0.0
            decision = classifier.decision_function(scaler.transform(feature)) + shift
            drawn = rng.random(decision.size) < sigmoid(decision)
            if drawn.any():
                i = torch.from_numpy((row_index[drawn] + start).astype(np.int64)).to(DEVICE)
                j = torch.from_numpy(column_index[drawn].astype(np.int64)).to(DEVICE)
                adj[i, j] = True
                added += int(drawn.sum())
        touched = adj.any(dim=0).cpu().numpy()
        last_received = np.where(touched, step, last_received).astype(np.int32)
        edges_to_files(adj, users, folder, step, partition_name)
        log(f'    step {step + 1}: active {int(active.sum())} | new {added} | edges so far '
            f'{int(adj.sum().item())}')
    return adj


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', default='generated_snapshots_v2')
    parser.add_argument('--pooling', default='mean_of_posts')
    parser.add_argument('--variant', default='v8')
    parser.add_argument('--content', choices=('real', 'permuted', 'off'), default='real')
    parser.add_argument('--max-pos', type=int, default=MAX_POS)
    parser.add_argument('--steps', type=int, default=N_SNAPSHOTS)
    args = parser.parse_args()
    with_content = args.content != 'off'
    open(LOG, 'w').close()
    rng = np.random.default_rng(SEED)

    log(f'partition {args.partition} | pooling {args.pooling} | variant {args.variant} '
        f'| content {args.content}')
    log('training the tNodeEmbed model on the real dataset')
    encoder, users, index, edge_sets, masks, model, embeddings = load_real()
    n = len(users)
    pairs, last_received = real_arrays(edge_sets, index, n)
    log(f'real users {n} | edges last snapshot {len(edge_sets[-1])}')

    log('building the edge-formation dataset on the real training transitions')
    features, labels, positives = [], [], 0
    for step in TRAIN_STEPS:
        matrix, label, count = training_matrix(model, embeddings, users, index, edge_sets, pairs,
                                               last_received, masks, step, rng, with_content, n,
                                               args.max_pos)
        features.append(matrix)
        labels.append(label)
        positives += count
        log(f'  transition {step}: {count} positives, {len(label) - count} negatives')
    features = np.vstack(features)
    labels = np.concatenate(labels)

    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(max_iter=3000, C=1.0).fit(scaler.transform(features), labels)
    log('fitted coefficients (standardised features):')
    for name, weight in zip(FEATURES, classifier.coef_[0]):
        log(f'  {name:22s} {weight:+.4f}')

    val_matrix, val_labels, val_count = training_matrix(
        model, embeddings, users, index, edge_sets, pairs, last_received, masks, VAL_STEP, rng,
        with_content, n, args.max_pos)
    val_probability = classifier.predict_proba(scaler.transform(val_matrix))[:, 1]
    auc = roc_auc_score(val_labels, val_probability)
    average_precision = average_precision_score(val_labels, val_probability)
    boosted = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
    boosted.fit(features, labels)
    auc_boosted = roc_auc_score(val_labels, boosted.predict_proba(val_matrix)[:, 1])

    i, j = pairs[VAL_STEP - 1]
    state_val = step_state(i, j, n, last_received[VAL_STEP - 1], VAL_STEP,
                           np.asarray(masks[VAL_STEP], dtype=bool))
    rate = candidate_rate(edge_sets, index, pairs, state_val, rng, n, VAL_STEP)
    sample_prevalence = float(labels.mean())
    shift = (float(np.log(rate / (1 - rate)))
             - float(np.log(sample_prevalence / (1 - sample_prevalence))))
    log(f'validation transition {VAL_STEP}: AUC {auc:.4f} | AP {average_precision:.4f} | '
        f'AUC (alberi) {auc_boosted:.4f}')
    log(f'candidate pairs that became an edge (sampled): {rate:.3e} | sampled prevalence '
        f'{sample_prevalence:.4f} | prior shift {shift:+.4f}')

    os.makedirs(RESULTS, exist_ok=True)
    with open(COEFFICIENTS, 'w', encoding='utf-8', newline='') as f:
        f.write('feature,coefficient\n')
        for name, weight in zip(FEATURES, classifier.coef_[0]):
            f.write(f'{name},{weight:.6f}\n')
        f.write(f'intercept,{classifier.intercept_[0]:.6f}\n')
        f.write(f'prior_shift,{shift:.6f}\n')

    synthetic_users, _, synthetic_matrices, synthetic_masks, synthetic_present, _ = \
        synthetic_representations(encoder, os.path.join(BASE, args.partition), args.pooling)
    if args.content == 'permuted':
        order = rng.permutation(len(synthetic_users))
        synthetic_matrices = [matrix[order] for matrix in synthetic_matrices]
        log('  content permuted: same profiles, different owner, one permutation for all the steps')
    synthetic_embeddings = node_embeddings(model, synthetic_matrices)
    folder = os.path.join(BASE, f'{args.partition}_{args.variant}')
    log(f'generating on {args.partition} -> {os.path.basename(folder)} (sampling, with feedback)')
    generate(model, synthetic_embeddings, synthetic_users, synthetic_masks, synthetic_present,
             scaler, classifier, shift, with_content, folder, args.partition, rng,
             steps=args.steps)

    row = {'variant': args.variant, 'partition': args.partition,
           'content': args.content,
           'operating_point': 'sampling, corrected prior', 'val_auc': round(auc, 4),
           'val_ap': round(average_precision, 4), 'val_auc_gbm': round(auc_boosted, 4)}
    row.update(metrics_from_folder(folder))
    existing = []
    if os.path.exists(REPORT):
        with open(REPORT, encoding='utf-8', newline='') as f:
            existing = list(csv.DictReader(f))
    with open(REPORT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing + [row])
    log(f'  -> {row}')


if __name__ == '__main__':
    try:
        main()
    except Exception:                                        # noqa: BLE001
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise
