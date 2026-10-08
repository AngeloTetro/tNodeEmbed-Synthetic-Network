"""V7 -- generation by inference of an edge-formation model fitted on the real dataset.

The critique this answers: turning the model's pair scores into a network with hand-set rules
(how many edges, who may link to whom, how much reciprocity) means the rules -- not the model --
build the network. Here nothing is hand-set: a single edge-formation model is fitted on the real
transitions and then applied to the synthetic dataset.

    features for a candidate pair (u, v) at time step t:
        - content_score : the tNodeEmbed head score (the learned content signal)
        - log_degree_left / log_degree_right : log1p of the degrees at t-1
        - new_left / new_right : whether the endpoint had no edge before t
        - step : normalised time index
    label: did the pair become an edge at t?

The model is a logistic regression (standardised features), fitted on the training transitions,
with the operating point chosen on the validation transition (F1-optimal on all its positives
plus 1M sampled negatives). The test transitions are never touched here.

Generation on the synthetic dataset is then pure inference: at every time step the model scores
the candidate pairs and keeps those above the operating point. Nothing else: two already-present
users may link if the model says so, edges between new users are allowed, reciprocity is not
imposed, the number of edges is not chosen -- it follows from the model.

Output: Esame Angelantonio Fedele Murolo/generated_snapshots[_growth]_v7/,
        results/v7_report.csv, results/v7_coefficients.csv, results/_v7_run.txt

Usage: uv run python src/generate_inference.py
"""

import argparse
import csv
import os
import sys
import traceback

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from apply_edge_rules import edges_to_files, metrics_from_folder
from transfer_to_synthetic import (BASE, BLOCK, DEVICE, N_SNAPSHOTS, SEED, node_embeddings,
                                   prepare_pairs, procrustes_chain, real_representations,
                                   synthetic_representations, train_model)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
REPORT = os.path.join(RESULTS, 'v7_report.csv')
COEFFICIENTS = os.path.join(RESULTS, 'v7_coefficients.csv')
LOG = os.path.join(RESULTS, '_v7_run.txt')
TRAIN_STEPS = (1, 2, 3, 4, 5)
VAL_STEP = 6
NEG_PER_POS = 10
MAX_POS = 30000
VAL_NEGATIVES = 1_000_000
FEATURE_NAMES = ['content_score', 'log_degree_left', 'log_degree_right', 'new_left', 'new_right',
                 'step']
FIELDS = ['variant', 'partition', 'mode', 'operating_point', 'val_auc', 'val_ap', 'nodes',
          'edges', 'density', 'reciprocity', 'mean_out_degree', 'max_out_degree']


def log(text):
    print(text, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def degrees_and_presence(edge_sets, n_users, index):
    """Degree (out) and presence at every snapshot, restricted to the users we can encode."""
    degrees, presence = [], []
    for edges in edge_sets:
        degree = np.zeros(n_users, dtype=np.float32)
        present = np.zeros(n_users, dtype=bool)
        for a, b in edges:
            i, j = index.get(a), index.get(b)
            if i is not None:
                degree[i] += 1
                present[i] = True
            if j is not None:
                degree[j] += 1
                present[j] = True
        degrees.append(degree)
        presence.append(present)
    return degrees, presence


def pair_scores(model, hidden, left, right):
    """The tNodeEmbed head score for a block of pairs."""
    scores = []
    with torch.no_grad():
        for start in range(0, len(left), 100_000):
            i = torch.from_numpy(left[start:start + 100_000]).to(DEVICE)
            j = torch.from_numpy(right[start:start + 100_000]).to(DEVICE)
            pair = torch.cat([hidden[i], hidden[j], hidden[i] * hidden[j]], dim=1)
            scores.append(model.head(pair).squeeze(1).cpu().numpy())
    return np.concatenate(scores)


def pair_features(scores, left, right, degree_prev, present_prev, step):
    return np.column_stack([
        scores,
        np.log1p(degree_prev[left]),
        np.log1p(degree_prev[right]),
        ~present_prev[left],
        ~present_prev[right],
        np.full(len(left), step / (N_SNAPSHOTS - 1), dtype=np.float32),
    ]).astype(np.float32)


def candidates_universe(edge_sets, index, step):
    """Users that exist at `step`: those of the previous network PLUS the newcomers of this step.

    Sampling negatives only among the users already present would leak the answer: in the real
    dataset every new edge involves at least one newcomer, so the "is new" feature would separate
    positives from negatives by construction.
    """
    names = {n for edge in edge_sets[step] for n in edge}
    return np.array([index[u] for u in names if u in index])


def training_pairs(edge_sets, index, rng, step):
    """Positives (new edges at this transition) and negatives from the same candidate universe."""
    delta = [(index[a], index[b]) for a, b in (edge_sets[step] - edge_sets[step - 1])
             if a in index and b in index]
    if len(delta) > MAX_POS:
        picked = rng.choice(len(delta), size=MAX_POS, replace=False)
        positives = [delta[i] for i in picked]
    else:
        positives = delta
    universe = candidates_universe(edge_sets, index, step)
    linked_before = edge_sets[step - 1]
    users = list(index)
    negatives = []
    wanted = NEG_PER_POS * len(positives)
    while len(negatives) < wanted:
        batch = 2 * (wanted - len(negatives))
        left = universe[rng.integers(0, len(universe), batch)]
        right = universe[rng.integers(0, len(universe), batch)]
        for i, j in zip(left, right):
            if i == j:
                continue
            if (users[i], users[j]) in linked_before:
                continue
            negatives.append((int(i), int(j)))
            if len(negatives) >= wanted:
                break
    return positives, negatives


def build_matrix(model, hidden, pairs, degree_prev, present_prev, step):
    left = np.array([p[0] for p in pairs])
    right = np.array([p[1] for p in pairs])
    scores = pair_scores(model, hidden, left, right)
    return pair_features(scores, left, right, degree_prev, present_prev, step)


def f1_threshold(labels, probabilities):
    """Operating point maximising F1 on the validation set."""
    order = np.argsort(-probabilities)
    sorted_labels = labels[order]
    positives = sorted_labels.sum()
    true_positive = np.cumsum(sorted_labels)
    predicted = np.arange(1, len(sorted_labels) + 1)
    f1 = 2 * true_positive / (predicted + positives)
    best = int(np.argmax(f1))
    return float(probabilities[order][best])


def generate_inference(model, scaler, classifier, prior_shift, threshold_mode, mode, rate,
                       embeddings, users, masks, present_masks, folder, partition):
    """Pure inference over the candidate pairs of every time step.

    mode 'bayes': an edge appears when the model says the pair is more likely than not to be an
    edge GIVEN THE TRUE PREVALENCE (prior correction), i.e. corrected logit >= 0.
    mode 'rate' : same ranking, but the operating point is the real rate of new edges per
    candidate pair (measured on the real validation transition) applied to the synthetic
    candidates. Both are data-driven; nothing is hand-set.

    threshold_mode 'global': one threshold per time step, estimated on a sample of the step's
    candidate logits and then applied everywhere (symmetric pairs stay symmetric).
    threshold_mode 'block' : a threshold per block of rows (fast fallback, breaks the symmetry).
    """
    n_users = len(users)
    adj = torch.zeros((n_users, n_users), dtype=torch.bool, device=DEVICE)
    eye = np.eye(n_users, dtype=bool)
    degree_prev = np.zeros(n_users, dtype=np.float32)
    present_prev = np.zeros(n_users, dtype=bool)

    def logit_block(hidden, start, stop, t):
        left = hidden[start:stop].unsqueeze(1).expand(-1, n_users, -1)
        right = hidden.unsqueeze(0).expand(stop - start, -1, -1)
        pair = torch.cat([left, right, left * right], dim=2)
        with torch.no_grad():
            block_scores = model.head(pair).squeeze(2).cpu().numpy()
        left_index = np.repeat(np.arange(start, stop), n_users)
        right_index = np.tile(np.arange(n_users), stop - start)
        features = pair_features(block_scores.reshape(-1), left_index, right_index,
                                 degree_prev, present_prev, t)
        logits = classifier.decision_function(scaler.transform(features)) + prior_shift
        return logits.reshape(stop - start, n_users)

    for t in range(N_SNAPSHOTS):
        hidden = embeddings[t]
        allowed_mask = masks[t][:, None] & present_masks[t][None, :] & ~eye
        fraction = rate
        if mode == 'rate' and threshold_mode == 'global':
            values, rows, columns = [], [], []
            for start in range(0, n_users, BLOCK):
                stop = min(start + BLOCK, n_users)
                logits = logit_block(hidden, start, stop, t)
                allowed = allowed_mask[start:stop, :] & ~adj[start:stop, :].cpu().numpy()
                row_index, column_index = np.nonzero(allowed)
                values.append(logits[row_index, column_index])
                rows.append(row_index + start)
                columns.append(column_index)
                del logits, allowed
            values = np.concatenate(values)
            step_threshold = float(np.quantile(values, 1 - fraction)) if values.size else 0.0
            rows = np.concatenate(rows)[values >= step_threshold]
            columns = np.concatenate(columns)[values >= step_threshold]
            log(f'    time step {t + 1}: candidate pairs {values.size}, threshold '
                f'{step_threshold:+.4f}, accepted {rows.size}')
            if rows.size:
                block = torch.zeros_like(adj)
                block[torch.from_numpy(rows).to(DEVICE), torch.from_numpy(columns).to(DEVICE)] = True
                adj = adj | block
            del values
        else:
            for start in range(0, n_users, BLOCK):
                stop = min(start + BLOCK, n_users)
                logits = logit_block(hidden, start, stop, t)
                allowed = allowed_mask[start:stop, :] & ~adj[start:stop, :].cpu().numpy()
                if mode == 'bayes':
                    accepted = logits >= 0
                else:
                    block_values = logits[allowed]
                    if block_values.size == 0:
                        accepted = np.zeros_like(allowed)
                    else:
                        accepted = logits >= float(np.quantile(block_values, 1 - fraction))
                block = torch.from_numpy(allowed & accepted).to(DEVICE)
                adj[start:stop, :] = adj[start:stop, :] | block
        degree_prev = adj.sum(dim=1).cpu().numpy().astype(np.float32)
        present_prev = (adj.any(dim=1) | adj.any(dim=0)).cpu().numpy()
        edges_to_files(adj, users, folder, t, partition)
        if mode != 'rate' or threshold_mode != 'global':
            log(f'    time step {t + 1}: edges so far {int(adj.sum().item())}')
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default='minilm')
    parser.add_argument('--threshold', default='block', choices=['block', 'global'])
    parser.add_argument('--variant', default='v7b')
    parser.add_argument('--pooling', default='concat', choices=('concat', 'mean_of_posts'))
    parser.add_argument('--partitions', default='generated_snapshots_v2,generated_snapshots_growth_v2',
                        help='comma separated list of partition folders')
    args = parser.parse_args()
    open(LOG, 'w').close()
    rng = np.random.default_rng(SEED)
    log(f'encoder {args.encoder} | threshold mode {args.threshold} | variant {args.variant} '
        f'| pooling {args.pooling}')
    log('training the tNodeEmbed model on the real dataset')
    encoder, users, index, edge_sets, matrices, masks = real_representations(args.encoder,
                                                                            'period',
                                                                            args.pooling)
    aligned = procrustes_chain(matrices, masks)
    build = prepare_pairs(edge_sets, users, index, matrices, masks, 'matched')
    model = train_model(aligned, build, index, matrices[0].shape[1])
    embeddings = node_embeddings(model, matrices)
    degrees, presence = degrees_and_presence(edge_sets, len(users), index)

    log('building the edge-formation dataset on the real training transitions')
    features, labels = [], []
    for step in TRAIN_STEPS:
        positives, negatives = training_pairs(edge_sets, index, rng, step)
        pairs = positives + negatives
        matrix = build_matrix(model, embeddings[step], pairs, degrees[step - 1],
                              presence[step - 1], step)
        features.append(matrix)
        labels.append(np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))]))
        log(f'  transition {step}: {len(positives)} positives, {len(negatives)} negatives')
    features = np.vstack(features)
    labels = np.concatenate(labels)

    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(features), labels)
    log('fitted coefficients (standardised features):')
    for name, weight in zip(FEATURE_NAMES, classifier.coef_[0]):
        log(f'  {name:20s} {weight:+.4f}')

    val_positives, _ = training_pairs(edge_sets, index, rng, VAL_STEP)
    universe = candidates_universe(edge_sets, index, VAL_STEP)
    left = universe[rng.integers(0, len(universe), VAL_NEGATIVES)]
    right = universe[rng.integers(0, len(universe), VAL_NEGATIVES)]
    real_names = list(index)
    linked_before_val = edge_sets[VAL_STEP - 1]
    val_negatives = [(int(i), int(j)) for i, j in zip(left, right)
                     if i != j and (real_names[int(i)], real_names[int(j)]) not in linked_before_val]
    val_pairs = val_positives + val_negatives
    val_matrix = build_matrix(model, embeddings[VAL_STEP], val_pairs, degrees[VAL_STEP - 1],
                              presence[VAL_STEP - 1], VAL_STEP)
    val_labels = np.concatenate([np.ones(len(val_positives)), np.zeros(len(val_negatives))])
    val_probability = classifier.predict_proba(scaler.transform(val_matrix))[:, 1]
    auc = roc_auc_score(val_labels, val_probability)
    average_precision = average_precision_score(val_labels, val_probability)
    f1_point = f1_threshold(val_labels, val_probability)

    sample_prevalence = float(labels.mean())
    n_present_val = len(universe)
    n_candidates_val = n_present_val * (n_present_val - 1) - len(linked_before_val)
    real_rate = len(edge_sets[VAL_STEP] - edge_sets[VAL_STEP - 1]) / n_candidates_val
    prior_shift = (float(np.log(real_rate / (1 - real_rate)))
                   - float(np.log(sample_prevalence / (1 - sample_prevalence))))
    log(f'validation transition {VAL_STEP}: AUC {auc:.4f} | AP {average_precision:.4f} | '
        f'F1-optimal probability {f1_point:.6f} (sampled base rate)')
    log(f'candidate pairs at validation {n_candidates_val} | real new edges per candidate pair '
        f'{real_rate:.3e} | sample base rate {sample_prevalence:.4f} | prior shift '
        f'{prior_shift:+.4f}')

    with open(COEFFICIENTS, 'w', encoding='utf-8', newline='') as f:
        f.write('feature,coefficient\n')
        for name, weight in zip(FEATURE_NAMES, classifier.coef_[0]):
            f.write(f'{name},{weight:.6f}\n')
        f.write(f'intercept,{classifier.intercept_[0]:.6f}\n')
        f.write(f'prior_shift,{prior_shift:.6f}\n')

    rows = []
    partition_names = [name.strip() for name in args.partitions.split(',')]
    for partition in partition_names:
        (synthetic_users, _, synthetic_matrices, synthetic_masks,
         synthetic_present, _) = synthetic_representations(encoder, os.path.join(BASE, partition),
                                                           args.pooling)
        synthetic_embeddings = node_embeddings(model, synthetic_matrices)
        for variant, mode in ((args.variant, 'rate'),):
            folder = os.path.join(BASE, f'{partition}_{variant}')
            log(f'\n[{variant}/{partition}] generation by inference, mode {mode}, '
                f'threshold {args.threshold}')
            adj = generate_inference(model, scaler, classifier, prior_shift, args.threshold, mode,
                                     real_rate, synthetic_embeddings, synthetic_users,
                                     synthetic_masks, synthetic_present, folder, partition)
            operating = (f'real rate {real_rate:.2e} per candidate pair, '
                         f'threshold {args.threshold}')
            row = {'variant': variant, 'partition': partition, 'mode': mode,
                   'operating_point': operating, 'val_auc': round(auc, 4),
                   'val_ap': round(average_precision, 4)}
            row.update(metrics_from_folder(folder))
            rows.append(row)
            log(f'  -> {row}')
            del adj
            torch.cuda.empty_cache()

    with open(REPORT, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
    log(f'appended to {REPORT}')


if __name__ == '__main__':
    try:
        main()
    except Exception:                                        # noqa: BLE001
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise
