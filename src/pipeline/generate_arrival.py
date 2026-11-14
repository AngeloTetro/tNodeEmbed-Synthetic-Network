"""Arrival events -- the unit of edge formation is the event, not the pair.

Diagnosis: a pairwise approach reproduces density and volume but gets the relational properties wrong
(reciprocity 0.004 against 0.418, transitivity 0.003 against 0.034, maximum degree 102 against 1,290)
because it decides every pair independently and trains on uniform negatives. In the real data those
three gaps are a single phenomenon: the arrival of a new user. Every new edge touches an entrant,
40-44% arrive with their own inverse in the same transition, and the degree tail exists because
arrivals choose the few nodes that are already well connected.

The generator estimates on the real data, for every arrival event (a node entering the network in that
transition):
    1. how many new edges it carries, with the direction. The count is fitted **rescaled to the
       population** (k * reference_population / transition_population), because k grows with the size
       of the network: it is the same device used for degree percentiles.
    2. where those edges go: a choice model over the candidates -- degree, percentile, "is new",
       activity, tNodeEmbed content score, direction, and **co-occurrence** ("the counterpart is close
       to another counterpart of the same event", that is whether the event's edges cluster or avoid
       each other). The weights are the scientific result.
    3. reciprocation: with what probability an incoming edge picks back a partner already chosen among
       the outgoing ones. In the real data the k edges of an event already include the reciprocated
       pairs, so reciprocity is not added afterwards: it is distributed inside the count.

Generation imposes no counts: the edges of the event are sampled one by one on the choice score
(Gumbel), rescaling k to the current synthetic population, then the state is updated.

Content modes (`--content`), synthetic side only; the model is always fitted on the real data:
    real      authentic text profiles;
    permuted  the same profiles reassigned to other users (one permutation for every step);
    off       text channel zeroed.

Usage:
    uv run python -m src.pipeline.generate_arrival --partition synthetic_progressive_posts \
        --output synthetic_progressive
    uv run python -m src.pipeline.generate_arrival --partition synthetic_all_present_posts \
        --output synthetic_all_present
Output: <output or partition_variant>/, results/network_report.csv,
        results/coefficients_<output>.csv, results/_run.txt
"""

import argparse
import csv
import os
import shutil
import traceback

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.paths import ROOT, TABLES, BASE

from src.lib.graph_io import metrics_from_folder
from src.lib.transfer_to_synthetic import (DEVICE, N_SNAPSHOTS, SEED, node_embeddings,
                                   prepare_pairs, procrustes_chain, real_representations,
                                   synthetic_representations, train_model)

REPORT = os.path.join(TABLES, 'network_report.csv')
LOG = os.path.join(TABLES, '_run.txt')


def coefficients_path(name):
    """One coefficient file per run, named after the generated folder."""
    return os.path.join(TABLES, f'coefficients_{name}.csv')


TRAIN_STEPS = (1, 2, 3, 4, 5)
VAL_STEP = 6
NEG_PER_POS = 10
MAX_EVENTS = 400
MAX_EVENT = 5000
MAX_COUNT = 1_000_000
SMALL = 1e-3
FEATURES = ['content_score', 'partner_degree_log', 'partner_percentile', 'partner_new',
            'partner_active', 'outgoing', 'partner_co_occurrence']
FIELDS = ['variant', 'partition', 'content', 'operating_point', 'val_auc', 'events',
          'edges_per_event', 'reverse_share', 'nodes', 'edges', 'density', 'reciprocity',
          'mean_out_degree', 'max_out_degree']


def log(text):
    print(text, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def edges_to_files(adj, users, folder, t, source_partition):
    """Write snapshot_t/social_network.edg and copy the partition'sposts for that step."""
    directory = os.path.join(folder, f'snapshot_{t + 1}')
    os.makedirs(directory, exist_ok=True)
    for name in ('posts_current_snapshot.tsv', 'posts_incremental.tsv'):
        origin = os.path.join(BASE, source_partition, f'snapshot_{t + 1}', name)
        destination = os.path.join(directory, name)
        if not os.path.exists(destination) and os.path.exists(origin):
            shutil.copyfile(origin, destination)
    rows, columns = torch.nonzero(adj, as_tuple=True)
    with open(os.path.join(directory, 'social_network.edg'), 'w', encoding='utf-8',
              newline='') as f:
        for source, target in zip(rows.tolist(), columns.tolist()):
            f.write(f'{users[source]}\t{users[target]}\r\n')


def arrivals(edge_sets, index, step):
    """Arrival events of transition `step`: every new edge belongs to the event of its new endpoint
    (if both are new, the source wins), with the direction relative to that endpoint.

    Edges are iterated in deterministic order (`sorted`): the iteration order of a set of strings
    depends on the process hash and would change the random sequence downstream, making repetitions
    differ from each other.
    """
    previous, current = edge_sets[step - 1], edge_sets[step]
    before = {u for edge in previous for u in edge}
    events = {}
    for source, target in sorted(current - previous):
        if source in index and source not in before:
            owner, direction, partner = source, 1.0, target
        elif target in index and target not in before:
            owner, direction, partner = target, 0.0, source
        else:
            continue
        events.setdefault(owner, []).append((direction, partner))
    return events


def population(edge_sets, step):
    return len({u for edge in edge_sets[step] for u in edge})


def fit_counts(edge_sets, index, steps, exponent):
    """Table of k rescaled to the population: k / P^beta, with beta and the unit fitted on the real.

    k grows with the population of the network faster than linearly: the exponent beta is the log-log
    slope of mean k against population. The table holds the normalised value, multiplied by a unit that
    preserves its resolution (median at 1000); generation rescales it back.
    """
    events_by_step = {step: arrivals(edge_sets, index, step) for step in steps}
    scales = {step: population(edge_sets, step - 1) ** exponent for step in steps}
    normalised = [len(edges) / scales[step]
                  for step in steps for edges in events_by_step[step].values()]
    unit = 1000.0 / max(float(np.median(normalised)), 1e-12)
    tables, shares, sizes = {}, {}, {}
    for step in steps:
        counts, outgoing = [], []
        for edges in events_by_step[step].values():
            counts.append(min(int(round(len(edges) / scales[step] * unit)), MAX_COUNT))
            outgoing.append(sum(1 for direction, _ in edges if direction > 0.5) / len(edges))
        values = np.asarray(counts or [1], dtype=np.int64)
        tables[step] = np.bincount(values, minlength=int(values.max()) + 2).astype(np.float64) + SMALL
        shares[step] = float(np.mean(outgoing)) if outgoing else 0.5
        sizes[step] = len(counts)
    return tables, shares, sizes, unit


def population_exponent(edge_sets, index, steps):
    """Esponente beta di k ~ P^beta, fittato sulle transizioni di training (pendenza log-log)."""
    sizes, means = [], []
    for step in steps:
        events = arrivals(edge_sets, index, step)
        if not events:
            continue
        sizes.append(population(edge_sets, step - 1))
        means.append(float(np.mean([len(edges) for edges in events.values()])))
    if len(sizes) < 2 or min(means) <= 0:
        return 1.0
    return float(np.polyfit(np.log(sizes), np.log(means), 1)[0])


def reciprocity_table(edge_sets, index, steps):
    """P(reverse edge in the same transition | new edge), by direction and degree class."""
    boxes = {}
    for step in steps:
        previous = edge_sets[step - 1]
        before = {u for edge in previous for u in edge}
        new_edges = edge_sets[step] - previous
        degree = {}
        for edge in previous:
            for user in edge:
                degree[user] = degree.get(user, 0) + 1
        for source, target in new_edges:
            reverse = (target, source) in new_edges
            for owner, direction, partner in ((source, 1.0, target), (target, 0.0, source)):
                if owner in index and owner not in before:
                    value = degree.get(partner, 0)
                    bucket = 0 if value == 0 else (1 if value < 50 else 2)
                    box = boxes.setdefault((direction, bucket), [0, 0])
                    box[0] += 1
                    if reverse:
                        box[1] += 1
    return {key: (hits + 1.0) / (total + 2.0) for key, (total, hits) in boxes.items()}


def state_arrays(edge_sets, index, masks, users, step):
    """What the fit computes from the state at the previous step."""
    previous = edge_sets[step - 1]
    before = {u for edge in previous for u in edge if u in index}
    degree = {}
    for edge in previous:
        for user in edge:
            degree[user] = degree.get(user, 0) + 1
    present = np.array(sorted(index[u] for u in before), dtype=np.int64)
    n = len(users)
    degree_log = np.zeros(n, dtype=np.float32)
    percentile = np.zeros(n, dtype=np.float32)
    is_new = np.ones(n, dtype=np.float32)
    active = masks[step].astype(np.float32) if masks is not None else np.zeros(n, dtype=np.float32)
    if present.size:
        values = np.array([degree.get(users[i], 0) for i in present], dtype=np.float32)
        degree_log[present] = np.log1p(values)
        if present.size > 1:
            percentile[present] = values.argsort().argsort() / (present.size - 1)
        is_new[present] = 0.0
    return present, degree_log, percentile, is_new, active


def choice_features(scores, degree_log, percentile, is_new, active, targets, directions, adjacent):
    return np.column_stack([
        scores,
        degree_log[targets],
        percentile[targets],
        is_new[targets],
        active[targets],
        directions,
        adjacent,
    ]).astype(np.float32)


def content_scores(model, embeddings, owners, targets, with_content):
    if not with_content or len(owners) == 0:
        return np.zeros(len(owners), dtype=np.float32)
    out = []
    with torch.no_grad():
        for start in range(0, len(owners), 200_000):
            i = torch.from_numpy(np.asarray(owners[start:start + 200_000], dtype=np.int64)).to(DEVICE)
            j = torch.from_numpy(np.asarray(targets[start:start + 200_000], dtype=np.int64)).to(DEVICE)
            pair = torch.cat([embeddings[i], embeddings[j], embeddings[i] * embeddings[j]], dim=1)
            out.append(model.head(pair).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def neighbours_of(edge_sets, step, index):
    """Neighbourhood (both directions) in the network of the transition, indexed as `index`.

    Iteration is ordered here too, for the same reason as in `arrivals`.
    """
    out = {}
    for a, b in sorted(edge_sets[step]):
        i, j = index.get(a), index.get(b)
        if i is None or j is None:
            continue
        out.setdefault(i, set()).add(j)
        out.setdefault(j, set()).add(i)
    return out


def closure_value(partners, neighbours_of_candidate, mode):
    """Triadic closure inside the event: 'binary' (at least one) or 'count' (the fraction)."""
    if not partners or not neighbours_of_candidate:
        return 0.0
    shared = len(partners & neighbours_of_candidate)
    if mode == 'binary':
        return 1.0 if shared else 0.0
    return shared / len(partners)


def build_choice_dataset(model, embeddings, users, index, edge_sets, masks, rng, with_content,
                         steps, stratified=False, closure='binary'):
    """Positives = the real edges of the events; negatives = candidates of the same event.

    Co-occurrence is 1 if the counterpart is close to another counterpart of the same event: it is the
    term that says whether the edges of an event cluster around the same neighbourhoods or avoid them.

    With `stratified=True` the negatives become a mixture: 30% asymmetric inversions, that is the pair
    reversed from a positive but only if that reverse edge does not really exist, 30% hard on degree
    (candidates from the most connected decile of the pool) and 40% uniform as in the documented
    configuration. The semantic part of hard negatives is absent: choosing the counterpart most similar
    in text would cost one model evaluation per candidate and the text channel does not carry the tie.

    `closure` says how triadic closure inside the event is measured: 'binary' (the documented variable,
    the candidate is close to at least one other counterpart of the event) or 'count' (the fraction of
    the other counterparts the candidate is close to). The same function must hold in fitting and in
    generation: changing it on one side only would make the coefficient meaningless.
    """
    features, labels = [], []
    for step in steps:
        events = arrivals(edge_sets, index, step)
        present, degree_log, percentile, is_new, active = state_arrays(edge_sets, index, masks,
                                                                      users, step)
        if present.size == 0 or not events:
            continue
        # Negatives must come from the same pool the generation samples from (the users already
        # present plus the newcomers of this step): otherwise "the partner is new" separates
        # positives from negatives by construction and the fitted weight is meaningless.
        newcomers = np.array(sorted(index[owner] for owner in events if owner in index),
                             dtype=np.int64)
        pool = np.union1d(present, newcomers)
        if pool.size == 0:
            continue
        values = degree_log[pool]
        percentile[pool] = (values.argsort().argsort() / max(1, pool.size - 1)).astype(np.float32)
        neighbours = neighbours_of(edge_sets, step, index)
        directed = {(index[a], index[b]) for a, b in edge_sets[step] if a in index and b in index}
        hard_degree = pool[np.argsort(-degree_log[pool])[:max(1, pool.size // 10)]]
        chosen = list(events)
        if len(chosen) > MAX_EVENTS:
            chosen = list(rng.choice(chosen, size=MAX_EVENTS, replace=False))
        owners, targets, directions, tags, adjacent = [], [], [], [], []
        for owner in chosen:
            edges = events[owner]
            partners = [index[partner] for _, partner in edges if partner in index]
            if not partners:
                continue
            partner_set = {index[partner] for _, partner in edges if partner in index}
            inversions = [(index[partner], 1.0 - direction) for direction, partner in edges
                          if partner in index
                          and (index[partner], index[owner]) not in directed]
            for direction, partner in edges:
                if partner not in index:
                    continue
                owners.append(index[owner])
                targets.append(index[partner])
                directions.append(direction)
                tags.append(1.0)
                others = partner_set - {index[partner]}
                adjacent.append(closure_value(others, neighbours.get(index[partner], set()), closure))
            for _ in range(NEG_PER_POS * len(partners)):
                # The uniform path draws no extra numbers: with `--negatives uniform` the random
                # sequence (and therefore the generated network) stays the documented one.
                if stratified and inversions and rng.random() < 0.30:
                    candidate, direction = inversions[int(rng.integers(0, len(inversions)))]
                elif stratified and rng.random() < 3.0 / 7.0:
                    candidate = int(hard_degree[int(rng.integers(0, hard_degree.size))])
                    direction = 1.0 if rng.random() < 0.5 else 0.0
                    if candidate in partner_set:
                        candidate = int(pool[rng.integers(0, pool.size)])
                else:
                    candidate = int(pool[rng.integers(0, pool.size)])
                    direction = 1.0 if rng.random() < 0.5 else 0.0
                owners.append(index[owner])
                targets.append(candidate)
                directions.append(direction)
                tags.append(0.0)
                adjacent.append(closure_value(partner_set, neighbours.get(candidate, set()), closure))
        if not owners:
            continue
        scores = content_scores(model, embeddings[step], owners, targets, with_content)
        features.append(choice_features(scores, degree_log, percentile, is_new, active,
                                        np.asarray(targets), np.asarray(directions, dtype=np.float32),
                                        np.asarray(adjacent, dtype=np.float32)))
        labels.append(np.asarray(tags, dtype=np.float32))
        log(f'    step {step}: {len(owners)} rows ({int(np.sum(tags))} positives)')
    return np.vstack(features), np.concatenate(labels)


def sample_counts(table, share, population_now, exponent, unit, rng):
    """Normalised k sampled, rescaled to the current population, then the outgoing share."""
    probability = table / table.sum()
    scaled = int(rng.choice(table.size, p=probability))
    total = int(round(scaled / unit * population_now ** exponent))
    if total <= 0:
        return 0, 0
    return total, int(rng.binomial(total, float(np.clip(share, 0.0, 1.0))))


def head_scores(model, hidden, owner, candidates):
    with torch.no_grad():
        left = hidden[torch.from_numpy(np.full(candidates.size, owner, dtype=np.int64)).to(DEVICE)]
        right = hidden[torch.from_numpy(candidates.astype(np.int64)).to(DEVICE)]
        pair = torch.cat([left, right, left * right], dim=1)
        return model.head(pair).squeeze(1).cpu().numpy()


def generate(model, embeddings, users, masks, present_masks, scaler, classifier, tables, shares,
             reverse_rate, exponent, unit, with_content, folder, partition_name, rng,
             closure='binary'):
    """Generates by arrival events, with the state recomputed at every step."""
    n = len(users)
    adj = torch.zeros((n, n), dtype=torch.bool, device=DEVICE)
    previous_present = np.zeros(n, dtype=bool)
    events_total, edges_total, reversed_total = 0, 0, 0
    train_steps = sorted(tables)
    for step in range(N_SNAPSHOTS):
        active = np.asarray(masks[step], dtype=bool)
        present = np.asarray(present_masks[step], dtype=bool)
        candidates = np.nonzero(present)[0]
        newcomers = np.nonzero(present & ~previous_present)[0]
        table_key = train_steps[min(step, len(train_steps) - 1)]
        population_now = int(previous_present.sum()) or int(present.sum())
        if candidates.size and newcomers.size:
            degrees = (adj.sum(dim=0) + adj.sum(dim=1)).cpu().numpy()
            values = degrees[candidates].astype(np.float32)
            percentile = np.zeros(n, dtype=np.float32)
            if candidates.size > 1:
                percentile[candidates] = values.argsort().argsort() / (candidates.size - 1)
            degree_log = np.log1p(degrees).astype(np.float32)
            is_new = (~previous_present).astype(np.float32)
            active_flag = active.astype(np.float32)
            for owner in newcomers:
                total, outgoing = sample_counts(tables[table_key], shares[table_key], population_now,
                                                exponent, unit, rng)
                if total <= 0:
                    continue
                if with_content:
                    content = head_scores(model, embeddings[step], int(owner), candidates)
                else:
                    content = np.zeros(candidates.size, dtype=np.float32)
                out_linked = adj[owner].cpu().numpy()[candidates].copy()
                in_linked = adj[:, owner].cpu().numpy()[candidates].copy()
                near_chosen = np.zeros(candidates.size, dtype=np.float32)
                chosen_all = set()
                out_linked[candidates == owner] = True
                in_linked[candidates == owner] = True
                chosen = []
                for position in range(min(total, MAX_EVENT)):
                    direction = 1.0 if position < outgoing else 0.0
                    if direction < 0.5 and chosen and rng.random() < reverse_rate:
                        partner = int(chosen[rng.integers(0, len(chosen))])
                        place = int(np.searchsorted(candidates, partner))
                        if in_linked[place]:
                            continue
                    else:
                        allowed = ~(out_linked if direction > 0.5 else in_linked)
                        if not allowed.any():
                            break
                        closure_now = (near_chosen if closure == 'binary'
                                       else near_chosen / max(1, len(chosen_all)))
                        matrix = choice_features(
                            content, degree_log, percentile, is_new, active_flag, candidates,
                            np.full(candidates.size, direction, dtype=np.float32), closure_now)
                        weight = classifier.decision_function(scaler.transform(matrix))
                        key = np.where(allowed, weight + rng.gumbel(size=weight.size), -np.inf)
                        partner = int(candidates[int(np.argmax(key))])
                    source, target = (int(owner), partner) if direction > 0.5 else (partner, int(owner))
                    if not bool(adj[source, target]):
                        adj[source, target] = True
                        edges_total += 1
                        if direction < 0.5:
                            reversed_total += 1
                    if direction > 0.5:
                        out_linked[candidates == partner] = True
                        chosen.append(partner)
                    else:
                        in_linked[candidates == partner] = True
                    if partner not in chosen_all:
                        chosen_all.add(partner)
                        picks = np.nonzero((adj[partner].cpu().numpy()[candidates])
                                           | (adj[:, partner].cpu().numpy()[candidates]))[0]
                        if closure == 'binary':
                            near_chosen[picks] = 1.0
                        else:
                            near_chosen[picks] += 1.0
                events_total += 1
        edges_to_files(adj, users, folder, step, partition_name)
        previous_present = present
        log(f'    step {step + 1}: new {int(newcomers.size)} | population {population_now} '
            f'| edges so far {int(adj.sum().item())}')
    return events_total, edges_total, reversed_total


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', default='synthetic_progressive_posts')
    parser.add_argument('--pooling', default='mean_of_posts')
    parser.add_argument('--variant', default='network')
    parser.add_argument('--output', default=None,
                        help='name of the destination folder (default: <partition>_<variant>)')
    parser.add_argument('--content', choices=('real', 'permuted', 'off'), default='real')
    parser.add_argument('--exponent', type=float, default=1.0,
                        help='exponent of the rescaling of k on the population: 1 = linear, '
                             'the value that brings the generated mean degree onto the real one')
    parser.add_argument('--negatives', choices=('uniform', 'stratified'), default='uniform',
                        help='uniform: the documented configuration; stratified: 30/30/40 mixture')
    parser.add_argument('--closure', choices=('binary', 'count'), default='binary',
                        help='triadic closure inside the event: binary (documented) or the fraction '
                             'of the other counterparts')
    args = parser.parse_args()
    with_content = args.content != 'off'
    stratified = args.negatives == 'stratified'
    closure = args.closure
    open(LOG, 'w').close()
    rng = np.random.default_rng(SEED)

    log(f'partition {args.partition} | pooling {args.pooling} | variant {args.variant} '
        f'| content {args.content} | negatives {args.negatives} | closure {args.closure}')
    log('training the tNodeEmbed model on the real dataset')
    encoder, users, index, edge_sets, matrices, masks = real_representations('minilm', 'period',
                                                                            args.pooling)
    aligned = procrustes_chain(matrices, masks)
    model = train_model(aligned, prepare_pairs(edge_sets, users, index, masks, 'matched'),
                        index, matrices[0].shape[1])
    embeddings = node_embeddings(model, matrices)
    log(f'real users {len(users)} | edges last snapshot {len(edge_sets[-1])}')

    fitted = population_exponent(edge_sets, index, TRAIN_STEPS)
    exponent = args.exponent
    log(f'rescaling exponent of k: {exponent:.3f} (log-log slope estimated on the real: '
        f'{fitted:.3f}). The synthetic dataset has 10,000 users against the 25,714 of the real one '
        f'and its last steps inherit the table of the last training transition: without rescaling '
        f'the mean degree would come out at 80 against 32, with exponent 1 it lands in the right '
        f'order of magnitude.')
    tables, shares, sizes, unit = fit_counts(edge_sets, index, TRAIN_STEPS, exponent)
    log(f'scale unit of the normalised table: {unit:.1f}')
    for step in sorted(tables):
        table = tables[step] / tables[step].sum()
        expected = float(np.sum(np.arange(table.size) * table))
        log(f'  transition {step}: events {sizes[step]} | expected normalised k {expected:.1f} '
            f'| outgoing share {shares[step]:.3f}')
    cells = reciprocity_table(edge_sets, index, TRAIN_STEPS)
    for (direction, bucket), value in sorted(cells.items()):
        log(f'    reciprocation: outgoing {int(direction)} | degree {bucket}: {value:.3f}')
    reverse_rate = float(np.mean([value for (direction, _), value in cells.items()
                                  if direction > 0.5]))
    log(f'  share of reciprocated pairs used in generation: {reverse_rate:.3f}')

    log('fitted coefficients of the target-choice model (standardised features):')
    features, labels = build_choice_dataset(model, embeddings, users, index, edge_sets, masks, rng,
                                           with_content, TRAIN_STEPS, stratified, closure)
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(max_iter=2000).fit(scaler.transform(features), labels)
    for name, weight in zip(FEATURES, classifier.coef_[0]):
        log(f'  {name:24s} {weight:+.4f}')

    val_features, val_labels = build_choice_dataset(model, embeddings, users, index, edge_sets,
                                                    masks, rng, with_content, (VAL_STEP,), stratified,
                                                    closure)
    auc = roc_auc_score(val_labels,
                        classifier.predict_proba(scaler.transform(val_features))[:, 1])
    log(f'validation transition {VAL_STEP}: AUC {auc:.4f} (fit diagnostic; the quality of the '
        f'network is judged on its statistics, not here)')

    name = args.output or f'{args.partition}_{args.variant}'
    os.makedirs(TABLES, exist_ok=True)
    with open(coefficients_path(name), 'w', encoding='utf-8',
              newline='') as f:
        f.write('feature,coefficient\n')
        for feature, weight in zip(FEATURES, classifier.coef_[0]):
            f.write(f'{feature},{weight:.6f}\n')
        f.write(f'val_auc,{auc:.6f}\n')

    synthetic_users, _, synthetic_matrices, synthetic_masks, synthetic_present, _ = \
        synthetic_representations(encoder, os.path.join(BASE, args.partition), args.pooling)
    if args.content == 'permuted':
        order = rng.permutation(len(synthetic_users))
        synthetic_matrices = [matrix[order] for matrix in synthetic_matrices]
        log('  content permuted: same profiles, different owner, one permutation for all the steps')
    synthetic_embeddings = node_embeddings(model, synthetic_matrices)
    folder = os.path.join(BASE, name)
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, 'tnodeembed_embeddings.npz'),
                        users=np.array(synthetic_users),
                        **{f'step_{t + 1}': synthetic_embeddings[t].cpu().numpy()
                           for t in range(N_SNAPSHOTS)})
    log(f'generating on {args.partition} -> {os.path.basename(folder)} (arrival events)')
    events, edges, reversed_edges = generate(
        model, synthetic_embeddings, synthetic_users, synthetic_masks, synthetic_present, scaler,
        classifier, tables, shares, reverse_rate, exponent, unit, with_content, folder,
        args.partition, rng, closure)

    row = {'variant': args.variant, 'partition': args.partition, 'content': args.content,
           'operating_point': 'arrival events, sampled jointly', 'val_auc': round(float(auc), 4),
           'events': events, 'edges_per_event': round(edges / max(1, events), 3),
           'reverse_share': round(reversed_edges / max(1, edges), 4)}
    row.update(metrics_from_folder(folder))
    existing = []
    if os.path.exists(REPORT):
        with open(REPORT, encoding='utf-8', newline='') as f:
            existing = list(csv.DictReader(f))
    key = (args.variant, args.partition, args.content)
    existing = [r for r in existing
                if (r.get('variant'), r.get('partition'), r.get('content')) != key]
    with open(REPORT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing + [row])
    log(f'  -> {row}')


if __name__ == '__main__':
    try:
        run()
    except Exception:                                            # noqa: BLE001
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise
