"""Alternative edge-selection rules for the synthetic network (lever 3).

The model is trained once on the real dataset (same seeds as transfer_to_synthetic.py, so the
same model); then the synthetic network is generated for each rule:

  v1 threshold    (partition "all")    : an edge is created when the score passes a threshold
                                         calibrated so that the mean number of new edges per
                                         active user matches the reference rule (~1.2)
  v2 real_k       (partition "all")    : each user's total out-degree is drawn from the real
                                         out-degree distribution (mean ~32, heavy tail)
  v3 reciprocity  (partition "all")    : reference rule, then reverse edges are added until the
                                         reciprocity reaches the real value (~0.42)
  v4 growth       (partition "growth") : an edge can only involve a user entering the network in
                                         that time step (no old->old edges), and targets are
                                         drawn with probability proportional to their previous
                                         degree (the accretion mechanism measured on the real)
  v5 combined     (partition "growth") : v2 + v3 + v4

Selection is vectorised (one score matrix per time step, no per-user Python loop).
Each variant writes social_network.edg snapshot by snapshot inside
Esame Angelantonio Fedele Murolo/generated_snapshots_rule_<variant>/ and appends one row to
results/edge_rules_report.csv.

Usage:
  uv run python src/apply_edge_rules.py --variants v4,v5
  uv run python src/apply_edge_rules.py --rebuild      # rebuild the report from the folders
"""

import argparse
import csv
import os
import shutil

import numpy as np
import torch

from transfer_to_synthetic import (BASE, BLOCK, DEVICE, LAMBDA, N_SNAPSHOTS, SEED,
                                   node_embeddings, prepare_pairs, procrustes_chain,
                                   real_representations, synthetic_representations, train_model)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
REPORT = os.path.join(RESULTS, 'edge_rules_report.csv')
RECIPROCITY_TARGET = 0.42
EDGES_PER_USER = 1.2
VARIANTS = {
    'v1': ('generated_snapshots', {'threshold': True}),
    'v2': ('generated_snapshots', {'real_k': True}),
    'v3': ('generated_snapshots', {'reciprocity': True}),
    'v4': ('generated_snapshots_growth', {'growth': True}),
    'v5': ('generated_snapshots_growth', {'real_k': True, 'reciprocity': True, 'growth': True}),
}
FIELDS = ['variant', 'partition', 'rule', 'nodes', 'edges', 'density', 'reciprocity',
          'mean_out_degree', 'max_out_degree', 'seconds']


def real_out_degrees():
    """Out-degree distribution of the real users in the last snapshot (for v2)."""
    folder = os.path.join(BASE, 'real_snapshots')
    last = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))[-1]
    degree = {}
    with open(os.path.join(folder, last, 'social_network.edg'), encoding='utf-8-sig',
              newline='') as f:
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                source = parts[0].strip()
                degree[source] = degree.get(source, 0) + 1
    return np.array([v for v in degree.values() if v > 0], dtype=float)


def scores_for_step(model, hidden):
    """Model score for every ordered pair, as an (n_users, n_users) tensor."""
    n_users = hidden.shape[0]
    rows = []
    for start in range(0, n_users, BLOCK):
        stop = min(start + BLOCK, n_users)
        left = hidden[start:stop].unsqueeze(1).expand(-1, n_users, -1)
        right = hidden.unsqueeze(0).expand(stop - start, -1, -1)
        pair = torch.cat([left, right, left * right], dim=2)
        with torch.no_grad():
            rows.append(model.head(pair).squeeze(2))
    return torch.cat(rows)


def edges_to_files(adj, users, folder, t, source_partition):
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


def generate(model, embeddings, users, masks, present_masks, rule, real_degrees, rng, folder,
             partition):
    n_users = len(users)
    adj = torch.zeros((n_users, n_users), dtype=torch.bool, device=DEVICE)
    eye = torch.eye(n_users, dtype=torch.bool, device=DEVICE)
    in_degree = torch.zeros(n_users, device=DEVICE)
    first_edge = torch.full((n_users,), N_SNAPSHOTS, device=DEVICE)
    quota = (torch.from_numpy(rng.choice(real_degrees, size=n_users)).float().to(DEVICE)
             if rule.get('real_k') else None)
    indexes = torch.arange(n_users, device=DEVICE)

    for t in range(N_SNAPSHOTS):
        scores = scores_for_step(model, embeddings[t])
        active = torch.from_numpy(masks[t]).to(DEVICE).unsqueeze(1)
        present = torch.from_numpy(present_masks[t]).to(DEVICE)
        newcomer = first_edge == N_SNAPSHOTS
        allowed = present.unsqueeze(0) & ~adj & ~eye
        if rule.get('growth'):
            allowed = allowed & (newcomer.unsqueeze(1) | newcomer.unsqueeze(0))
        allowed = allowed & active
        masked = scores.masked_fill(~allowed, -torch.inf)

        if rule.get('threshold'):
            # threshold such that the expected number of allowed targets above it is
            # EDGES_PER_USER per active user (estimated on a sample of allowed scores)
            values = masked[torch.isfinite(masked)]
            sample = values if values.numel() <= 2_000_000 else values[
                torch.randint(0, values.numel(), (2_000_000,), device=DEVICE)]
            active_users = int(active.sum().item())
            allowed_total = int(torch.isfinite(masked).sum().item())
            target_edges = max(1, int(round(EDGES_PER_USER * active_users)))
            quantile = max(0.0, 1.0 - target_edges / max(1, allowed_total))
            tau = float(torch.quantile(sample, quantile))
            new = allowed & (scores >= tau)
        else:
            if rule.get('real_k'):
                k = torch.clamp(torch.round(quota / N_SNAPSHOTS), min=1)
            else:
                k = torch.poisson(torch.full((n_users,), LAMBDA, device=DEVICE))
            k = torch.minimum(k, allowed.sum(dim=1))
            k = torch.where(active.squeeze(1), k, torch.zeros_like(k))
            k_max = int(k.max().item())
            if k_max == 0:
                new = torch.zeros_like(adj)
            else:
                if rule.get('growth'):
                    # sampling without replacement proportional to the previous degree
                    weights = torch.log(1.0 + in_degree).unsqueeze(0).expand(n_users, n_users)
                    noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-9) + 1e-9)
                    rank = (weights + noise).masked_fill(~allowed, -torch.inf)
                    top = torch.topk(rank, k_max, dim=1).indices
                else:
                    top = torch.topk(masked, k_max, dim=1).indices
                keep = torch.arange(k_max, device=DEVICE).unsqueeze(0) < k.unsqueeze(1)
                new = torch.zeros_like(adj)
                new[indexes.unsqueeze(1).expand_as(top)[keep], top[keep]] = True
        added = int(new.sum().item())
        adj = adj | new
        touched = new.any(dim=1) | new.any(dim=0)
        first_edge = torch.where(touched & (first_edge == N_SNAPSHOTS),
                                 torch.full_like(first_edge, t), first_edge)
        in_degree = in_degree + new.sum(dim=0).to(torch.float32)

        if rule.get('reciprocity'):
            added += add_reciprocal(adj, rng)
        edges_to_files(adj, users, folder, t, partition)
        print(f'  step {t + 1}: +{added} edges | total {int(adj.sum().item())}', flush=True)
    return adj


def add_reciprocal(adj, rng):
    """Add the reverse of a random subset of non-reciprocal edges to reach the real value."""
    total = int(adj.sum().item())
    if total == 0:
        return 0
    rows, columns = torch.nonzero(adj, as_tuple=True)
    is_reciprocal = adj[columns, rows]
    candidates = [(int(u), int(v)) for u, v, r in zip(rows.tolist(), columns.tolist(),
                                                      is_reciprocal.tolist()) if not r]
    reciprocal = total - len(candidates)
    current = reciprocal / total
    if current >= RECIPROCITY_TARGET or not candidates:
        return 0
    needed = int((RECIPROCITY_TARGET * total - reciprocal) / (1 - RECIPROCITY_TARGET))
    needed = min(needed, len(candidates))
    picked = rng.choice(len(candidates), size=needed, replace=False)
    for index in picked:
        source, target = candidates[int(index)]
        adj[target, source] = True
    return needed


def metrics_from_folder(folder):
    directory = sorted(os.path.join(folder, d) for d in os.listdir(folder)
                       if os.path.isdir(os.path.join(folder, d)))[-1]
    edges = set()
    with open(os.path.join(directory, 'social_network.edg'), encoding='utf-8-sig',
              newline='') as f:
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                edges.add((parts[0].strip(), parts[1].strip()))
    nodes = {n for edge in edges for n in edge}
    out_degree = {}
    for source, _ in edges:
        out_degree[source] = out_degree.get(source, 0) + 1
    reciprocal = sum(1 for a, b in edges if (b, a) in edges) / max(1, len(edges))
    return {'nodes': len(nodes), 'edges': len(edges),
            'density': round(len(edges) / max(1, len(nodes) * (len(nodes) - 1)), 6),
            'reciprocity': round(reciprocal, 4),
            'mean_out_degree': round(len(edges) / max(1, len(nodes)), 2),
            'max_out_degree': max(out_degree.values()) if out_degree else 0}


def write_row(row):
    exists = os.path.exists(REPORT)
    with open(REPORT, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_variants(names, encoder=None, model=None):
    if model is None:
        print('training the model on the real dataset (same seeds as the transfer)', flush=True)
        encoder, users, index, edge_sets, matrices, masks = real_representations('minilm',
                                                                                 'period')
        aligned = procrustes_chain(matrices, masks)
        build = prepare_pairs(edge_sets, users, index, matrices, masks, 'matched')
        model = train_model(aligned, build, index, matrices[0].shape[1])
    degrees = real_out_degrees()
    print(f'real out-degree distribution: mean {degrees.mean():.1f} | max {int(degrees.max())}',
          flush=True)
    for name in names:
        partition, rule = VARIANTS[name]
        print(f'\n[{name}] partition {partition} | rule {sorted(rule)}', flush=True)
        (synthetic_users, _, synthetic_matrices, synthetic_masks,
         synthetic_present, _) = synthetic_representations(encoder, os.path.join(BASE, partition))
        embeddings = node_embeddings(model, synthetic_matrices)
        folder = os.path.join(BASE, f'generated_snapshots_rule_{name}')
        generate(model, embeddings, synthetic_users, synthetic_masks, synthetic_present, rule,
                 degrees, np.random.default_rng(SEED), folder, partition)
        row = {'variant': name, 'partition': partition,
               'rule': ' '.join(sorted(rule)) or 'reference'}
        row.update(metrics_from_folder(folder))
        write_row(row)
        print(f'  -> {row}', flush=True)


def rebuild_report():
    rows = []
    for name, (partition, rule) in VARIANTS.items():
        folder = os.path.join(BASE, f'generated_snapshots_rule_{name}')
        if not os.path.isdir(folder):
            continue
        row = {'variant': name, 'partition': partition,
               'rule': ' '.join(sorted(rule)) or 'reference'}
        row.update(metrics_from_folder(folder))
        rows.append(row)
    with open(REPORT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'rebuilt {REPORT} with {len(rows)} rows')
    for row in rows:
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', default='')
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if args.rebuild:
        rebuild_report()
        return
    names = [n.strip() for n in args.variants.split(',') if n.strip()] or list(VARIANTS)
    run_variants(names)


if __name__ == '__main__':
    main()
