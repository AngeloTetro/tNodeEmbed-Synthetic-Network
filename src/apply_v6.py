"""V6 -- calibrated edge rule: the number of edges comes from the real dataset, not from us.

Two calibrations (both anchored on statistics measured on the real dataset):
  v6a "degree"  : the threshold is fitted (bisection) so that the final mean out-degree of the
                  generated network equals the real one (32.05)
  v6b "growth"  : the threshold is the quantile of the model's scores on the REAL validation
                  transition that reproduces the real rate of new edges per pair

Both are applied with the structural mechanism measured on the real:
  - an old user can only open edges towards users entering the network in that time step
    (so no old->old edges: in the real they are 0 out of 707,604);
  - targets are drawn with probability proportional to their previous degree (in the real the
    existing endpoint has mean degree 437 against 64);
  - reverse edges are added to reach the real reciprocity (0.42), which keeps the first
    constraint because it is symmetric.

Four networks are produced: {all, growth} x {v6a, v6b}. Report: results/v6_report.csv.

Usage: uv run python src/apply_v6.py
"""

import csv
import os

import numpy as np
import torch

from apply_edge_rules import edges_to_files, metrics_from_folder, scores_for_step
from transfer_to_synthetic import (BASE, DEVICE, N_SNAPSHOTS, SEED, node_embeddings,
                                   prepare_pairs, procrustes_chain, real_representations,
                                   synthetic_representations, train_model)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
REPORT = os.path.join(RESULTS, 'v6_report.csv')
RECIPROCITY_TARGET = 0.42       # real dataset: 0.4177
VAL_STEP = 6
SCORE_SAMPLE = 4_000_000
FIELDS = ['variant', 'calibration', 'partition', 'target', 'threshold', 'nodes', 'edges',
          'density', 'reciprocity', 'mean_out_degree', 'max_out_degree']


def real_scores_and_rate(model, edge_sets, real_users, real_index, matrices, rng):
    """Scores of sampled pairs at the validation transition, plus the real rate of new edges.

    The candidate set is the network of the PREVIOUS snapshot (users that really exist at that
    point), which is what the real rate of new edges must be measured against.
    """
    hidden = node_embeddings(model, matrices)[VAL_STEP]
    previous_nodes = {n for edge in edge_sets[VAL_STEP - 1] for n in edge}
    present = np.array([real_index[u] for u in sorted(previous_nodes) if u in real_index])
    n_present = len(present)
    new_edges = len(edge_sets[VAL_STEP] - edge_sets[VAL_STEP - 1])
    rate = new_edges / (n_present * (n_present - 1))
    left = rng.choice(present, size=SCORE_SAMPLE)
    right = rng.choice(present, size=SCORE_SAMPLE)
    keep = left != right
    left, right = left[keep], right[keep]
    scores = []
    with torch.no_grad():
        for start in range(0, len(left), 100_000):
            i = torch.from_numpy(left[start:start + 100_000]).to(DEVICE)
            j = torch.from_numpy(right[start:start + 100_000]).to(DEVICE)
            pair = torch.cat([hidden[i], hidden[j], hidden[i] * hidden[j]], dim=1)
            scores.append(model.head(pair).squeeze(1).cpu().numpy())
    return np.concatenate(scores), rate, new_edges, n_present


def add_reciprocal_exact(adj, rng, target=RECIPROCITY_TARGET):
    """Add reverse edges so the reciprocity lands on the real value (pair-based count).

    Reciprocity is counted on directed edges: adding the reverse of (u,v) adds one edge and
    makes TWO directed edges reciprocal, so the number of reverse edges to add is
    (r*E - 2P)/(2 - r) with P = reciprocal pairs.
    """
    total = int(adj.sum().item())
    if total == 0:
        return 0
    rows, columns = torch.nonzero(adj, as_tuple=True)
    is_reciprocal = adj[columns, rows]
    reciprocal_pairs = int(is_reciprocal.sum().item()) // 2
    current = 2 * reciprocal_pairs / total
    if current >= target:
        return 0
    needed = int((target * total - 2 * reciprocal_pairs) / (2 - target))
    candidates = [(int(u), int(v)) for u, v, r in zip(rows.tolist(), columns.tolist(),
                                                      is_reciprocal.tolist()) if not r]
    needed = min(needed, len(candidates))
    if needed <= 0:
        return 0
    picked = rng.choice(len(candidates), size=needed, replace=False)
    for index in picked:
        source, target_node = candidates[int(index)]
        adj[target_node, source] = True
    return needed


def generate(model, embeddings, users, masks, present_masks, partition, threshold, rng,
             write_folder=None, reciprocity=True, step_targets=None, quantile=None):
    """Vectorised generation with the real accretion mechanism.

    The selection can be driven in three ways, all equivalent in spirit (keep the top of the
    model's ranking):
      - fixed `threshold` on the score;
      - `step_targets[t]`: per step, the number of new edges is given (quantile computed on
        the eligible scores);
      - `quantile`: per step, keep the top fraction of the eligible scores.
    """
    n_users = len(users)
    adj = torch.zeros((n_users, n_users), dtype=torch.bool, device=DEVICE)
    eye = torch.eye(n_users, dtype=torch.bool, device=DEVICE)
    first_edge = torch.full((n_users,), N_SNAPSHOTS, device=DEVICE)
    for t in range(N_SNAPSHOTS):
        scores = scores_for_step(model, embeddings[t])
        active = torch.from_numpy(masks[t]).to(DEVICE).unsqueeze(1)
        present = torch.from_numpy(present_masks[t]).to(DEVICE)
        newcomer = first_edge == N_SNAPSHOTS
        allowed = present.unsqueeze(0) & ~adj & ~eye & active
        if bool(newcomer.any()):
            old_sources = ~newcomer
            allowed[old_sources, :] = allowed[old_sources, :] & newcomer.unsqueeze(0)
        else:
            allowed = torch.zeros_like(allowed)
        new = torch.zeros_like(adj)
        if bool(allowed.any()):
            if step_targets is not None or quantile is not None:
                values = scores[allowed]
                sample = values if values.numel() <= 1_000_000 else values[
                    torch.randint(0, values.numel(), (1_000_000,), device=DEVICE)]
                if step_targets is not None:
                    fraction = min(1.0, step_targets[t] / max(1, int(values.numel())))
                else:
                    fraction = quantile
                step_threshold = float(torch.quantile(sample, 1 - fraction))
            else:
                step_threshold = threshold
            new = allowed & (scores >= step_threshold)
        adj = adj | new
        touched = new.any(dim=1) | new.any(dim=0)
        first_edge = torch.where(touched & (first_edge == N_SNAPSHOTS),
                                 torch.full_like(first_edge, t), first_edge)
        if reciprocity:
            add_reciprocal_exact(adj, rng)
        if write_folder is not None:
            edges_to_files(adj, users, write_folder, t, partition)
    return adj


def final_mean_out_degree(adj):
    total = int(adj.sum().item())
    nodes = int((adj.any(dim=1) | adj.any(dim=0)).sum().item())
    return total / max(1, nodes)


def bisect_quantile(model, embeddings, users, masks, present_masks, partition, target, rng):
    """Keep-the-top fraction q giving the target final mean out-degree.

    The model's scores span a very narrow window (see results/_diag_v6.txt), so the search is
    done on the quantile q (log scale) instead of on the raw score.
    """
    low, high = 1e-7, 0.5
    best_q, best_gap = None, None
    for _ in range(6):
        mid = float(np.sqrt(low * high))
        adj = generate(model, embeddings, users, masks, present_masks, partition, None,
                       np.random.default_rng(SEED), quantile=mid)
        value = final_mean_out_degree(adj)
        gap = abs(value - target)
        print(f'    top fraction {mid:.2e} -> mean out-degree {value:.2f}', flush=True)
        if best_gap is None or gap < best_gap:
            best_q, best_gap = mid, gap
        if value > target:
            low = mid
        else:
            high = mid
        del adj
        torch.cuda.empty_cache()
    return best_q


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print('training the model on the real dataset (same seeds as the transfer)', flush=True)
    encoder, users, index, edge_sets, real_matrices, real_masks = real_representations(
        'minilm', 'period')
    aligned = procrustes_chain(real_matrices, real_masks)
    build = prepare_pairs(edge_sets, users, index, real_matrices, real_masks, 'matched')
    model = train_model(aligned, build, index, real_matrices[0].shape[1])

    rng = np.random.default_rng(SEED)
    scores, rate, new_edges, n_present = real_scores_and_rate(model, edge_sets, users, index,
                                                              real_matrices, rng)
    growth_threshold = float(np.quantile(scores, 1 - rate))
    print(f'real validation transition: {new_edges} new edges over {n_present} users present '
          f'in the network -> rate {rate:.3e} per pair | threshold {growth_threshold:+.3f}',
          flush=True)

    target_degree = float(metrics_from_folder(os.path.join(BASE, 'real_snapshots'))
                          ['mean_out_degree'])
    real_new_edges = [len(edge_sets[t] - edge_sets[t - 1]) for t in range(1, len(edge_sets))]
    real_nodes = len({n for edge in edge_sets[-1] for n in edge})
    mean_real_new_edges = sum(real_new_edges) / len(real_new_edges)
    print(f'target: final mean out-degree of the real network = {target_degree:.2f}',
          flush=True)
    print(f'real new edges per transition: mean {mean_real_new_edges:.0f} | '
          f'real nodes {real_nodes}', flush=True)

    rows = []
    for partition in ('generated_snapshots', 'generated_snapshots_growth'):
        (synthetic_users, _, synthetic_matrices, synthetic_masks,
         synthetic_present, _) = synthetic_representations(encoder, os.path.join(BASE, partition))
        embeddings = node_embeddings(model, synthetic_matrices)
        for variant, calibration in (('v6a', 'degree'), ('v6b', 'growth')):
            if calibration == 'degree':
                print(f'\n[{variant}/{partition}] fitting the top fraction for a final mean '
                      f'out-degree ~ {target_degree:.2f}', flush=True)
                quantile = bisect_quantile(model, embeddings, synthetic_users, synthetic_masks,
                                           synthetic_present, partition, target_degree,
                                           np.random.default_rng(SEED))
                target = f'mean out-degree {target_degree:.2f}'
                folder = os.path.join(BASE, f'{partition}_{variant}')
                adj = generate(model, embeddings, synthetic_users, synthetic_masks,
                               synthetic_present, partition, None,
                               np.random.default_rng(SEED), write_folder=folder,
                               quantile=quantile)
                threshold_label = f'top {quantile:.2e}'
            else:
                ratio = len(synthetic_users) / real_nodes
                step_targets = [round(mean_real_new_edges * ratio)] + \
                               [round(count * ratio) for count in real_new_edges]
                target = (f'real per-snapshot new edges x {ratio:.3f} '
                          f'({sum(step_targets)} edges in total)')
                print(f'\n[{variant}/{partition}] per-step targets from the real growth '
                      f'(x {ratio:.3f}): {step_targets}', flush=True)
                folder = os.path.join(BASE, f'{partition}_{variant}')
                adj = generate(model, embeddings, synthetic_users, synthetic_masks,
                               synthetic_present, partition, None,
                               np.random.default_rng(SEED), write_folder=folder,
                               step_targets=step_targets)
                threshold_label = 'per step'
            row = {'variant': variant, 'calibration': calibration, 'partition': partition,
                   'target': target, 'threshold': threshold_label}
            row.update(metrics_from_folder(folder))
            rows.append(row)
            print(f'  -> {row}', flush=True)
            del adj
            torch.cuda.empty_cache()

    with open(REPORT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nwritten {REPORT}')


if __name__ == '__main__':
    import sys
    import traceback

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, '_v6_run.txt'), 'w', encoding='utf-8') as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        try:
            main()
        except Exception:                                    # noqa: BLE001
            traceback.print_exc()
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
    print('run finished: see results/_v6_run.txt')
