"""Diagnostic for the V6 threshold: writes its own log (shell redirection is unreliable here)."""

import os

import numpy as np

from apply_v6 import generate
from apply_edge_rules import metrics_from_folder, scores_for_step
from transfer_to_synthetic import (BASE, N_SNAPSHOTS, SEED, node_embeddings, prepare_pairs,
                                   procrustes_chain, real_representations,
                                   synthetic_representations, train_model)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, 'results', '_diag_v6.txt')
THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
PARTITION = 'generated_snapshots_growth'


def log(text):
    print(text, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def main():
    open(LOG, 'w').close()
    log('training the model')
    encoder, users, index, edge_sets, matrices, masks = real_representations('minilm', 'period')
    aligned = procrustes_chain(matrices, masks)
    build = prepare_pairs(edge_sets, users, index, matrices, masks, 'matched')
    model = train_model(aligned, build, index, matrices[0].shape[1])

    (synthetic_users, _, synthetic_matrices, synthetic_masks,
     synthetic_present, _) = synthetic_representations(encoder, os.path.join(BASE, PARTITION))
    embeddings = node_embeddings(model, synthetic_matrices)
    log(f'partition {PARTITION} | synthetic users {len(synthetic_users)}')

    for step in (0, 4, 8):
        scores = scores_for_step(model, embeddings[step])
        finite = scores[torch_finite(scores)]
        quantiles = np.percentile(finite.cpu().numpy(), [50, 90, 99, 99.9, 99.99])
        log(f'  step {step + 1} scores: median {quantiles[0]:+.3f} | p90 {quantiles[1]:+.3f} '
            f'| p99 {quantiles[2]:+.3f} | p99.9 {quantiles[3]:+.3f} | p99.99 {quantiles[4]:+.3f}')

    for threshold in THRESHOLDS:
        adj = generate(model, embeddings, synthetic_users, synthetic_masks, synthetic_present,
                       PARTITION, threshold, np.random.default_rng(SEED), reciprocity=False)
        total = int(adj.sum().item())
        nodes = int((adj.any(dim=1) | adj.any(dim=0)).sum().item())
        log(f'  threshold {threshold:+.2f}: edges {total} | nodes {nodes} | '
            f'mean out-degree {total / max(1, nodes):.2f}')
    log('done')


def torch_finite(tensor):
    import torch
    return torch.isfinite(tensor)


if __name__ == '__main__':
    main()
