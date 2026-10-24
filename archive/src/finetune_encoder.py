"""Lever 1 -- fine-tune the content encoder with a contrastive objective on the real dataset.

The encoder is the only upstream lever: a generic pre-trained model does not know that, in this
dataset, connected users should end up close in content space, and it does not know about time.
Here we teach both:

  positives            real edges of the training transitions, with the content of that time step
  matched negatives    same source, destination with similar arrival snapshot and similar degree
  temporal negatives   pairs that do become edges LATER but are not edges yet at this step
                       (the same pair, "not yet": this is the time signal in the context)
  in-batch negatives   handled by the ranking loss itself

Protocol discipline:
  - training uses only the training transitions (2023 ... 2025_06);
  - after each epoch a proxy validation AUC is computed on the validation transition (cosine
    between the content embeddings of the two endpoints of real new edges, against the matched
    and temporal negatives of that transition) and the best model is kept (early stopping);
  - the test transitions are never used here: they are evaluated later, with the same protocol
    as the other encoders (`tnodeembed_real.py`).

Usage: uv run python src/finetune_encoder.py --base minilm|bge-base [--epochs 3]
Output: models/finetuned-<base>/, results/finetune_report.csv, results/_finetune_<base>.txt
"""

import argparse
import csv
import os
import random
import traceback

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from graph_io import snapshots_folder
from transfer_to_synthetic import load_docs, load_edges

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
RESULTS = os.path.join(ROOT, 'results')
MODELS = os.path.join(ROOT, 'models')
REPORT = os.path.join(RESULTS, 'finetune_report.csv')
BASES = {'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
         'bge-base': 'BAAI/bge-base-en-v1.5'}
TRAIN_STEPS = (1, 2, 3, 4, 5)
VAL_STEP = 6
PAIRS_PER_STEP = 20000
BATCH = 64
SEED = 0

random.seed(SEED)
np.random.seed(SEED)


def text_for(docs, user, step):
    return ' '.join(docs[step].get(user, []))[:512] or 'empty'


def build_pairs(edge_sets, users, index, degrees, arrival, step, size):
    """Positives plus matched negatives (same source, similar arrival/degree) and temporal ones.

    The matched negative is searched inside the arrival bucket of the true target, near it in the
    degree-sorted order, with a bounded number of attempts (the search must stay linear).
    """
    delta = [(a, b) for a, b in (edge_sets[step] - edge_sets[step - 1]) if a in index and b in index]
    positives = random.sample(delta, min(size, len(delta)))
    previous_degree = degrees[step - 1]
    pool = sorted(range(len(users)), key=lambda u: (arrival[u], previous_degree.get(users[u], 0)))
    position = {u: i for i, u in enumerate(pool)}
    bounds = {}
    for position_index, user_index in enumerate(pool):
        bounds.setdefault(int(arrival[user_index]), [position_index, position_index])[1] = position_index
    rng = np.random.default_rng(SEED + step)
    linked = edge_sets[-1]
    matched, seen = [], set()
    for a, b in positives:
        target = index[b]
        low, high = bounds[int(arrival[target])]
        anchor = position[target]
        chosen = None
        for _ in range(200):
            candidate_position = int(np.clip(anchor + rng.integers(-50, 51), low, high))
            candidate = pool[candidate_position]
            if (candidate != index[a] and (a, users[candidate]) not in linked
                    and (a, users[candidate]) not in seen):
                chosen = users[candidate]
                break
        if chosen is not None:
            seen.add((a, chosen))
            matched.append((a, chosen))

    later = []
    for future in range(step + 1, len(edge_sets)):
        later.extend((a, b) for a, b in (edge_sets[future] - edge_sets[future - 1])
                     if a in index and b in index and (a, b) not in edge_sets[step])
    temporal = random.sample(later, min(len(positives), len(later))) if later else []
    return positives, matched, temporal


def validate(model, docs, positives, matched, temporal):
    """Proxy validation AUC on the validation transition: cosine of the endpoint embeddings."""
    negatives = matched + temporal
    users = sorted({u for pair in positives + negatives for u in pair})
    index = {u: i for i, u in enumerate(users)}
    vectors = normalize(model.encode([text_for(docs, u, VAL_STEP) for u in users], batch_size=64,
                                     show_progress_bar=False, convert_to_numpy=True))

    def score(pairs):
        i = np.array([index[a] for a, b in pairs])
        j = np.array([index[b] for a, b in pairs])
        return np.einsum('ij,ij->i', vectors[i], vectors[j])

    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    return roc_auc_score(labels, np.concatenate([score(positives), score(negatives)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='minilm', choices=list(BASES))
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--pairs', type=int, default=PAIRS_PER_STEP)
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    args = parser.parse_args()
    log_path = os.path.join(RESULTS, f'_finetune_{args.base}.txt')

    def log(text):
        print(text, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

    open(log_path, 'w').close()
    directories, names = snapshots_folder(os.path.join(BASE, 'real_snapshots'))
    edge_sets = [load_edges(os.path.join(d, 'social_network.edg')) for d in directories]
    period_docs = [load_docs(os.path.join(d, 'posts_current_snapshot.tsv')) for d in directories]
    users = sorted({u for docs in period_docs for u in docs})
    index = {u: i for i, u in enumerate(users)}
    degrees = []
    for edges in edge_sets:
        degree = {}
        for a, b in edges:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        degrees.append(degree)
    arrival = np.full(len(users), len(directories), dtype=int)
    for step, docs in enumerate(period_docs):
        present = np.array([bool(docs.get(u, [])) for u in users])
        arrival[present & (arrival == len(directories))] = step

    log(f'base {args.base} | users {len(users)} | training transitions {TRAIN_STEPS}')
    examples = []
    for step in TRAIN_STEPS:
        positives, matched, temporal = build_pairs(edge_sets, users, index, degrees, arrival, step,
                                                   args.pairs)
        made = 0
        for (a, b), (_, hard) in zip(positives, matched):
            texts = [text_for(period_docs, a, step), text_for(period_docs, b, step),
                     text_for(period_docs, hard, step)]
            if made < len(temporal):
                texts.append(text_for(period_docs, temporal[made][1], step))
            examples.append(InputExample(texts=texts))
            made += 1
        log(f'  transition {step}: {made} positives (matched negatives {len(matched)}, '
            f'temporal negatives {len(temporal)})')

    val_positives, val_matched, val_temporal = build_pairs(edge_sets, users, index, degrees,
                                                           arrival, VAL_STEP, args.pairs)
    model = SentenceTransformer(BASES[args.base], device='cuda')
    model.eval()
    baseline_auc = validate(model, period_docs, val_positives, val_matched, val_temporal)
    log(f'baseline without fine-tuning: validation proxy AUC {baseline_auc:.4f}')
    model.train()
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch,
                        collate_fn=model.smart_batching_collate)
    criterion = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_auc, best_epoch, history = -1.0, -1, []
    output = os.path.join(MODELS, f'finetuned-{args.base}')
    for epoch in range(1, args.epochs + 1):
        total, steps = 0.0, 0
        for features, labels in loader:
            features = [feature.to(model.device) for feature in features]
            optimizer.zero_grad()
            value = criterion(features, labels)
            value.backward()
            optimizer.step()
            total += float(value.detach())
            steps += 1
            if steps % 200 == 0:
                log(f'    epoch {epoch}: step {steps}, loss {total / steps:.4f}')
        model.eval()
        auc = validate(model, period_docs, val_positives, val_matched, val_temporal)
        model.train()
        history.append(auc)
        log(f'  epoch {epoch}: mean loss {total / max(1, steps):.4f} | '
            f'validation proxy AUC {auc:.4f}')
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            os.makedirs(output, exist_ok=True)
            model.save(output)
    log(f'best epoch {best_epoch} with validation proxy AUC {best_auc:.4f} '
        f'(baseline {baseline_auc:.4f}) -> {output}')

    os.makedirs(RESULTS, exist_ok=True)
    new_file = not os.path.exists(REPORT)
    with open(REPORT, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['base', 'epochs', 'pairs', 'baseline_auc', 'best_epoch',
                             'val_proxy_auc', 'history'])
        writer.writerow([args.base, args.epochs, args.pairs, f'{baseline_auc:.4f}', best_epoch,
                         f'{best_auc:.4f}', '|'.join(f'{value:.4f}' for value in history)])
    log(f'report updated at {REPORT}')


if __name__ == '__main__':
    try:
        main()
    except Exception:                                        # noqa: BLE001
        with open(os.path.join(RESULTS, '_finetune_error.txt'), 'w', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise
