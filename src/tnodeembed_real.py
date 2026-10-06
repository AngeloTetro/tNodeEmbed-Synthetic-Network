"""tNodeEmbed on the real dataset, with the content encoder as the only varying part.

Pipeline (one snapshot = one time step, 9 steps):
  1. per-snapshot node representations from post content (encoder: see content_encoders.py);
  2. Q_t initialisation from those representations;
  3. chain of orthogonal Procrustes rotations between consecutive snapshots
     (estimated on the nodes present in both, applied to all);
  4. LSTM over the sequence of representations of each node;
  5. link-prediction head on the pair of temporal embeddings, trained end to end.

Protocol: training = transitions into 2023..2025_06, validation = transition into the val
snapshot, test = transitions into test1 and test2. Positives = new edges of the snapshot;
negatives = pairs never connected, sampled uniformly ("random") or matched to the positive
on the source and on the target's arrival snapshot and degree ("matched").

Output: results/encoder_comparison.csv (one row per encoder and negative mode).
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import roc_auc_score
from torch import nn

from content_encoders import get_encoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
RESULTS = os.path.join(ROOT, 'results')
N_SNAPSHOTS = 9
TRAIN_STEPS = (1, 2, 3, 4, 5)      # transitions into 2023, 2024, 2025_01-03, 2025_04-05, 2025_06
VAL_STEP = 6                       # transition into the val snapshot
TEST_STEPS = (7, 8)                # transitions into test1 and test2
HIDDEN = 128
EPOCHS = 8
BATCH = 2048
MAX_PAIRS = 20000
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
                edges.add((norm(parts[0]), norm(parts[1])))
    return edges


def load_docs(path):
    docs = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = [h.strip() for h in next(reader)]
        user_col = header.index('account_id')
        text_col = header.index('content')
        for row in reader:
            if len(row) != len(header):
                continue
            user, text = norm(row[user_col]), row[text_col].strip()
            if user and text:
                docs.setdefault(user, []).append(text)
    return docs


def procrustes_chain(matrices, masks):
    """Bring every snapshot into the frame of the first, with a chain of rotations."""
    aligned = [matrices[0]]
    cumulative = np.eye(matrices[0].shape[1], dtype=np.float32)
    for t in range(1, len(matrices)):
        common = np.where(masks[t] & masks[t - 1])[0]
        if len(common) >= matrices[t].shape[1]:
            rotation, _ = orthogonal_procrustes(matrices[t][common], matrices[t - 1][common])
            cumulative = (rotation.astype(np.float32) @ cumulative).astype(np.float32)
        aligned.append((matrices[t] @ cumulative).astype(np.float32))
    return aligned


class TNodeEmbed(nn.Module):
    def __init__(self, dim, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, left, right):
        h_left = self.lstm(left)[0][:, -1]
        h_right = self.lstm(right)[0][:, -1]
        return self.head(torch.cat([h_left, h_right, h_left * h_right], dim=1)).squeeze(1)


def run(encoder_name, content, negatives_mode, verbose=True):
    snapshots = sorted(os.listdir(os.path.join(BASE, 'real_snapshots')))
    edge_sets = [load_edges(os.path.join(BASE, 'real_snapshots', s, 'social_network.edg'))
                 for s in snapshots]
    file_name = 'posts_current_snapshot.tsv' if content == 'period' else 'posts_incremental.tsv'
    docs_per_snapshot = [load_docs(os.path.join(BASE, 'real_snapshots', s, file_name))
                         for s in snapshots]

    users = sorted({u for docs in docs_per_snapshot for u in docs})
    index = {u: i for i, u in enumerate(users)}
    n_users = len(users)

    encoder = get_encoder(encoder_name)
    # the encoder is fitted on the training content only, then applied to every snapshot
    fit_docs = [' '.join(docs_per_snapshot[t].get(u, [])) for t in TRAIN_STEPS for u in users]
    encoder.fit([d for d in fit_docs if d])

    matrices, masks = [], []
    for t in range(N_SNAPSHOTS):
        documents = [' '.join(docs_per_snapshot[t].get(u, [])) for u in users]
        present = np.array([bool(d) for d in documents])
        vectors = np.zeros((n_users, 1), dtype=np.float32)
        if present.any():
            encoded = encoder.transform([d for d in documents if d])
            vectors = np.zeros((n_users, encoded.shape[1]), dtype=np.float32)
            vectors[present] = encoded
        matrices.append(vectors)
        masks.append(present)
    if verbose:
        print(f'[{encoder_name}/{content}] users {n_users} | dim {matrices[0].shape[1]} '
              f'| users with content per snapshot {[int(m.sum()) for m in masks]}')

    aligned = procrustes_chain(matrices, masks)

    degrees = []
    for edges in edge_sets:
        deg = {}
        for a, b in edges:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        degrees.append(deg)
    arrival = np.full(n_users, N_SNAPSHOTS, dtype=int)
    for t, mask in enumerate(masks):
        arrival[mask & (arrival == N_SNAPSHOTS)] = t

    def build_pairs(step):
        delta = [(a, b) for a, b in (edge_sets[step] - edge_sets[step - 1])
                 if a in index and b in index]
        positives = random.sample(delta, min(MAX_PAIRS, len(delta)))
        negatives, seen = [], set()
        if negatives_mode == 'random':
            while len(negatives) < len(positives):
                a, b = random.choice(users), random.choice(users)
                if a != b and (a, b) not in seen and (a, b) not in edge_sets[-1]:
                    seen.add((a, b))
                    negatives.append((a, b))
        else:
            degree_previous = degrees[step - 1]
            pool = sorted(range(n_users), key=lambda u: (arrival[u], degree_previous.get(users[u], 0)))
            position = {u: i for i, u in enumerate(pool)}
            for a, b in positives:
                target = index[b]
                p, chosen = position[target], None
                for offset in range(1, 20000):
                    for candidate_position in (p - offset, p + offset):
                        if 0 <= candidate_position < n_users:
                            candidate = pool[candidate_position]
                            candidate_id = users[candidate]
                            if (candidate != index[a] and arrival[candidate] == arrival[target]
                                    and (a, candidate_id) not in edge_sets[-1]
                                    and (a, candidate_id) not in seen):
                                chosen = candidate_id
                                break
                    if chosen is not None:
                        break
                if chosen is not None:
                    seen.add((a, chosen))
                    negatives.append((a, chosen))
        return positives, negatives

    def tensors(step, pairs):
        """Sequence of representations from the first snapshot up to this time step."""
        left = np.stack([np.stack([aligned[s][index[a]] for s in range(step + 1)]) for a, _ in pairs])
        right = np.stack([np.stack([aligned[s][index[b]] for s in range(step + 1)]) for _, b in pairs])
        return left.astype(np.float32), right.astype(np.float32)

    train_batches = []
    for step in TRAIN_STEPS:
        positives, negatives = build_pairs(step)
        pairs = positives + negatives
        labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))]).astype(np.float32)
        for start in range(0, len(pairs), BATCH):
            train_batches.append((step, pairs[start:start + BATCH], labels[start:start + BATCH]))

    model = TNodeEmbed(matrices[0].shape[1]).to('cuda' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    final_loss = float('nan')
    for epoch in range(EPOCHS):
        random.shuffle(train_batches)
        total = 0.0
        for step, chunk, labels in train_batches:
            left, right = tensors(step, chunk)
            optimizer.zero_grad()
            scores = model(torch.from_numpy(left).to(device), torch.from_numpy(right).to(device))
            loss = loss_fn(scores, torch.from_numpy(labels).to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
        final_loss = total / len(train_batches)
        if verbose:
            print(f'  epoch {epoch + 1}/{EPOCHS}: loss {final_loss:.4f}')

    def predict(step, pairs):
        scores = []
        with torch.no_grad():
            for start in range(0, len(pairs), BATCH):
                left, right = tensors(step, pairs[start:start + BATCH])
                scores.append(torch.sigmoid(model(torch.from_numpy(left).to(device),
                                                  torch.from_numpy(right).to(device))).cpu().numpy())
        return np.concatenate(scores)

    metrics, restrictions = {}, {}
    model.eval()
    for step in (VAL_STEP,) + TEST_STEPS:
        positives, negatives = build_pairs(step)
        scores = np.concatenate([predict(step, positives), predict(step, negatives)])
        labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
        metrics[step] = roc_auc_score(labels, scores)
        previous_nodes = {n for edge in edge_sets[step - 1] for n in edge}
        restrictions[step] = sum(1 for a, b in (edge_sets[step] - edge_sets[step - 1])
                                 if a in previous_nodes and b in previous_nodes)
        if verbose:
            print(f'  step {step}: AUC {metrics[step]:.4f} | edges satisfying the paper '
                  f'restriction: {restrictions[step]}')
    return metrics, restrictions, final_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoders', default='tfidf,minilm')
    parser.add_argument('--content', default='period', choices=('period', 'cumulative'))
    parser.add_argument('--negatives', default='random,matched')
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, 'encoder_comparison.csv')
    new_file = not os.path.exists(path)
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['encoder', 'content', 'negatives', 'val_auc', 'test1_auc', 'test2_auc',
                             'restriction_val', 'restriction_test1', 'restriction_test2',
                             'final_loss', 'seconds'])
        for encoder_name in args.encoders.split(','):
            for negatives_mode in args.negatives.split(','):
                started = time.time()
                metrics, restrictions, final_loss = run(encoder_name.strip(), args.content,
                                                        negatives_mode.strip())
                row = [encoder_name.strip(), args.content, negatives_mode.strip(),
                       f'{metrics[VAL_STEP]:.4f}', f'{metrics[TEST_STEPS[0]]:.4f}',
                       f'{metrics[TEST_STEPS[1]]:.4f}',
                       restrictions[VAL_STEP], restrictions[TEST_STEPS[0]], restrictions[TEST_STEPS[1]],
                       f'{final_loss:.4f}', f'{time.time() - started:.0f}']
                writer.writerow(row)
                f.flush()
                print('  ->', row)
    print(f'written to {path}')


if __name__ == '__main__':
    main()
