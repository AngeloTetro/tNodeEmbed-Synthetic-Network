"""Train tNodeEmbed on the real dataset, then infer the synthetic network.

Steps
  1. representations of the real users per time step (encoder fitted on the real training
     content only) and chain of Procrustes rotations;
  2. tNodeEmbed training on the real transitions (training = 2023..2025_06, with the
     validation and test transitions kept aside for the reported metrics);
  3. representations of the synthetic users with the SAME encoder (shared space -> no
     rotation needed) over the 9 time steps of the chosen partition;
  4. node embeddings from the LSTM, then scoring of the ordered pairs and selection of the
     new edges: each active user adds k ~ Poisson(1.2) edges towards its highest-scoring
     not-yet-connected users (same counting rule for every configuration);
  5. the network is written snapshot by snapshot inside the partition folder, together
     with results/transfer_report.csv.

Usage: uv run python src/transfer_to_synthetic.py <partition_folder> <encoder> <negatives>
"""

import csv
import os
import random
import re
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from content_encoders import encode_grouped, get_encoder
from tnodeembed_real import (BASE, TEST_STEPS, TRAIN_STEPS, VAL_STEP, TNodeEmbed, load_docs,
                             load_edges, norm, procrustes_chain)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
N_SNAPSHOTS = 9
HIDDEN = 128
EPOCHS = 8
BATCH = 2048
MAX_PAIRS = 20000
BLOCK = 16          # users scored at once when ranking candidates
LAMBDA = 1.2        # mean new outgoing edges per active user per time step
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WORDS = re.compile(r"[a-z']+")


def words(text):
    return WORDS.findall(text.lower())


def real_representations(encoder_name, content, pooling='concat'):
    snapshots = sorted(os.listdir(os.path.join(BASE, 'real_snapshots')))
    edge_sets = [load_edges(os.path.join(BASE, 'real_snapshots', s, 'social_network.edg'))
                 for s in snapshots]
    file_name = 'posts_current_snapshot.tsv' if content == 'period' else 'posts_incremental.tsv'
    docs = [load_docs(os.path.join(BASE, 'real_snapshots', s, file_name)) for s in snapshots]
    users = sorted({u for d in docs for u in d})
    index = {u: i for i, u in enumerate(users)}
    encoder = get_encoder(encoder_name)
    if pooling == 'mean_of_posts':
        encoder.fit([text for t in TRAIN_STEPS for u in users
                     for text in docs[t].get(u, []) if text])
    else:
        encoder.fit([text for t in TRAIN_STEPS for text in
                     (' '.join(docs[t].get(u, [])) for u in users) if text])
    matrices, masks = [], []
    for t in range(N_SNAPSHOTS):
        if pooling == 'mean_of_posts':
            groups = [docs[t].get(u, []) for u in users]
            present = np.array([bool(group) for group in groups])
            matrix = (encode_grouped(encoder, groups) if present.any()
                      else np.zeros((len(users), 1), dtype=np.float32))
        else:
            documents = [' '.join(docs[t].get(u, [])) for u in users]
            present = np.array([bool(d) for d in documents])
            encoded = (encoder.transform([d for d in documents if d]) if present.any()
                       else np.zeros((0, 1), dtype=np.float32))
            matrix = np.zeros((len(users), encoded.shape[1]), dtype=np.float32)
            matrix[present] = encoded
        matrices.append(matrix)
        masks.append(present)
    return encoder, users, index, edge_sets, matrices, masks


def prepare_pairs(edge_sets, users, index, matrices, masks, negatives_mode):
    n_users = len(users)
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

    def build(step):
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
            previous_degree = degrees[step - 1]
            pool = sorted(range(n_users), key=lambda u: (arrival[u], previous_degree.get(users[u], 0)))
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

    return build


def sequence_tensor(aligned, step, pairs, index):
    left = np.stack([np.stack([aligned[s][index[a]] for s in range(step + 1)]) for a, _ in pairs])
    right = np.stack([np.stack([aligned[s][index[b]] for s in range(step + 1)]) for _, b in pairs])
    return left.astype(np.float32), right.astype(np.float32)


def train_model(aligned, build, index, dim):
    model = TNodeEmbed(dim).to(DEVICE)
    batches = []
    for step in TRAIN_STEPS:
        positives, negatives = build(step)
        pairs = positives + negatives
        labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))]).astype(np.float32)
        for start in range(0, len(pairs), BATCH):
            batches.append((step, pairs[start:start + BATCH], labels[start:start + BATCH]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(EPOCHS):
        random.shuffle(batches)
        total = 0.0
        for step, chunk, labels in batches:
            left, right = sequence_tensor(aligned, step, chunk, index)
            optimizer.zero_grad()
            scores = model(torch.from_numpy(left).to(DEVICE), torch.from_numpy(right).to(DEVICE))
            loss = loss_fn(scores, torch.from_numpy(labels).to(DEVICE))
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'  epoch {epoch + 1}/{EPOCHS}: loss {total / len(batches):.4f}')
    return model


def real_metrics(model, aligned, build, index):
    metrics = {}
    for step in (VAL_STEP,) + TEST_STEPS:
        positives, negatives = build(step)
        scores = []
        with torch.no_grad():
            for group in (positives, negatives):
                left, right = sequence_tensor(aligned, step, group, index)
                scores.append(torch.sigmoid(model(torch.from_numpy(left).to(DEVICE),
                                                  torch.from_numpy(right).to(DEVICE))).cpu().numpy())
        labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
        metrics[step] = roc_auc_score(labels, np.concatenate(scores))
    return metrics


def synthetic_representations(encoder, folder, pooling='concat'):
    snapshots = [os.path.join(folder, f'snapshot_{t}') for t in range(1, N_SNAPSHOTS + 1)]
    docs = [load_docs(os.path.join(s, 'posts_current_snapshot.tsv')) for s in snapshots]
    cumulative = [load_docs(os.path.join(s, 'posts_incremental.tsv')) for s in snapshots]
    users = sorted({u for d in cumulative for u in d})
    index = {u: i for i, u in enumerate(users)}
    matrices, masks, present_masks = [], [], []
    for t in range(N_SNAPSHOTS):
        if pooling == 'mean_of_posts':
            groups = [docs[t].get(u, []) for u in users]
            present = np.array([bool(group) for group in groups])
            matrix = (encode_grouped(encoder, groups) if present.any()
                      else np.zeros((len(users), 1), dtype=np.float32))
        else:
            documents = [' '.join(docs[t].get(u, [])) for u in users]
            present = np.array([bool(d) for d in documents])
            encoded = encoder.transform([d for d in documents if d]) if present.any() else \
                np.zeros((0, 1), dtype=np.float32)
            matrix = np.zeros((len(users), encoded.shape[1]), dtype=np.float32)
            matrix[present] = encoded
        matrices.append(matrix)
        masks.append(present)                                   # writes in this time step
        present_masks.append(np.array([bool(cumulative[t].get(u, [])) for u in users]))  # born <= t
    return users, index, matrices, masks, present_masks, cumulative


def node_embeddings(model, matrices):
    """LSTM embedding of every node at every time step (features are in a shared space)."""
    n_users = matrices[0].shape[0]
    embeddings = []
    with torch.no_grad():
        for t in range(N_SNAPSHOTS):
            sequence = np.stack([np.stack([matrices[s][u] for s in range(t + 1)])
                                 for u in range(n_users)])
            tensor = torch.from_numpy(sequence.astype(np.float32)).to(DEVICE)
            hidden = []
            for start in range(0, n_users, 2048):
                hidden.append(model.lstm(tensor[start:start + 2048])[0][:, -1])
            embeddings.append(torch.cat(hidden))
    return embeddings


def generate_network(model, embeddings, users, index, masks, present_masks, folder, rng):
    """masks = who writes in this time step (may open new edges); present_masks = who is
    already in the network (born at or before this time step) and can therefore be a target."""
    edges = set()
    per_snapshot = []
    for t in range(N_SNAPSHOTS):
        hidden = embeddings[t]
        active = torch.from_numpy(masks[t]).to(DEVICE)
        present = torch.from_numpy(present_masks[t]).to(DEVICE)
        already = {u: set() for u in range(len(users))}
        for source, target in edges:
            already[source].add(target)
        new_edges = 0
        for start in range(0, len(users), BLOCK):
            stop = min(start + BLOCK, len(users))
            block = hidden[start:stop]                              # (b, H)
            left = block.unsqueeze(1).expand(-1, len(users), -1)     # (b, n, H)
            right = hidden.unsqueeze(0).expand(stop - start, -1, -1)
            pair = torch.cat([left, right, left * right], dim=2)     # (b, n, 3H)
            with torch.no_grad():
                scores = model.head(pair).squeeze(2)                 # (b, n)
            for offset, user in enumerate(range(start, stop)):
                if not bool(active[user]):
                    continue
                row = scores[offset].clone()
                row[user] = -torch.inf
                row[~present] = -torch.inf         # targets: users already present (born <= t)
                for target in already[user]:
                    row[target] = -torch.inf
                k = int(rng.poisson(LAMBDA))
                if k <= 0:
                    continue
                values, indices = torch.topk(row, k)
                for value, target in zip(values.tolist(), indices.tolist()):
                    if value == -torch.inf:
                        continue
                    if (user, target) not in edges:
                        edges.add((user, target))
                        already[user].add(target)
                        new_edges += 1
        per_snapshot.append(sorted(edges))
        directory = os.path.join(folder, f'snapshot_{t + 1}')
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'social_network.edg'), 'w', encoding='utf-8',
                  newline='') as f:
            for source, target in per_snapshot[t]:
                f.write(f'{users[source]}\t{users[target]}\r\n')
        print(f'  time step {t + 1}: +{new_edges} edges | total {len(edges)}')
    return per_snapshot


def vocabulary_overlap(real_docs, synthetic_cumulative):
    real_words = set()
    for texts in real_docs.values():
        for text in texts:
            real_words.update(words(text))
    synthetic_words = set()
    for texts in synthetic_cumulative.values():
        for text in texts:
            synthetic_words.update(words(text))
    shared = synthetic_words & real_words
    return len(real_words), len(synthetic_words), len(shared), len(shared) / max(1, len(synthetic_words))


def main():
    folder_name = sys.argv[1] if len(sys.argv) > 1 else 'generated_snapshots'
    encoder_name = sys.argv[2] if len(sys.argv) > 2 else 'tfidf'
    negatives_mode = sys.argv[3] if len(sys.argv) > 3 else 'random'
    pooling = sys.argv[4] if len(sys.argv) > 4 else 'concat'
    folder = os.path.join(BASE, folder_name)
    started = time.time()

    print(f'[{folder_name}] encoder {encoder_name} | negatives {negatives_mode} '
          f'| pooling {pooling}')
    encoder, users, index, edge_sets, matrices, masks = real_representations(encoder_name, 'period',
                                                                            pooling)
    aligned = procrustes_chain(matrices, masks)
    build = prepare_pairs(edge_sets, users, index, matrices, masks, negatives_mode)
    model = train_model(aligned, build, index, matrices[0].shape[1])
    metrics = real_metrics(model, aligned, build, index)
    print(f'  real validation {metrics[VAL_STEP]:.4f} | test '
          f'{metrics[TEST_STEPS[0]]:.4f} / {metrics[TEST_STEPS[1]]:.4f}')

    synthetic_users, synthetic_index, synthetic_matrices, synthetic_masks, synthetic_present, \
        synthetic_docs = synthetic_representations(encoder, folder, pooling)
    embeddings = node_embeddings(model, synthetic_matrices)
    np.savez_compressed(os.path.join(folder, 'tnodeembed_embeddings.npz'),
                        users=np.array(synthetic_users),
                        **{f'step_{t + 1}': embeddings[t].cpu().numpy() for t in range(N_SNAPSHOTS)})
    per_snapshot = generate_network(model, embeddings, synthetic_users, synthetic_index,
                                    synthetic_masks, synthetic_present, folder,
                                    np.random.default_rng(SEED))

    real_words, synthetic_words, shared, ratio = vocabulary_overlap(
        load_docs(os.path.join(BASE, 'real_snapshots', sorted(os.listdir(os.path.join(BASE, 'real_snapshots')))[-1],
                               'posts_incremental.tsv')), synthetic_docs[-1])
    final = per_snapshot[-1]
    reciprocity = sum(1 for a, b in final if (b, a) in final) / max(1, len(final))
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, 'transfer_report.csv')
    new_file = not os.path.exists(path)
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['partition', 'encoder', 'negatives', 'real_val_auc', 'real_test1_auc',
                             'real_test2_auc', 'nodes', 'edges_final', 'density_final', 'reciprocity_final',
                             'real_words', 'synthetic_words', 'shared_words', 'overlap_ratio', 'seconds'])
        writer.writerow([folder_name, encoder_name, negatives_mode, f'{metrics[VAL_STEP]:.4f}',
                         f'{metrics[TEST_STEPS[0]]:.4f}', f'{metrics[TEST_STEPS[1]]:.4f}',
                         len(synthetic_users), len(final),
                         f'{len(final) / (len(synthetic_users) * (len(synthetic_users) - 1)):.5f}',
                         f'{reciprocity:.3f}', real_words, synthetic_words, shared, f'{ratio:.3f}',
                         f'{time.time() - started:.0f}'])
    print(f'  vocabulary overlap {ratio:.3f} ({shared}/{synthetic_words} synthetic words)')
    print(f'report updated at {path}')


if __name__ == '__main__':
    main()
