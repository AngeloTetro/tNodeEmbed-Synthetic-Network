"""Shared temporal pipeline: representations, tNodeEmbed, pairs, synthetic representations.

Steps
  1. representations of the real users per time step (encoder fitted on the real training
     content only) and chain of Procrustes rotations;
  2. tNodeEmbed training on the real transitions (training = 2023..2025_06, with the
     validation and test transitions kept aside for the reported metrics);
  3. representations of the synthetic users with the SAME encoder (shared space -> no
     rotation needed) over the 9 time steps of the chosen partition;
  4. node embeddings from the LSTM.

This module is a library: the generation lives in `generate_arrival.py`. The old
command line of this file (k ~ Poisson(1.2)) has been removed.
"""

import os
import random

import numpy as np
import torch

from src.paths import ROOT, TABLES, BASE

from src.lib.content_encoders import encode_grouped, get_encoder
from src.lib.tnodeembed_real import (TRAIN_STEPS, TNodeEmbed, load_docs, load_edges,
                             make_pair_builder, norm, procrustes_chain)

N_SNAPSHOTS = 9
HIDDEN = 128
EPOCHS = 8
BATCH = 2048
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def prepare_pairs(edge_sets, users, index, masks, negatives_mode):
    """(positives, negatives) sampler for one transition; see `tnodeembed_real.make_pair_builder`."""
    return make_pair_builder(edge_sets, users, index, masks, negatives_mode)


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

