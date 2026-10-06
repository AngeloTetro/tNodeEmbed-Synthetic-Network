"""Profile classification on a generated folder, plus its figures.

Uses the labels that exist only in the synthetic dataset (political_leaning, profession,
age_interval) and compares feature sets:
  - tnodeembed : the LSTM embeddings saved by transfer_to_synthetic.py (last time step)
  - content    : TF-IDF + SVD of the cumulative content
  - content+graph : the content vector plus the mean content of the network neighbours
  - tnodeembed+graph : the LSTM embedding plus the mean LSTM embedding of the neighbours

Node split: stratified 80/20, seed 0. Metrics: accuracy, macro-F1, AUC (one-vs-rest macro).

Usage: uv run python src/analyze_profiles.py <folder_name>
Output: results/profiles_<folder>.csv, results/figures/profiles_<folder>_*.png
"""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from graph_io import load_attributes, load_docs, load_edges, snapshots_folder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(RESULTS, 'figures')
ATTRIBUTES = ('political_leaning', 'profession', 'age_interval')
MIN_USERS_PER_CLASS = 5
SEED = 0


def main():
    folder_name = sys.argv[1] if len(sys.argv) > 1 else 'generated_snapshots'
    folder = os.path.join(BASE, folder_name)
    os.makedirs(FIGURES, exist_ok=True)
    directories, names = snapshots_folder(folder)
    last = directories[-1]

    docs = load_docs(os.path.join(last, 'posts_incremental.tsv'))
    labels = load_attributes(os.path.join(last, 'posts_incremental.tsv'), ATTRIBUTES)
    edges = load_edges(os.path.join(last, 'social_network.edg'))
    users = sorted(docs)
    index = {u: i for i, u in enumerate(users)}

    vectorizer = TfidfVectorizer(min_df=5, sublinear_tf=True, lowercase=True, strip_accents='unicode')
    content = normalize(TruncatedSVD(n_components=128, random_state=SEED).fit_transform(
        vectorizer.fit_transform([' '.join(docs[u]) for u in users])))

    neighbours = [[] for _ in users]
    for a, b in edges:
        if a in index and b in index:
            neighbours[index[a]].append(index[b])
            neighbours[index[b]].append(index[a])

    def propagate(matrix):
        propagated = np.zeros_like(matrix)
        for u in range(len(users)):
            if neighbours[u]:
                propagated[u] = matrix[neighbours[u]].mean(axis=0)
        return propagated

    embeddings_path = os.path.join(folder, 'tnodeembed_embeddings.npz')
    features = {'content': content, 'content+graph': np.hstack([content, propagate(content)])}
    if os.path.exists(embeddings_path):
        archive = np.load(embeddings_path, allow_pickle=True)
        embedding_users = [str(u) for u in archive['users']]
        position = {u: i for i, u in enumerate(embedding_users)}
        embedding = archive[f'step_{len(names)}']
        ordered = np.array([embedding[position[u]] if u in position else np.zeros(embedding.shape[1])
                            for u in users], dtype=np.float32)
        features['tnodeembed'] = ordered
        features['tnodeembed+graph'] = np.hstack([ordered, propagate(ordered)])
    else:
        print('no tnodeembed embeddings found: run transfer_to_synthetic.py first')

    rows = []
    for attribute in ATTRIBUTES:
        table = labels[attribute]
        counts = {}
        for u in users:
            if u in table:
                counts[table[u]] = counts.get(table[u], 0) + 1
        keep = {value for value, n in counts.items() if n >= MIN_USERS_PER_CLASS}
        covered = [u for u in users if u in table and table[u] in keep]
        if len(covered) < 100:
            continue
        values = np.array([table[u] for u in covered])
        train_users, test_users, y_train, y_test = train_test_split(
            covered, values, test_size=0.2, random_state=SEED, stratify=values)
        train_rows = [index[u] for u in train_users]
        test_rows = [index[u] for u in test_users]
        for name, matrix in features.items():
            model = LogisticRegression(max_iter=2000)
            model.fit(matrix[train_rows], y_train)
            predicted = model.predict(matrix[test_rows])
            try:
                auc = roc_auc_score(y_test, model.predict_proba(matrix[test_rows]),
                                    multi_class='ovr', average='macro', labels=list(sorted(set(values))))
            except ValueError:
                auc = float('nan')
            row = {'attribute': attribute, 'features': name, 'classes': len(keep),
                   'users': len(covered),
                   'accuracy': accuracy_score(y_test, predicted),
                   'macro_f1': f1_score(y_test, predicted, average='macro', zero_division=0),
                   'auc': auc}
            rows.append(row)
            print(f'{attribute:>18} | {name:<18} accuracy {row["accuracy"]:.3f} '
                  f'macro-F1 {row["macro_f1"]:.3f} AUC {row["auc"]:.3f}')

    with open(os.path.join(RESULTS, f'profiles_{folder_name}.csv'), 'w', encoding='utf-8',
              newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    names_attributes = sorted({row['attribute'] for row in rows})
    names_features = list(features)
    width = 0.8 / len(names_features)
    figure, ax = plt.subplots(figsize=(9, 4.5))
    for offset, feature in enumerate(names_features):
        values = [next((row['auc'] for row in rows
                        if row['attribute'] == attribute and row['features'] == feature), float('nan'))
                  for attribute in names_attributes]
        ax.bar(np.arange(len(names_attributes)) + offset * width, values, width, label=feature)
    ax.set_xticks(np.arange(len(names_attributes)) + 0.4 - width / 2)
    ax.set_xticklabels(names_attributes)
    ax.set_ylabel('AUC (one-vs-rest macro)')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_title(f'profile classification - {folder_name}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, f'profiles_{folder_name}_auc.png'))
    plt.close(figure)
    print(f'written results/profiles_{folder_name}.csv')


if __name__ == '__main__':
    main()
