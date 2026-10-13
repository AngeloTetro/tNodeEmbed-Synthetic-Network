"""L'omofilia del contenuto del reale: struttura o scelta di somiglianza?

Riproduce la misura di analyze_content.py (similarità coseno fra i profili TF-IDF+SVD degli
estremi di ogni arco, rapporto contro coppie casuali) e la confronta con controlli costruiti con lo
stesso protocollo già usato per i negativi appaiati. Per ogni arco si prende un non-arco i cui due
estremi hanno la stessa posizione strutturale:
    casuale        due utenti a caso: la misura attuale;
    grado          stesso grado (classi a raddoppio) al momento dell'arco;
    grado stretto  dentro +-10% del grado esatto;
    arrivo+grado   anche lo stesso passo di ingresso nella rete;
    testo          stesso volume di testo cumulativo (caratteri), +-10%: la leva della densità dei
                   profili lessicali, indipendente dalla scelta del legame;
    grado+testo    entrambe le cose (classi a raddoppio).
Riporta il rapporto contro ogni controllo e l'AUC fra archi veri e coppie appaiate (media dei passi),
più la curva della somiglianza per decile di grado e il grado medio degli estremi.

Usage: uv run python src/analyze_homophily.py [folder]
Output: results/homophily_control.csv
"""

import csv
import os
import sys

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

from graph_io import cumulative_docs, load_network

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
RESULTS = os.path.join(ROOT, 'results')
SEED = 0
COMPONENTS = 128
LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
CLASSES = len(LADDER) + 1
WINDOW = 1.1
MIN_POOL = 10
DECILE_SAMPLE = 50_000
NAMES = ('casuale', 'grado', 'grado stretto', 'arrivo+grado', 'testo', 'grado+testo')


def degree_class(values):
    out = np.zeros(values.size, dtype=np.int32)
    for limit in LADDER:
        out += (values >= limit).astype(np.int32)
    return out


def class_of(value):
    return sum(1 for limit in LADDER if value >= limit)


def pair_similarity(matrix, index, left, right):
    if not left:
        return np.zeros(0, dtype=np.float64)
    i = np.array([index[u] for u in left])
    j = np.array([index[u] for u in right])
    return np.einsum('ij,ij->i', matrix[i], matrix[j])


def matched_partners(keys, pools, rng):
    """Un partner per posizione, pescato dal gruppo indicato dalla chiave della posizione."""
    unique, inverse = np.unique(keys, return_inverse=True)
    partners = np.empty(keys.size, dtype=np.int64)
    order = np.argsort(inverse, kind='stable')
    boundaries = np.searchsorted(inverse[order], np.arange(len(unique) + 1))
    for position, key in enumerate(unique):
        rows = order[boundaries[position]:boundaries[position + 1]]
        partners[rows] = rng.choice(pools[key], size=rows.size, replace=True)
    return partners


def window_pool(sorted_values, sorted_users, target):
    """Utenti con un valore vicino al bersaglio (finestra allargata se il gruppo è troppo piccolo)."""
    factor = WINDOW
    while True:
        low = np.searchsorted(sorted_values, max(1.0, target / factor))
        high = np.searchsorted(sorted_values, target * factor, side='right')
        if high - low >= MIN_POOL or factor >= 10:
            break
        factor *= 1.5
    if high <= low:
        position = int(np.clip(np.searchsorted(sorted_values, target), 0, sorted_values.size - 1))
        return sorted_users[max(0, position - 1):position + 2]
    return sorted_users[low:high]


def main():
    folder_name = sys.argv[1] if len(sys.argv) > 1 else 'real_snapshots'
    folder = os.path.join(BASE, folder_name)
    snapshots = load_network(folder)
    edge_sets = [set(edges) for _, edges, _ in snapshots]
    _, docs = cumulative_docs(folder)
    users = sorted(docs[-1])
    index = {u: i for i, u in enumerate(users)}
    final = sorted(edge_sets[-1])
    print(f'{folder_name}: archi {len(final)} | utenti con testo {len(users)} | passi {len(snapshots)}')

    vectorizer = TfidfVectorizer(min_df=5, sublinear_tf=True, lowercase=True, strip_accents='unicode',
                                 max_features=200000)
    matrix = normalize(TruncatedSVD(n_components=COMPONENTS, random_state=SEED).fit_transform(
        vectorizer.fit_transform([' '.join(docs[-1][u]) for u in users])))

    arrival_of_edge = {}
    for step, edges in enumerate(edge_sets):
        for edge in edges:
            arrival_of_edge.setdefault(edge, step)
    arrival_of_user = {}
    for step, (_, _, nodes) in enumerate(snapshots):
        for user in nodes:
            arrival_of_user.setdefault(user, step)
    degrees, volumes = [], []
    for step, edges in enumerate(edge_sets):
        counts = {}
        for a, b in edges:
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
        degrees.append(counts)
        volumes.append({u: sum(len(text) for text in docs[step].get(u, [])) for u in users})

    rng = np.random.default_rng(SEED)
    by_step = {}
    for edge in final:
        by_step.setdefault(arrival_of_edge[edge], []).append(edge)
    real_by_step = []
    control_by_step = {name: [] for name in NAMES}

    for step in sorted(by_step):
        rows = [(a, b) for a, b in by_step[step] if a in index and b in index]
        previous = degrees[step - 1] if step > 0 else {}
        text_previous = volumes[step - 1] if step > 0 else {}
        left = [a for a, _ in rows]
        right = [b for _, b in rows]
        real_by_step.append(pair_similarity(matrix, index, left, right))
        control_by_step['casuale'].append(pair_similarity(
            matrix, index,
            [users[i] for i in rng.integers(0, len(users), len(left))],
            [users[i] for i in rng.integers(0, len(users), len(right))]))

        degree_left = np.array([previous.get(u, 0) for u in left])
        degree_right = np.array([previous.get(u, 0) for u in right])
        text_left = np.array([text_previous.get(u, 0) for u in left])
        text_right = np.array([text_previous.get(u, 0) for u in right])
        step_left = np.array([arrival_of_user.get(u, len(edge_sets)) for u in left])
        step_right = np.array([arrival_of_user.get(u, len(edge_sets)) for u in right])
        width = len(edge_sets) + 1

        classes = {}
        for name, value in (('grado', degree_left), ('testo', text_left)):
            classes[name] = degree_class(value)
        classes['grado_destra'] = degree_class(degree_right)
        classes['testo_destra'] = degree_class(text_right)

        present = [u for u in users if arrival_of_user.get(u, len(edge_sets)) <= step]
        values = np.array([previous.get(u, 0) for u in present])
        texts = np.array([text_previous.get(u, 0) for u in present])
        indices = np.array([index[u] for u in present])
        order_degree = np.argsort(values)
        order_text = np.argsort(texts)
        degree_sorted, degree_users = values[order_degree], indices[order_degree]
        text_sorted, text_users = texts[order_text], indices[order_text]

        pools = {name: {} for name in NAMES[1:]}
        for user in users:
            born = arrival_of_user.get(user, len(edge_sets))
            if born > step:
                continue
            degree = class_of(previous.get(user, 0))
            text = class_of(text_previous.get(user, 0))
            pools['grado'].setdefault(degree, []).append(index[user])
            pools['arrivo+grado'].setdefault(degree * width + born, []).append(index[user])
            pools['grado+testo'].setdefault(degree * CLASSES + text, []).append(index[user])
        for name in ('grado', 'arrivo+grado', 'grado+testo'):
            pools[name] = {k: np.array(v) for k, v in pools[name].items()}
        for degree in np.unique(np.concatenate([degree_left, degree_right])):
            pools['grado stretto'][degree] = window_pool(degree_sorted, degree_users, float(degree))
        for text in np.unique(np.concatenate([text_left, text_right])):
            pools['testo'][text] = window_pool(text_sorted, text_users, float(text))

        keys = {
            'grado': (classes['grado'].astype(np.int64), classes['grado_destra'].astype(np.int64)),
            'grado stretto': (degree_left.astype(np.int64), degree_right.astype(np.int64)),
            'testo': (text_left.astype(np.int64), text_right.astype(np.int64)),
            'arrivo+grado': (classes['grado'].astype(np.int64) * width + step_left,
                             classes['grado_destra'].astype(np.int64) * width + step_right),
            'grado+testo': (classes['grado'].astype(np.int64) * CLASSES
                            + classes['testo'].astype(np.int64),
                            classes['grado_destra'].astype(np.int64) * CLASSES
                            + classes['testo_destra'].astype(np.int64))}
        for name in NAMES[1:]:
            key_left, key_right = keys[name]
            partners_left = matched_partners(key_left, pools[name], rng)
            partners_right = matched_partners(key_right, pools[name], rng)
            keep = partners_left != partners_right
            table = edge_sets[step - 1] if step > 0 else set()
            for position in np.nonzero(keep)[0]:
                if (users[partners_left[position]], users[partners_right[position]]) in table:
                    keep[position] = False
            control_by_step[name].append(pair_similarity(
                matrix, index, [users[i] for i in partners_left[keep]],
                [users[i] for i in partners_right[keep]]))
        detail = ' | '.join(f'{name} {control_by_step[name][-1].mean():.4f}' for name in NAMES[1:])
        print(f'  passo {step}: archi {len(rows)} | archi {real_by_step[-1].mean():.4f} | {detail}')

    real = np.concatenate(real_by_step)
    rows = []
    print(f'\narchi veri: somiglianza media {real.mean():.4f}')
    for name in NAMES:
        values = np.concatenate(control_by_step[name])
        aucs = [roc_auc_score(np.concatenate([np.ones(a.size), np.zeros(b.size)]),
                              np.concatenate([a, b]))
                for a, b in zip(real_by_step, control_by_step[name]) if a.size and b.size]
        rows.append({'control': name, 'pairs': values.size,
                     'edge_similarity': round(float(real.mean()), 6),
                     'control_similarity': round(float(values.mean()), 6),
                     'ratio': round(float(real.mean() / values.mean()), 4),
                     'auc_edges_vs_control': round(float(np.mean(aucs)), 4)})
        print(f'  vs {name:13s} coppie {values.size:>7} | somiglianza {values.mean():.4f} '
              f'| rapporto {rows[-1]["ratio"]:.3f} | AUC media sui passi {np.mean(aucs):.4f}')

    total = degrees[-1]
    endpoints = [a for a, _ in final] + [b for _, b in final]
    mean_edge_degree = float(np.mean([total.get(u, 0) for u in endpoints]))
    mean_user_degree = float(np.mean([total.get(u, 0) for u in users]))
    text_final = np.array([volumes[-1][u] for u in users])
    edge_characters = float(np.median([volumes[-1].get(u, 0) for u in endpoints]))
    user_characters = float(np.median(text_final))
    print(f'\ngrado medio: estremi degli archi {mean_edge_degree:.1f} | utenti con testo '
          f'{mean_user_degree:.1f}')
    print(f'caratteri cumulativi per utente: mediana {user_characters:.0f} | '
          f'agli estremi degli archi {edge_characters:.0f}')
    values = np.array([total.get(u, 0) for u in users])
    boundaries = np.quantile(values, np.linspace(0, 1, 11))
    for position in range(10):
        low, high = boundaries[position], boundaries[position + 1]
        inside = [u for u in users if low <= total.get(u, 0) <= high]
        if len(inside) < 20:
            continue
        sample_left = [inside[i] for i in rng.integers(0, len(inside), DECILE_SAMPLE)]
        sample_right = [inside[i] for i in rng.integers(0, len(inside), DECILE_SAMPLE)]
        value = pair_similarity(matrix, index, sample_left, sample_right).mean()
        rows.append({'control': f'decile {position + 1} ({int(low)}-{int(high)} gradi)',
                     'pairs': 2 * DECILE_SAMPLE, 'edge_similarity': '',
                     'control_similarity': round(float(value), 6), 'ratio': '',
                     'auc_edges_vs_control': ''})
        print(f'  decile {position + 1:>2} ({int(low):>4}-{int(high):>4} gradi, {len(inside):>5} '
              f'utenti): somiglianza interna {value:.4f}')

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, 'homophily_control.csv')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['control', 'pairs', 'edge_similarity',
                                               'control_similarity', 'ratio', 'auc_edges_vs_control'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nwritten {path}')


if __name__ == '__main__':
    main()
