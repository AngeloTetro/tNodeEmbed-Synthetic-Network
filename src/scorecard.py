"""Scorecard of the delivered network and of its control against the real one, last snapshot.

Two networks are compared: the official one (V9 on the progressive-arrival partition) and the
control (V9 with every user present from the first step). For every metric the file reports the
value and the relative distance from the real network, defined once and for all as

    distance = |value - real| / |real|

stored as a fraction with four decimals. The raw edge count and the node count do not enter the
aggregate: they move with the population (9.900 synthetic users against 25.714 real ones), so
they are not comparable across domains. Homophily does not enter either: every generated network
sits at 1,00, so it contributes the same constant to all of them.

The aggregate is the sum of the seven comparable distances (and its mean); it is a descriptive
index, not a criterion. The two partitions differ by 0,04 on it, and reversing the normalisation
(e.g. dividing by the maximum instead of the real value) reverses the order: they are
equivalent, and the official one is chosen for the realism of arrivals, not for this number.

Usage: uv run python src/scorecard.py
Output: results/scorecard.csv, results/scorecard.md
"""

import csv
import os

import numpy as np

from graph_io import load_network

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
RESULTS = os.path.join(ROOT, 'results')

FOLDERS = [
    ('real_snapshots', 'reale'),
    ('synthetic_network', 'ufficiale'),
    ('synthetic_control_network', 'controllo'),
]

METRICS = [
    ('edges', 'archi'),
    ('nodes', 'nodi'),
    ('density', 'densita'),
    ('out_degree_mean', 'grado medio'),
    ('out_degree_max', 'grado massimo'),
    ('reciprocity', 'reciprocita'),
    ('modularity', 'modularita'),
    ('transitivity', 'transitivity'),
    ('avg_clustering', 'clustering'),
    ('homophily_ratio', 'omofilia'),
]

DECIMALS = {'edges': 0, 'nodes': 0, 'out_degree_max': 0, 'density': 6, 'out_degree_mean': 2,
            'reciprocity': 4, 'modularity': 3, 'transitivity': 4, 'avg_clustering': 4,
            'homophily_ratio': 2}

SCALE_DEPENDENT = {'edges', 'nodes'}
NO_DISCRIMINATION = {'homophily_ratio'}
COMPARABLE = [key for key, _ in METRICS
              if key not in SCALE_DEPENDENT and key not in NO_DISCRIMINATION]


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def folder_metrics(folder):
    """Metrics of the final snapshot of one folder, plus the tail slope of its out-degree."""
    row = {}
    structure = read_rows(os.path.join(RESULTS, f'structure_{folder}.csv'))
    heavy = read_rows(os.path.join(RESULTS, f'structure_heavy_{folder}.csv'))
    content = read_rows(os.path.join(RESULTS, f'content_{folder}.csv'))
    row['edges'] = structure[-1]['edges'] if structure else ''
    row['nodes'] = structure[-1]['nodes'] if structure else ''
    for key in ('density', 'out_degree_mean', 'out_degree_max', 'reciprocity', 'modularity'):
        row[key] = structure[-1][key] if structure else ''
    for key in ('transitivity', 'avg_clustering'):
        row[key] = heavy[-1][key] if heavy else ''
    row['homophily_ratio'] = content[-1]['homophily_ratio'] if content else ''

    try:
        snapshots = load_network(os.path.join(BASE, folder))
    except (FileNotFoundError, IndexError, NotADirectoryError):
        row['tail_slope'], row['tail_gamma'], row['tail_from'], row['tail_points'] = '', '', '', ''
        return row
    degrees = {}
    for source, _ in snapshots[-1][1]:
        degrees[source] = degrees.get(source, 0) + 1
    values = np.sort(np.array(list(degrees.values()), dtype=float))
    if values.size < 20:
        row['tail_slope'], row['tail_gamma'], row['tail_from'], row['tail_points'] = '', '', '', ''
        return row
    ccdf = 1.0 - np.arange(values.size) / values.size
    threshold = float(np.quantile(values, 0.90))
    mask = values >= threshold
    slope = float(np.polyfit(np.log(values[mask]), np.log(ccdf[mask]), 1)[0])
    row['tail_slope'] = round(slope, 3)
    row['tail_gamma'] = round(1.0 - slope, 2)
    row['tail_from'] = round(threshold, 1)
    row['tail_points'] = int(mask.sum())
    return row


def distance(value, reference):
    try:
        return abs(float(value) - float(reference)) / abs(float(reference))
    except (TypeError, ValueError, ZeroDivisionError):
        return float('nan')


def main():
    table = {label: folder_metrics(folder) for folder, label in FOLDERS}
    reference = table['reale']
    labels = [label for _, label in FOLDERS]

    rows = []
    for label in labels:
        row = {'network': label}
        row.update(table[label])
        comparable = []
        for key, _ in METRICS:
            if label == 'reale':
                row[f'dist_{key}'] = ''
                continue
            value = distance(table[label].get(key), reference.get(key))
            row[f'dist_{key}'] = '' if np.isnan(value) else round(value, 4)
            if key in COMPARABLE and not np.isnan(value):
                comparable.append(value)
        row['dist_total'] = round(float(np.sum(comparable)), 4) if comparable else ''
        row['dist_mean'] = round(float(np.mean(comparable)), 4) if comparable else ''
        rows.append(row)

    with open(os.path.join(RESULTS, 'scorecard.csv'), 'w', encoding='utf-8',
              newline='') as f:
        fields = ['network'] + [key for key, _ in METRICS] + \
                 ['tail_slope', 'tail_gamma', 'tail_from', 'tail_points'] + \
                 [f'dist_{key}' for key, _ in METRICS] + ['dist_total', 'dist_mean']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ['| metrica | ' + ' | '.join(labels) + ' | piu\' vicino |',
             '|---' * (len(labels) + 2) + '|']
    for key, name in METRICS:
        cells = []
        for label in labels:
            value = table[label].get(key)
            cells.append('-' if value in ('', None) else format_number(value, key))
        values = {label: distance(table[label].get(key), reference.get(key))
                  for label in labels if label != 'reale'}
        usable = {label: value for label, value in values.items() if not np.isnan(value)}
        best = min(usable.values()) if usable else float('nan')
        winners = [label for label, value in usable.items() if value == best]
        if key in SCALE_DEPENDENT:
            note = 'non confrontabile (scala)'
        elif key in NO_DISCRIMINATION:
            note = 'nessuna: tutte vicine a 1,00'
        else:
            note = ', '.join(winners) if len(winners) < len(usable) else 'pari'
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' | ' + note + ' |')
    tail_cells = [format_number(table[label].get('tail_slope'), 'tail_slope') for label in labels]
    lines.append('| pendenza di coda (CCDF, ultimo decile) | ' + ' | '.join(tail_cells) + ' | - |')
    lines.append('')
    lines.append('Distanza relativa dal reale, `|v - reale| / |reale|`, sommata sulle sette '
                 'metriche confrontabili (densita, grado medio, grado massimo, reciprocita, '
                 'modularita, transitivity, clustering):')
    lines.append('')
    lines.append('| network | sum | mean |')
    lines.append('|---|---:|---:|')
    for row in rows:
        if row['network'] == 'reale':
            continue
        lines.append(f"| {row['network']} | {row['dist_total']:.2f} | {row['dist_mean']:.3f} |")
    with open(os.path.join(RESULTS, 'scorecard.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print()
    real_mean = float(reference['out_degree_mean'])
    print(f'volume alla scala del reale: grado medio reale {real_mean:.2f} x nodi della rete')
    for label in labels:
        row = table[label]
        if row.get('edges') in ('', None):
            continue
        edges, nodes = float(row['edges']), float(row['nodes'])
        expected = real_mean * nodes
        print(f'{label:>10}: archi {edges:>9.0f} | nodi {nodes:>6.0f} | attesi {expected:>9.0f} '
              f'| rapporto {edges / expected:.2f}')
    print('written results/scorecard.csv and results/scorecard.md')


def format_number(value, key):
    if value in ('', None):
        return '-'
    decimals = DECIMALS.get(key, 3)
    if decimals == 0:
        return f'{float(value):.0f}'
    return f'{float(value):.{decimals}f}'.replace('.', ',')


if __name__ == '__main__':
    main()
