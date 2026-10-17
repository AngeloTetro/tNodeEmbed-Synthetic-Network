"""Scorecard of the generated networks against the real one, on the final snapshot.

One row per folder: the structural metrics already in results/, the relative distance from
the real network for every metric, and the slope of the out-degree CCDF tail (the measure
that tells a tail that is too light from one that is too heavy, which the maximum degree
alone cannot).

The raw edge count is printed with the others but does NOT enter the count of wins: it is
the mean out-degree multiplied by the number of users, so it moves with the population
(10.000 synthetic users against 25.714 real ones) and is not comparable across domains.

Usage: uv run python src/variant_scorecard.py
Output: results/variant_scorecard.csv, results/variant_scorecard.md
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
    ('generated_snapshots_rule_v2', 'v2'),
    ('generated_snapshots_v2_v6a', 'v6a'),
    ('generated_snapshots_v2_v6b', 'v6b'),
    ('generated_snapshots_v2_v7c', 'v7c'),
    ('generated_snapshots_v2_v8', 'v8'),
    ('generated_snapshots_v2_v9_real', 'V9'),
    ('generated_snapshots_growth_v2_v9', 'V9-prog'),
]

METRICS = [
    ('edges', 'struttura', 'archi'),
    ('nodes', 'struttura', 'nodi'),
    ('density', 'struttura', 'densita'),
    ('out_degree_mean', 'struttura', 'grado medio'),
    ('out_degree_max', 'struttura', 'grado massimo'),
    ('reciprocity', 'struttura', 'reciprocita'),
    ('modularity', 'struttura', 'modularita'),
    ('transitivity', 'pesante', 'transitivity'),
    ('avg_clustering', 'pesante', 'clustering'),
    ('homophily_ratio', 'contenuto', 'omofilia'),
]

DECIMALS = {'edges': 0, 'nodes': 0, 'out_degree_max': 0, 'density': 6, 'out_degree_mean': 2,
            'reciprocity': 4, 'modularity': 3, 'transitivity': 4, 'avg_clustering': 4,
            'homophily_ratio': 2}

SCALE_DEPENDENT = {'edges', 'nodes'}
NO_DISCRIMINATION = {'homophily_ratio'}


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

    snapshots = load_network(os.path.join(BASE, folder))
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
        row = {'rete': label}
        row.update(table[label])
        for key, _, _ in METRICS:
            row[f'dist_{key}'] = '' if label == 'reale' else fmt_distance(
                distance(table[label].get(key), reference.get(key)))
        rows.append(row)

    counts = {label: [] for label in labels if label != 'reale'}
    for key, _, name in METRICS:
        if key in SCALE_DEPENDENT or key in NO_DISCRIMINATION:
            continue
        values = {label: distance(table[label].get(key), reference.get(key))
                  for label in labels if label != 'reale'}
        usable = {label: value for label, value in values.items() if not np.isnan(value)}
        if not usable:
            continue
        best = min(usable.values())
        for label, value in usable.items():
            if value == best:
                counts[label].append(name)

    with open(os.path.join(RESULTS, 'variant_scorecard.csv'), 'w', encoding='utf-8',
              newline='') as f:
        fields = ['rete'] + [key for key, _, _ in METRICS] + \
                 ['tail_slope', 'tail_gamma', 'tail_from', 'tail_points'] + \
                 [f'dist_{key}' for key, _, _ in METRICS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ['| metrica | ' + ' | '.join(labels) + ' | chi e\' piu\' vicino |',
             '|---' * (len(labels) + 2) + '|']
    for key, _, name in METRICS:
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
            note = ', '.join(winners)
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' | ' + note + ' |')
    tail_cells = [format_number(table[label].get('tail_slope'), 'tail_slope') for label in labels]
    lines.append('| pendenza di coda (CCDF, ultimo decile) | ' + ' | '.join(tail_cells) + ' | - |')
    gamma_cells = [format_number(table[label].get('tail_gamma'), 'tail_gamma') for label in labels]
    lines.append('| esponente di coda (1 - pendenza) | ' + ' | '.join(gamma_cells) + ' | - |')
    lines.append('')
    lines.append('| rete | metriche su cui e\' piu\' vicina |')
    lines.append('|---|---|')
    for label, names in counts.items():
        lines.append(f'| {label} | ' + (', '.join(names) if names else 'nessuna') + ' |')
    with open(os.path.join(RESULTS, 'variant_scorecard.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    print()
    real_mean = float(reference['out_degree_mean'])
    print(f'volume atteso alla scala del reale: grado medio reale {real_mean:.2f} x nodi della rete')
    for label in labels:
        row = table[label]
        if row.get('edges') in ('', None):
            continue
        edges, nodes = float(row['edges']), float(row['nodes'])
        expected = real_mean * nodes
        print(f'{label:>8}: archi {edges:>9.0f} | nodi {nodes:>6.0f} | attesi {expected:>9.0f} '
              f'| rapporto {edges / expected:.2f}')
    print('written results/variant_scorecard.csv and results/variant_scorecard.md')


def format_number(value, key):
    if value in ('', None):
        return '-'
    decimals = DECIMALS.get(key, 3)
    if decimals == 0:
        return f'{float(value):.0f}'
    return f'{float(value):.{decimals}f}'.replace('.', ',')


def fmt_distance(value):
    if np.isnan(value):
        return ''
    return round(value, 3)


if __name__ == '__main__':
    main()
