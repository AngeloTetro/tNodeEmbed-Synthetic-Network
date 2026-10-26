"""Is the degree tail really underestimated, or is it the population difference?

The project claims (FINAL.md §10, docs/pipeline.md §6) that the residual gap of every variant is
one: the preference for the most connected nodes is underestimated when transferring from 25.714
users to 10.000. This script checks that claim with two measurements that do not need the
synthetic network to have a ground truth:

1. INTERNAL SCALING OF THE REAL. The real network has nine snapshots, so nine pairs
   (users, maximum out-degree) that trace how the hub grows with the population. A straight line
   in log-log gives the expected maximum degree at any population size, in particular at the
   10.000 users of the synthetic dataset. If a generated network reaches that value, its tail is
   NOT underestimated: it is what the real's own scaling law predicts at that size.

2. TAIL SHAPE AT MATCHED SIZE. Comparing the real (25.714 users) with a generated network (about
   10.000) compares different scales. The fair comparison is the real's FIRST snapshot (10.755
   users) against the generated one: same top-decile CCDF slope means the same relative shape.

3. THE HUB RATIO AGAINST THE POPULATION RATIO. The mean degree of the already-present endpoint is
   169 in V9 and 437 in the real: 169/437 = 0.39. The population ratio is 10.000/25.714 = 0.39.
   If the two coincide, the "underestimation" is the population factor itself, i.e. the model
   transfers the relative position faithfully and there is nothing to fix.

Usage: uv run python src/tail_scaling.py
Output: results/tail_scaling_real.csv, results/tail_scaling_summary.csv
"""

import csv
import os

import numpy as np

from graph_io import load_network

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'Esame Angelantonio Fedele Murolo')
RESULTS = os.path.join(ROOT, 'results')

FOLDERS = [
    ('synthetic_network', 'ufficiale'),
    ('synthetic_control_network', 'controllo'),
]
QUANTILE = 0.90


def out_degrees(edges):
    counts = {}
    for source, _ in edges:
        counts[source] = counts.get(source, 0) + 1
    return np.sort(np.array(list(counts.values()), dtype=float))


def tail_slope(values, quantile=QUANTILE):
    """(slope of the top-decile CCDF, threshold, points used)."""
    if values.size < 20:
        return float('nan'), float('nan'), 0
    ccdf = 1.0 - np.arange(values.size) / values.size
    threshold = float(np.quantile(values, quantile))
    mask = values >= threshold
    return (float(np.polyfit(np.log(values[mask]), np.log(ccdf[mask]), 1)[0]),
            threshold, int(mask.sum()))


def snapshot_row(name, edges, nodes):
    values = out_degrees(edges)
    slope, threshold, points = tail_slope(values)
    return {
        'snapshot': name,
        'nodes': len(nodes),
        'edges': len(edges),
        'mean_out_degree': round(float(values.mean()), 3),
        'max_out_degree': int(values.max()),
        'tail_slope': round(slope, 3),
        'tail_gamma': round(1.0 - slope, 2),
        'tail_from': round(threshold, 1),
        'tail_points': points,
    }


def last_row(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def main():
    real = load_network(os.path.join(BASE, 'real_snapshots'))
    real_rows = [snapshot_row(name, edges, nodes) for name, edges, nodes in real]

    with open(os.path.join(RESULTS, 'tail_scaling_real.csv'), 'w', encoding='utf-8',
              newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(real_rows[0]))
        writer.writeheader()
        writer.writerows(real_rows)

    sizes = np.array([row['nodes'] for row in real_rows], dtype=float)
    maxima = np.array([row['max_out_degree'] for row in real_rows], dtype=float)
    slope, intercept = np.polyfit(np.log(sizes), np.log(maxima), 1)
    fitted = intercept + slope * np.log(sizes)
    residual = float(np.sum((np.log(maxima) - fitted) ** 2))
    total = float(np.sum((np.log(maxima) - np.log(maxima).mean()) ** 2))
    r2 = 1.0 - residual / total

    print('legge di scala interna del reale (grado massimo contro numero di utenti):')
    print(f'  esponente {slope:.3f} | R2 {r2:.3f} | estremi: {int(sizes[0])} utenti -> '
          f'{int(maxima[0])} e {int(sizes[-1])} -> {int(maxima[-1])}')
    for row in real_rows:
        print(f"  {row['snapshot']:>20}: {row['nodes']:>6} utenti | grado massimo "
              f"{row['max_out_degree']:>5} | pendenza di coda {row['tail_slope']:+.3f}")

    small = real_rows[0]
    print()
    print(f"confronto a pari dimensione: reale al primo snapshot ({small['nodes']} utenti, "
          f"pendenza {small['tail_slope']:+.3f}) contro le reti generate (~10.000 utenti)")

    rows = []
    for folder, label in FOLDERS:
        name, edges, nodes = load_network(os.path.join(BASE, folder))[-1]
        row = snapshot_row(name, edges, nodes)
        expected = float(np.exp(intercept + slope * np.log(row['nodes'])))
        accretion = last_row(os.path.join(RESULTS, f'accretion_{folder}.csv'))
        rows.append({
            'network': label,
            'nodes': row['nodes'],
            'max_out_degree': row['max_out_degree'],
            'expected_max': int(round(expected)),
            'ratio_to_expected': round(row['max_out_degree'] / expected, 3),
            'tail_slope': row['tail_slope'],
            'tail_gamma': row['tail_gamma'],
            'existing_endpoint_degree': accretion.get('mean_prev_degree_of_existing_end', ''),
        })
        print(f"  {label:>8}: {row['nodes']:>6} utenti | grado massimo {row['max_out_degree']:>5} "
              f"| atteso dalla scala {expected:>6.0f} | rapporto {row['max_out_degree'] / expected:>5.2f} "
              f"| pendenza {row['tail_slope']:+.3f}")

    real_accretion = last_row(os.path.join(RESULTS, 'accretion_real_snapshots.csv'))
    real_hub = real_accretion.get('mean_prev_degree_of_existing_end', '')
    with open(os.path.join(RESULTS, 'tail_scaling_summary.csv'), 'w', encoding='utf-8',
              newline='') as f:
        fields = ['network'] + [key for key in rows[0] if key != 'network'] + \
                 ['real_existing_endpoint_degree']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, real_existing_endpoint_degree=real_hub))

    if real_hub:
        print()
        print('il grado medio dell\'estremo gia\' presente, contro il rapporto fra le popolazioni:')
        for row in rows:
            value = row['existing_endpoint_degree'] or float('nan')
            ratio = float(value) / float(real_hub) if value else float('nan')
            print(f"  {row['network']:>8}: {float(value):>6.1f} contro {float(real_hub):.1f} del reale "
                  f"| rapporto {ratio:.3f} | rapporto fra popolazioni "
                  f"{row['nodes'] / sizes[-1]:.3f}")
    print()
    print('written results/tail_scaling_real.csv and results/tail_scaling_summary.csv')


if __name__ == '__main__':
    main()
