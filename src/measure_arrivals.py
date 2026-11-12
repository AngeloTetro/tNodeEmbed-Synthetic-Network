"""Misura gli eventi di arrivo sul reale: quanti archi porta ciascun evento, la direzione e la quota
di coppie ricambiate, per transizione. E' la misura da cui il generatore prende la sua unita' di generazione, e
serve a rendere citabili i numeri che il paper riporta.

Usage: uv run python src/measure_arrivals.py
Output: results/arrival_events.csv
"""

import csv
import os

import numpy as np

from generate_arrival import arrivals, TRAIN_STEPS
from transfer_to_synthetic import real_representations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')


def main():
    _, _, index, edge_sets, _, _ = real_representations('minilm', 'period', 'mean_of_posts')
    rows = []
    for step in TRAIN_STEPS:
        events = arrivals(edge_sets, index, step)
        sizes = np.array([len(edges) for edges in events.values()])
        outgoing = np.mean([sum(1 for direction, _ in edges if direction > 0.5) / len(edges)
                            for edges in events.values()])
        rows.append({'transition': step,
                     'arrival_events': int(sizes.size),
                     'new_edges': int(sizes.sum()),
                     'edges_per_event_mean': round(float(sizes.mean()), 2),
                     'edges_per_event_median': float(np.median(sizes)),
                     'edges_per_event_max': int(sizes.max()),
                     'outgoing_share': round(float(outgoing), 4)})
        print(f'transizione {step}: eventi {rows[-1]["arrival_events"]} | archi '
              f'{rows[-1]["new_edges"]} | media {rows[-1]["edges_per_event_mean"]} | mediana '
              f'{rows[-1]["edges_per_event_median"]:.0f} | massimo '
              f'{rows[-1]["edges_per_event_max"]} | uscenti {rows[-1]["outgoing_share"]}')
    total = np.array([row['new_edges'] for row in rows]).sum()
    mean = np.array([row['new_edges'] / max(1, row['arrival_events']) for row in rows])
    print(f'totale archi nuovi {total} | media per evento fra {mean.min():.0f} e {mean.max():.0f}')
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, 'arrival_events.csv')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({'transition': 'all', 'arrival_events': '', 'new_edges': int(total),
                         'edges_per_event_mean': '', 'edges_per_event_median': '',
                         'edges_per_event_max': '', 'outgoing_share': ''})
    print(f'written {path}')


if __name__ == '__main__':
    main()
