"""Measures the arrival events of the real network: how many edges each event carries, the direction
and the share of reciprocated pairs, per transition. This is the measurement the generator takes its
unit of generation from, and it makes the numbers quoted in the document citable.

Usage: uv run python -m src.analysis.measure_arrivals
Output: results/arrival_events.csv
"""

import csv
import os

import numpy as np

from src.pipeline.generate_arrival import arrivals, TRAIN_STEPS

from src.paths import ROOT, TABLES
from src.lib.transfer_to_synthetic import real_representations



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
        print(f'transition {step}: events {rows[-1]["arrival_events"]} | edges '
              f'{rows[-1]["new_edges"]} | media {rows[-1]["edges_per_event_mean"]} | mediana '
              f'{rows[-1]["edges_per_event_median"]:.0f} | massimo '
              f'{rows[-1]["edges_per_event_max"]} | uscenti {rows[-1]["outgoing_share"]}')
    total = np.array([row['new_edges'] for row in rows]).sum()
    mean = np.array([row['new_edges'] / max(1, row['arrival_events']) for row in rows])
    print(f'total new edges {total} | mean per event between {mean.min():.0f} and {mean.max():.0f}')
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, 'arrival_events.csv')
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
