"""Summary table of the network variants (encoders of lever 2 and rules of lever 3).

Reads the results CSVs already produced by the analyses and writes
results/variant_summary.csv: one row per dataset folder, with the real AUC of the model that
generated it (when available), the final-snapshot structure and the content homophily.

Usage: uv run python src/variant_summary.py
Output: results/variant_summary.csv, results/figures/variant_summary.png
"""

import csv
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(RESULTS, 'figures')


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def last_row(rows):
    return rows[-1] if rows else {}


def main():
    folders = sorted(os.path.basename(path)[len('structure_'):-len('.csv')]
                     for path in glob.glob(os.path.join(RESULTS, 'structure_*.csv'))
                     if 'heavy' not in os.path.basename(path))
    transfers = {(row['partition'], row['encoder']): row for row in
                 read_csv(os.path.join(RESULTS, 'transfer_report.csv'))}
    rules = {row['variant']: row for row in read_csv(os.path.join(RESULTS, 'edge_rules_report.csv'))}
    if rules:
        variant_of = {f'generated_snapshots_rule_{name}': name for name in rules}
    else:
        variant_of = {}

    rows = []
    for folder in folders:
        structure = last_row(read_csv(os.path.join(RESULTS, f'structure_{folder}.csv')))
        heavy = last_row(read_csv(os.path.join(RESULTS, f'structure_heavy_{folder}.csv')))
        content = last_row(read_csv(os.path.join(RESULTS, f'content_{folder}.csv')))
        transfer = next((row for (partition, _), row in transfers.items() if partition == folder), {})
        rule = rules.get(variant_of.get(folder, ''), {})
        rows.append({
            'folder': folder,
            'kind': ('real' if folder == 'real_snapshots' else
                     ('encoder ' + transfer.get('encoder', '?')) if transfer else
                     ('rule ' + variant_of.get(folder, '?'))),
            'real_val_auc': transfer.get('real_val_auc', ''),
            'real_test1_auc': transfer.get('real_test1_auc', ''),
            'real_test2_auc': transfer.get('real_test2_auc', ''),
            'nodes': structure.get('nodes', ''),
            'edges': structure.get('edges', ''),
            'density': structure.get('density', ''),
            'mean_out_degree': structure.get('out_degree_mean', ''),
            'max_out_degree': structure.get('out_degree_max', ''),
            'reciprocity': structure.get('reciprocity', ''),
            'modularity': structure.get('modularity', ''),
            'transitivity': heavy.get('transitivity', ''),
            'avg_clustering': heavy.get('avg_clustering', ''),
            'homophily_ratio': content.get('homophily_ratio', ''),
        })

    with open(os.path.join(RESULTS, 'variant_summary.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = (f'{"folder":<44}{"nodi":>7}{"archi":>9}{"densita":>9}{"recipr":>8}'
              f'{"modul":>8}{"transit":>9}{"omofilia":>10}')
    print(header)
    for row in rows:
        print(f'{row["folder"]:<44}{row["nodes"]:>7}{row["edges"]:>9}'
              f'{float(row["density"]):>9.5f}{float(row["reciprocity"]):>8.4f}'
              f'{float(row["modularity"]):>8.3f}{float(row["transitivity"]):>9.3f}'
              f'{float(row["homophily_ratio"]):>10.3f}' if row['density'] else row['folder'])

    os.makedirs(FIGURES, exist_ok=True)
    usable = [row for row in rows if row['density']]
    labels = [row['folder'].replace('generated_snapshots', 'gen').replace('real_snapshots', 'reale')
              for row in usable]
    metrics = [('reciprocity', 'reciprocità'), ('modularity', 'modularità'),
               ('transitivity', 'transitivity'), ('homophily_ratio', 'omofilia')]
    figure, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        values = [float(row[key]) if row[key] else float('nan') for row in usable]
        ax.bar(range(len(labels)), values, color=['tab:blue' if row['kind'] == 'real' else
                                                  'tab:orange' for row in usable])
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.grid(alpha=0.3, axis='y')
    figure.tight_layout()
    figure.savefig(os.path.join(FIGURES, 'variant_summary.png'))
    plt.close(figure)
    print(f'written results/variant_summary.csv and results/figures/variant_summary.png')


if __name__ == '__main__':
    main()
