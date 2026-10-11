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


VARIANTS = [
    ('_rule_v1', 'v1 soglia sul punteggio'),
    ('_rule_v2', 'v2 quota reale di archi'),
    ('_rule_v3', 'v3 reciprocita calibrata'),
    ('_rule_v4', 'v4 accrescimento'),
    ('_rule_v5', 'v5 accrescimento + reciprocita'),
    ('_v6a', 'v6a soglia sul grado finale'),
    ('_v6b', 'v6b soglia sulla crescita'),
    ('_v7a', 'v7a inferenza, decisione bayesiana'),
    ('_v7b', 'v7b inferenza, tasso reale'),
    ('_v7c', 'v7c inferenza, soglia globale'),
    ('_v7d', 'v7d inferenza, encoder fine-tuned'),
    ('_v7e', 'v7e inferenza, globale + fine-tuned'),
    ('_v8_nocontent', 'v8 ablazione (senza contenuto)'),
    ('_v8_real', 'v8 controllo A (testo autentico)'),
    ('_v8_permuted', 'v8 controllo B (testo permutato)'),
    ('_v8_off', 'v8 controllo C (senza testo)'),
    ('_v8', 'v8 modello struttura (campionamento)'),
]


def variant_label(folder):
    if folder == 'real_snapshots':
        return 'REALE'
    for suffix, label in VARIANTS:
        if folder.endswith(suffix):
            return label
    return 'riferimento (solo contenuto)'


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
            'kind': variant_label(folder),
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

    def fmt(row, key, width, decimals):
        text = row.get(key)
        if text in (None, ''):
            return f'{"-":>{width}}'
        return f'{float(text):>{width}.{decimals}f}'

    for row in rows:
        if not row['density']:
            print(f'{row["folder"]:<44}{"(nessun arco)":>12}')
            continue
        print(f'{row["folder"]:<44}{row["nodes"]:>7}{row["edges"]:>9}'
              f'{fmt(row, "density", 9, 5)}{fmt(row, "reciprocity", 8, 4)}'
              f'{fmt(row, "modularity", 8, 3)}{fmt(row, "transitivity", 9, 3)}'
              f'{fmt(row, "homophily_ratio", 10, 3)}')

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
