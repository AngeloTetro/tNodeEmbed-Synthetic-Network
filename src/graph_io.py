"""Shared readers for the datasets (real and generated)."""

import csv
import os

N_SNAPSHOTS = 9


def norm(value):
    value = value.strip()
    if not value:
        return None
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def load_edges(path):
    """Edges from a .edg file. Self-loops are dropped: they are a data artefact present in
    the real dataset and several networkx algorithms refuse them."""
    edges = set()
    if not os.path.exists(path):
        return edges
    with open(path, encoding='utf-8-sig', newline='') as f:
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                source, target = norm(parts[0]), norm(parts[1])
                if source != target:
                    edges.add((source, target))
    return edges


def load_docs(path):
    """{user: [text, ...]} from a posts file (current or incremental)."""
    docs = {}
    if not os.path.exists(path):
        return docs
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = [h.strip() for h in next(reader)]
        user_col = header.index('account_id') if 'account_id' in header else 0
        text_col = header.index('content') if 'content' in header else header.index('text')
        for row in reader:
            if len(row) != len(header):
                continue
            user, text = norm(row[user_col]), row[text_col].strip()
            if user and text:
                docs.setdefault(user, []).append(text)
    return docs


def load_attributes(path, attributes):
    """{attribute: {user: value}} from a posts file (profiles, synthetic only)."""
    table = {attribute: {} for attribute in attributes}
    if not os.path.exists(path):
        return table
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = [h.strip() for h in next(reader)]
        user_col = header.index('account_id') if 'account_id' in header else 0
        columns = {a: header.index(a) for a in attributes if a in header}
        for row in reader:
            if len(row) != len(header):
                continue
            user = norm(row[user_col])
            for attribute, column in columns.items():
                value = row[column].strip()
                if value and user not in table[attribute]:
                    table[attribute][user] = value
    return table


def snapshots_folder(folder):
    """Ordered snapshot directories of a dataset folder."""
    names = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
    return [os.path.join(folder, name) for name in names], names


def load_network(folder):
    """[(name, edges, nodes), ...] for every snapshot of the folder."""
    directories, names = snapshots_folder(folder)
    out = []
    for directory, name in zip(directories, names):
        edges = load_edges(os.path.join(directory, 'social_network.edg'))
        nodes = {n for edge in edges for n in edge}
        out.append((name, edges, nodes))
    return out


def cumulative_docs(folder):
    directories, names = snapshots_folder(folder)
    return names, [load_docs(os.path.join(d, 'posts_incremental.tsv')) for d in directories]
