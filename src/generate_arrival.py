"""V9 -- il meccanismo di arrivo: l'unita' di formazione degli archi e' l'evento, non la coppia.

Diagnosi (FINAL.md sezione 10, docs/pipeline.md): V8 riproduce densita' e volume ma sbaglia le
proprieta' relazionali -- reciprocita' 0,004 contro 0,418, transitivita' 0,003 contro 0,034, grado
massimo 102 contro 1.290 -- perche' decide ogni coppia in modo indipendente e addestra su negativi
uniformi. Nel reale quei tre gap sono un solo fenomeno: l'arrivo di un utente nuovo. Ogni arco nuovo
tocca un entrante, il 40-44% arriva con il proprio inverso nella stessa transizione, e la coda dei
gradi nasce perche' gli arrivi scelgono i pochi nodi gia' molto connessi.

V9 stima sul reale, per ogni evento di arrivo (un nodo che entra nella rete in quella transizione):
    1. quanti archi nuovi porta, con la direzione. Il conteggio si fitta **riscalato alla
       popolazione** (k * popolazione_di_riferimento / popolazione_della_transizione), perche' k
       cresce con la dimensione della rete: e' lo stesso accorgimento dei percentili di grado.
    2. dove vanno: un modello di scelta sui candidati -- grado, percentile, "e' nuovo", attivita',
       punteggio di contenuto di tNodeEmbed, direzione, e **co-occorrenza** ("la controparte e' vicina
       di un'altra controparte dello stesso evento", cioe' se gli archi dell'evento si ammassano o si
       evitano). I pesi sono il risultato scientifico.
    3. la bidirezionalita': con che probabilita' un arco entrante riprende un partner gia' scelto fra
       gli uscenti. Nel reale i k archi dell'evento includono gia' le coppie ricambiate, quindi la
       reciprocita' non si aggiunge dopo: si distribuisce dentro il conteggio.

La generazione non impone conteggi: si campionano gli archi dell'evento uno per uno sul punteggio di
scelta (Gumbel), riscalando k sulla popolazione sintetica corrente, poi si aggiorna lo stato.

Content modes (`--content`), solo sul lato sintetico; il modello e' sempre fittato sul reale:
    real      profili testuali autentici;
    permuted  gli stessi profili riassegnati ad altri utenti (una sola permutazione per tutti i passi);
    off       canale testuale azzerato.

Usage:
    uv run python src/generate_arrival.py --partition generated_snapshots_v2 --content real
    uv run python src/generate_arrival.py --partition generated_snapshots_v2 --content permuted
    uv run python src/generate_arrival.py --partition generated_snapshots_v2 --content off
Output: <partition>_<variant>/, results/v9_report.csv, results/v9_coefficients.csv,
        results/_v9_run.txt
"""

import argparse
import csv
import os
import traceback

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from apply_edge_rules import edges_to_files, metrics_from_folder
from transfer_to_synthetic import (BASE, DEVICE, N_SNAPSHOTS, SEED, node_embeddings,
                                   prepare_pairs, procrustes_chain, real_representations,
                                   synthetic_representations, train_model)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
REPORT = os.path.join(RESULTS, 'v9_report.csv')
LOG = os.path.join(RESULTS, '_v9_run.txt')


def coefficients_path(variant):
    """One coefficient file per run: the generic name was overwritten by every run."""
    return os.path.join(RESULTS, f'v9_{variant}_coefficients.csv')


TRAIN_STEPS = (1, 2, 3, 4, 5)
VAL_STEP = 6
NEG_PER_POS = 10
MAX_EVENTS = 400
MAX_EVENT = 5000
MAX_COUNT = 1_000_000
SMALL = 1e-3
FEATURES = ['content_score', 'partner_degree_log', 'partner_percentile', 'partner_new',
            'partner_active', 'outgoing', 'partner_co_occurrence']
FIELDS = ['variant', 'partition', 'content', 'operating_point', 'val_auc', 'events',
          'edges_per_event', 'reverse_share', 'nodes', 'edges', 'density', 'reciprocity',
          'mean_out_degree', 'max_out_degree']


def log(text):
    print(text, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def arrivals(edge_sets, index, step):
    """Eventi di arrivo della transizione `step`: ogni arco nuovo appartiene all'evento del suo
    estremo nuovo (se sono nuovi entrambi, vince la sorgente), con la direzione rispetto a lui.

    Gli archi si iterano in ordine deterministico (`sorted`): l'ordine di iterazione di un set di
    stringhe dipende dall'hash del processo e cambierebbe la sequenza casuale a valle, rendendo le
    ripetizioni diverse fra loro.
    """
    previous, current = edge_sets[step - 1], edge_sets[step]
    before = {u for edge in previous for u in edge}
    events = {}
    for source, target in sorted(current - previous):
        if source in index and source not in before:
            owner, direction, partner = source, 1.0, target
        elif target in index and target not in before:
            owner, direction, partner = target, 0.0, source
        else:
            continue
        events.setdefault(owner, []).append((direction, partner))
    return events


def population(edge_sets, step):
    return len({u for edge in edge_sets[step] for u in edge})


def fit_counts(edge_sets, index, steps, exponent):
    """Tabella di k riscalato alla popolazione: k / P^beta, con beta e unita' fittati sul reale.

    k cresce con la popolazione della rete piu' che linearmente: l'esponente beta e' la pendenza
    log-log di k medio contro popolazione. La tabella contiene il valore normalizzato, moltiplicato
    per un'unita' che ne conserva la risoluzione (mediana a 1000); in generazione si riscala.
    """
    events_by_step = {step: arrivals(edge_sets, index, step) for step in steps}
    scales = {step: population(edge_sets, step - 1) ** exponent for step in steps}
    normalised = [len(edges) / scales[step]
                  for step in steps for edges in events_by_step[step].values()]
    unit = 1000.0 / max(float(np.median(normalised)), 1e-12)
    tables, shares, sizes = {}, {}, {}
    for step in steps:
        counts, outgoing = [], []
        for edges in events_by_step[step].values():
            counts.append(min(int(round(len(edges) / scales[step] * unit)), MAX_COUNT))
            outgoing.append(sum(1 for direction, _ in edges if direction > 0.5) / len(edges))
        values = np.asarray(counts or [1], dtype=np.int64)
        tables[step] = np.bincount(values, minlength=int(values.max()) + 2).astype(np.float64) + SMALL
        shares[step] = float(np.mean(outgoing)) if outgoing else 0.5
        sizes[step] = len(counts)
    return tables, shares, sizes, unit


def population_exponent(edge_sets, index, steps):
    """Esponente beta di k ~ P^beta, fittato sulle transizioni di training (pendenza log-log)."""
    sizes, means = [], []
    for step in steps:
        events = arrivals(edge_sets, index, step)
        if not events:
            continue
        sizes.append(population(edge_sets, step - 1))
        means.append(float(np.mean([len(edges) for edges in events.values()])))
    if len(sizes) < 2 or min(means) <= 0:
        return 1.0
    return float(np.polyfit(np.log(sizes), np.log(means), 1)[0])


def reciprocity_table(edge_sets, index, steps):
    """P(arco inverso nella stessa transizione | arco nuovo), per direzione e classe di grado."""
    boxes = {}
    for step in steps:
        previous = edge_sets[step - 1]
        before = {u for edge in previous for u in edge}
        new_edges = edge_sets[step] - previous
        degree = {}
        for edge in previous:
            for user in edge:
                degree[user] = degree.get(user, 0) + 1
        for source, target in new_edges:
            reverse = (target, source) in new_edges
            for owner, direction, partner in ((source, 1.0, target), (target, 0.0, source)):
                if owner in index and owner not in before:
                    value = degree.get(partner, 0)
                    bucket = 0 if value == 0 else (1 if value < 50 else 2)
                    box = boxes.setdefault((direction, bucket), [0, 0])
                    box[0] += 1
                    if reverse:
                        box[1] += 1
    return {key: (hits + 1.0) / (total + 2.0) for key, (total, hits) in boxes.items()}


def state_arrays(edge_sets, index, masks, users, step):
    """Cio' che nel fit e' calcolato sullo stato al passo precedente."""
    previous = edge_sets[step - 1]
    before = {u for edge in previous for u in edge if u in index}
    degree = {}
    for edge in previous:
        for user in edge:
            degree[user] = degree.get(user, 0) + 1
    present = np.array(sorted(index[u] for u in before), dtype=np.int64)
    n = len(users)
    degree_log = np.zeros(n, dtype=np.float32)
    percentile = np.zeros(n, dtype=np.float32)
    is_new = np.ones(n, dtype=np.float32)
    active = masks[step].astype(np.float32) if masks is not None else np.zeros(n, dtype=np.float32)
    if present.size:
        values = np.array([degree.get(users[i], 0) for i in present], dtype=np.float32)
        degree_log[present] = np.log1p(values)
        if present.size > 1:
            percentile[present] = values.argsort().argsort() / (present.size - 1)
        is_new[present] = 0.0
    return present, degree_log, percentile, is_new, active


def choice_features(scores, degree_log, percentile, is_new, active, targets, directions, adjacent):
    return np.column_stack([
        scores,
        degree_log[targets],
        percentile[targets],
        is_new[targets],
        active[targets],
        directions,
        adjacent,
    ]).astype(np.float32)


def content_scores(model, embeddings, owners, targets, with_content):
    if not with_content or len(owners) == 0:
        return np.zeros(len(owners), dtype=np.float32)
    out = []
    with torch.no_grad():
        for start in range(0, len(owners), 200_000):
            i = torch.from_numpy(np.asarray(owners[start:start + 200_000], dtype=np.int64)).to(DEVICE)
            j = torch.from_numpy(np.asarray(targets[start:start + 200_000], dtype=np.int64)).to(DEVICE)
            pair = torch.cat([embeddings[i], embeddings[j], embeddings[i] * embeddings[j]], dim=1)
            out.append(model.head(pair).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def neighbours_of(edge_sets, step, index):
    """Vicinato (entrambe le direzioni) nella rete della transizione, indicizzato come `index`.

    Anche qui l'iterazione e' ordinata, per lo stesso motivo di `arrivals`.
    """
    out = {}
    for a, b in sorted(edge_sets[step]):
        i, j = index.get(a), index.get(b)
        if i is None or j is None:
            continue
        out.setdefault(i, set()).add(j)
        out.setdefault(j, set()).add(i)
    return out


def closure_value(partners, neighbours_of_candidate, mode):
    """Chiusura triadica dentro l'evento: 'binary' (almeno una) o 'count' (la frazione)."""
    if not partners or not neighbours_of_candidate:
        return 0.0
    shared = len(partners & neighbours_of_candidate)
    if mode == 'binary':
        return 1.0 if shared else 0.0
    return shared / len(partners)


def build_choice_dataset(model, embeddings, users, index, edge_sets, masks, rng, with_content,
                         steps, stratified=False, closure='binary'):
    """Positivi = archi reali degli eventi; negativi = candidati dello stesso evento.

    La co-occorrenza vale 1 se la controparte e' vicina di un'altra controparte dello stesso evento:
    e' il termine che dice se gli archi dell'evento si ammassano sugli stessi intorni o li evitano.

    Con `stratified=True` i negativi diventano misti (proposta di Gemini): 30% inversioni
    asimmetriche, cioe' la coppia al contrario di un positivo ma solo se quell'arco inverso non
    esiste davvero, 30% difficili sul grado (candidati del decile piu' connesso del pool) e 40%
    uniformi come nella configurazione documentata. La parte semantica dei negativi difficili non
    c'e': scegliere la controparte piu' simile nel testo costerebbe una valutazione del modello per
    candidato e il canale testuale non porta il legame (controllo A/B/C).

    `closure` dice come si misura la chiusura triadica dentro l'evento: 'binary' (la variabile
    documentata, il candidato e' vicino ad almeno un'altra controparte dell'evento) o 'count' (la
    frazione delle altre controparti a cui il candidato e' vicino). La stessa funzione deve valere
    in fit e in generazione: cambiarla da un lato solo renderebbe il coefficiente privo di senso.
    """
    features, labels = [], []
    for step in steps:
        events = arrivals(edge_sets, index, step)
        present, degree_log, percentile, is_new, active = state_arrays(edge_sets, index, masks,
                                                                      users, step)
        if present.size == 0 or not events:
            continue
        # Negatives must come from the same pool the generation samples from (the users already
        # present plus the newcomers of this step): otherwise "the partner is new" separates
        # positives from negatives by construction and the fitted weight is meaningless.
        newcomers = np.array(sorted(index[owner] for owner in events if owner in index),
                             dtype=np.int64)
        pool = np.union1d(present, newcomers)
        if pool.size == 0:
            continue
        values = degree_log[pool]
        percentile[pool] = (values.argsort().argsort() / max(1, pool.size - 1)).astype(np.float32)
        neighbours = neighbours_of(edge_sets, step, index)
        directed = {(index[a], index[b]) for a, b in edge_sets[step] if a in index and b in index}
        hard_degree = pool[np.argsort(-degree_log[pool])[:max(1, pool.size // 10)]]
        chosen = list(events)
        if len(chosen) > MAX_EVENTS:
            chosen = list(rng.choice(chosen, size=MAX_EVENTS, replace=False))
        owners, targets, directions, tags, adjacent = [], [], [], [], []
        for owner in chosen:
            edges = events[owner]
            partners = [index[partner] for _, partner in edges if partner in index]
            if not partners:
                continue
            partner_set = {index[partner] for _, partner in edges if partner in index}
            inversions = [(index[partner], 1.0 - direction) for direction, partner in edges
                          if partner in index
                          and (index[partner], index[owner]) not in directed]
            for direction, partner in edges:
                if partner not in index:
                    continue
                owners.append(index[owner])
                targets.append(index[partner])
                directions.append(direction)
                tags.append(1.0)
                others = partner_set - {index[partner]}
                adjacent.append(closure_value(others, neighbours.get(index[partner], set()), closure))
            for _ in range(NEG_PER_POS * len(partners)):
                # Il percorso uniforme non estrae numeri in piu': con `--negatives uniform` la
                # sequenza casuale (e quindi la rete generata) resta quella documentata.
                if stratified and inversions and rng.random() < 0.30:
                    candidate, direction = inversions[int(rng.integers(0, len(inversions)))]
                elif stratified and rng.random() < 3.0 / 7.0:
                    candidate = int(hard_degree[int(rng.integers(0, hard_degree.size))])
                    direction = 1.0 if rng.random() < 0.5 else 0.0
                    if candidate in partner_set:
                        candidate = int(pool[rng.integers(0, pool.size)])
                else:
                    candidate = int(pool[rng.integers(0, pool.size)])
                    direction = 1.0 if rng.random() < 0.5 else 0.0
                owners.append(index[owner])
                targets.append(candidate)
                directions.append(direction)
                tags.append(0.0)
                adjacent.append(closure_value(partner_set, neighbours.get(candidate, set()), closure))
        if not owners:
            continue
        scores = content_scores(model, embeddings[step], owners, targets, with_content)
        features.append(choice_features(scores, degree_log, percentile, is_new, active,
                                        np.asarray(targets), np.asarray(directions, dtype=np.float32),
                                        np.asarray(adjacent, dtype=np.float32)))
        labels.append(np.asarray(tags, dtype=np.float32))
        log(f'    passo {step}: {len(owners)} righe ({int(np.sum(tags))} positivi)')
    return np.vstack(features), np.concatenate(labels)


def sample_counts(table, share, population_now, exponent, unit, rng):
    """k normalizzato campionato, riscalato sulla popolazione corrente, poi la quota uscente."""
    probability = table / table.sum()
    scaled = int(rng.choice(table.size, p=probability))
    total = int(round(scaled / unit * population_now ** exponent))
    if total <= 0:
        return 0, 0
    return total, int(rng.binomial(total, float(np.clip(share, 0.0, 1.0))))


def head_scores(model, hidden, owner, candidates):
    with torch.no_grad():
        left = hidden[torch.from_numpy(np.full(candidates.size, owner, dtype=np.int64)).to(DEVICE)]
        right = hidden[torch.from_numpy(candidates.astype(np.int64)).to(DEVICE)]
        pair = torch.cat([left, right, left * right], dim=1)
        return model.head(pair).squeeze(1).cpu().numpy()


def generate(model, embeddings, users, masks, present_masks, scaler, classifier, tables, shares,
             reverse_rate, exponent, unit, with_content, folder, partition_name, rng,
             closure='binary'):
    """Genera per eventi di arrivo, con lo stato ricalcolato a ogni passo."""
    n = len(users)
    adj = torch.zeros((n, n), dtype=torch.bool, device=DEVICE)
    previous_present = np.zeros(n, dtype=bool)
    events_total, edges_total, reversed_total = 0, 0, 0
    train_steps = sorted(tables)
    for step in range(N_SNAPSHOTS):
        active = np.asarray(masks[step], dtype=bool)
        present = np.asarray(present_masks[step], dtype=bool)
        candidates = np.nonzero(present)[0]
        newcomers = np.nonzero(present & ~previous_present)[0]
        table_key = train_steps[min(step, len(train_steps) - 1)]
        population_now = int(previous_present.sum()) or int(present.sum())
        if candidates.size and newcomers.size:
            degrees = (adj.sum(dim=0) + adj.sum(dim=1)).cpu().numpy()
            values = degrees[candidates].astype(np.float32)
            percentile = np.zeros(n, dtype=np.float32)
            if candidates.size > 1:
                percentile[candidates] = values.argsort().argsort() / (candidates.size - 1)
            degree_log = np.log1p(degrees).astype(np.float32)
            is_new = (~previous_present).astype(np.float32)
            active_flag = active.astype(np.float32)
            for owner in newcomers:
                total, outgoing = sample_counts(tables[table_key], shares[table_key], population_now,
                                                exponent, unit, rng)
                if total <= 0:
                    continue
                if with_content:
                    content = head_scores(model, embeddings[step], int(owner), candidates)
                else:
                    content = np.zeros(candidates.size, dtype=np.float32)
                out_linked = adj[owner].cpu().numpy()[candidates].copy()
                in_linked = adj[:, owner].cpu().numpy()[candidates].copy()
                near_chosen = np.zeros(candidates.size, dtype=np.float32)
                chosen_all = set()
                out_linked[candidates == owner] = True
                in_linked[candidates == owner] = True
                chosen = []
                for position in range(min(total, MAX_EVENT)):
                    direction = 1.0 if position < outgoing else 0.0
                    if direction < 0.5 and chosen and rng.random() < reverse_rate:
                        partner = int(chosen[rng.integers(0, len(chosen))])
                        place = int(np.searchsorted(candidates, partner))
                        if in_linked[place]:
                            continue
                    else:
                        allowed = ~(out_linked if direction > 0.5 else in_linked)
                        if not allowed.any():
                            break
                        closure_now = (near_chosen if closure == 'binary'
                                       else near_chosen / max(1, len(chosen_all)))
                        matrix = choice_features(
                            content, degree_log, percentile, is_new, active_flag, candidates,
                            np.full(candidates.size, direction, dtype=np.float32), closure_now)
                        weight = classifier.decision_function(scaler.transform(matrix))
                        key = np.where(allowed, weight + rng.gumbel(size=weight.size), -np.inf)
                        partner = int(candidates[int(np.argmax(key))])
                    source, target = (int(owner), partner) if direction > 0.5 else (partner, int(owner))
                    if not bool(adj[source, target]):
                        adj[source, target] = True
                        edges_total += 1
                        if direction < 0.5:
                            reversed_total += 1
                    if direction > 0.5:
                        out_linked[candidates == partner] = True
                        chosen.append(partner)
                    else:
                        in_linked[candidates == partner] = True
                    if partner not in chosen_all:
                        chosen_all.add(partner)
                        picks = np.nonzero((adj[partner].cpu().numpy()[candidates])
                                           | (adj[:, partner].cpu().numpy()[candidates]))[0]
                        if closure == 'binary':
                            near_chosen[picks] = 1.0
                        else:
                            near_chosen[picks] += 1.0
                events_total += 1
        edges_to_files(adj, users, folder, step, partition_name)
        previous_present = present
        log(f'    passo {step + 1}: nuovi {int(newcomers.size)} | popolazione {population_now} '
            f'| archi finora {int(adj.sum().item())}')
    return events_total, edges_total, reversed_total


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', default='generated_snapshots_v2')
    parser.add_argument('--pooling', default='mean_of_posts')
    parser.add_argument('--variant', default='v9')
    parser.add_argument('--content', choices=('real', 'permuted', 'off'), default='real')
    parser.add_argument('--exponent', type=float, default=1.0,
                        help='esponente della riscalatura di k sulla popolazione: 1 = lineare, '
                             'il valore che porta il grado medio generato su quello reale')
    parser.add_argument('--steps', type=int, default=N_SNAPSHOTS)
    parser.add_argument('--negatives', choices=('uniform', 'stratified'), default='uniform',
                        help='uniform: la configurazione documentata; stratified: misti 30/30/40')
    parser.add_argument('--closure', choices=('binary', 'count'), default='binary',
                        help='chiusura triadica dentro l\'evento: binaria (documentata) o frazione '
                             'delle altre controparti')
    args = parser.parse_args()
    with_content = args.content != 'off'
    stratified = args.negatives == 'stratified'
    closure = args.closure
    open(LOG, 'w').close()
    rng = np.random.default_rng(SEED)

    log(f'partition {args.partition} | pooling {args.pooling} | variant {args.variant} '
        f'| content {args.content} | negatives {args.negatives} | closure {args.closure}')
    log('training the tNodeEmbed model on the real dataset')
    encoder, users, index, edge_sets, matrices, masks = real_representations('minilm', 'period',
                                                                            args.pooling)
    aligned = procrustes_chain(matrices, masks)
    model = train_model(aligned, prepare_pairs(edge_sets, users, index, matrices, masks, 'matched'),
                        index, matrices[0].shape[1])
    embeddings = node_embeddings(model, matrices)
    log(f'real users {len(users)} | edges last snapshot {len(edge_sets[-1])}')

    fitted = population_exponent(edge_sets, index, TRAIN_STEPS)
    exponent = args.exponent
    log(f'esponente della riscalatura di k: {exponent:.3f} (pendenza log-log stimata sul reale: '
        f'{fitted:.3f}). Il sintetico ha 10.000 utenti contro i 25.714 del reale e i suoi ultimi '
        f'passi ereditano la tabella dell\'ultima transizione di training: senza riscalatura il '
        f'grado medio uscirebbe 80 contro 32, con l\'esponente 1 esce nell\'ordine giusto.')
    tables, shares, sizes, unit = fit_counts(edge_sets, index, TRAIN_STEPS, exponent)
    log(f'unita\' di scala della tabella normalizzata: {unit:.1f}')
    for step in sorted(tables):
        table = tables[step] / tables[step].sum()
        expected = float(np.sum(np.arange(table.size) * table))
        log(f'  transizione {step}: eventi {sizes[step]} | k normalizzato atteso {expected:.1f} '
            f'| quota uscente {shares[step]:.3f}')
    cells = reciprocity_table(edge_sets, index, TRAIN_STEPS)
    for (direction, bucket), value in sorted(cells.items()):
        log(f'    bidirezionalita: uscente {int(direction)} | grado {bucket}: {value:.3f}')
    reverse_rate = float(np.mean([value for (direction, _), value in cells.items()
                                  if direction > 0.5]))
    log(f'  quota di coppie ricambiate usata in generazione: {reverse_rate:.3f}')

    log('fitted coefficients of the target-choice model (standardised features):')
    features, labels = build_choice_dataset(model, embeddings, users, index, edge_sets, masks, rng,
                                           with_content, TRAIN_STEPS, stratified, closure)
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(max_iter=2000).fit(scaler.transform(features), labels)
    for name, weight in zip(FEATURES, classifier.coef_[0]):
        log(f'  {name:24s} {weight:+.4f}')

    val_features, val_labels = build_choice_dataset(model, embeddings, users, index, edge_sets,
                                                    masks, rng, with_content, (VAL_STEP,), stratified,
                                                    closure)
    auc = roc_auc_score(val_labels,
                        classifier.predict_proba(scaler.transform(val_features))[:, 1])
    log(f'validation transition {VAL_STEP}: AUC {auc:.4f} (diagnostica del fit; la qualita\' della '
        f'rete si giudica sulle sue statistiche, non qui)')

    os.makedirs(RESULTS, exist_ok=True)
    with open(coefficients_path(args.variant), 'w', encoding='utf-8', newline='') as f:
        f.write('feature,coefficient\n')
        for name, weight in zip(FEATURES, classifier.coef_[0]):
            f.write(f'{name},{weight:.6f}\n')
        f.write(f'val_auc,{auc:.6f}\n')

    synthetic_users, _, synthetic_matrices, synthetic_masks, synthetic_present, _ = \
        synthetic_representations(encoder, os.path.join(BASE, args.partition), args.pooling)
    if args.content == 'permuted':
        order = rng.permutation(len(synthetic_users))
        synthetic_matrices = [matrix[order] for matrix in synthetic_matrices]
        log('  content permuted: same profiles, different owner, one permutation for all the steps')
    synthetic_embeddings = node_embeddings(model, synthetic_matrices)
    folder = os.path.join(BASE, f'{args.partition}_{args.variant}')
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, 'tnodeembed_embeddings.npz'),
                        users=np.array(synthetic_users),
                        **{f'step_{t + 1}': synthetic_embeddings[t].cpu().numpy()
                           for t in range(N_SNAPSHOTS)})
    log(f'generating on {args.partition} -> {os.path.basename(folder)} (arrival events)')
    events, edges, reversed_edges = generate(
        model, synthetic_embeddings, synthetic_users, synthetic_masks, synthetic_present, scaler,
        classifier, tables, shares, reverse_rate, exponent, unit, with_content, folder,
        args.partition, rng, closure)

    row = {'variant': args.variant, 'partition': args.partition, 'content': args.content,
           'operating_point': 'arrival events, sampled jointly', 'val_auc': round(float(auc), 4),
           'events': events, 'edges_per_event': round(edges / max(1, events), 3),
           'reverse_share': round(reversed_edges / max(1, edges), 4)}
    row.update(metrics_from_folder(folder))
    existing = []
    if os.path.exists(REPORT):
        with open(REPORT, encoding='utf-8', newline='') as f:
            existing = list(csv.DictReader(f))
    with open(REPORT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing + [row])
    log(f'  -> {row}')


if __name__ == '__main__':
    try:
        run()
    except Exception:                                            # noqa: BLE001
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise
