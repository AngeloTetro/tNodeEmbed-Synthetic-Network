# tnodeembed_riuso.md — cosa riusare del codice di tNodeEmbed

Analisi (sola lettura, nessuna modifica) della repository `tNodeEmbed/`: 15 file `.py` in `src/`, nessuno script di root, nessun notebook. Unico dato presente: `data/PPI/interactions_with_dates.csv`.

**Conclusione sintetica**: tutta la parte node2vec sono **poche righe** — il metodo `_initialize_embeddings` conta ~24 righe ma il codice node2vec vero e proprio sono **4 righe** (costruzione del `Node2Vec`, `fit`, estrazione degli embedding per nodo, `set_node_attributes`). Il resto è infrastruttura (tempo cumulativo, Procrustes, LSTM, pivot, negativi, dataloader, metriche) che ci serve e che riprenderemo **come contratto**, non come codice (è datato e non eseguibile oggi).

---

## 1. Mappa file per file

| File | Righe | Cosa fa | Giudizio |
|---|---:|---|---|
| `src/main.py` | 47 | Entry point: carica dataset, istanzia `tNodeEmbed`, crea train/test, fit/predict, confronta con node2vec statico | **DA PORTARE** (scheletro di pipeline; API Keras vecchie) |
| `src/config.py` | 20 | Dizionario `params` (dataset, task, test_size, train_skip, args node2vec/keras) | **RIUSABILE** (i parametri node2vec diventano inutili) |
| `src/metrics.py` | 52 | AUC, F1 micro/macro (sklearn) | **RIUSABILE COSÌ** |
| `src/models/tNodeEmbed.py` | 213 | **Cuore del metodo**: init (node2vec + Procrustes), dataset, LSTM, pivot time | **DA PORTARE** — è il file da modificare |
| `src/models/task_model.py` | 72 | Classe astratta `TaskModel`: `fit`, `predict`, generator | **DA PORTARE** (interfaccia; `fit_generator` deprecato) |
| `src/models/static_model.py` | 51 | Baseline statica node2vec (solo ultimo time step, senza LSTM) | **DA PORTARE** come baseline di confronto |
| `src/models/__init__.py` | 2 | Esporta `tNodeEmbed`, `StaticModel` | **RIUSABILE COSÌ** |
| `src/loader/dataset_loader.py` | 39 | `load_dataset('PPI')` → networkx con attributo `time`; `df2graph` generico | **DA PORTARE** (`df2graph` sì; manca il loader dei nostri dati) |
| `src/loader/task_loader.py` | 147 | Costruzione X,y per TLP (pivot + negative sampling) e NC | **DA PORTARE** (concettualmente riutilizzabile) |
| `src/loader/dataset_generator.py` | 67 | Generator thread-safe: batch di nodi → tensori di embedding per time step | **DA PORTARE** (adattare la shape delle feature) |
| `src/loader/__init__.py` | 3 | Riesporta i loader | **RIUSABILE COSÌ** |
| `src/utils/graph_utils.py` | 242 | `get_graph_T`, `get_graph_times`, `get_pivot_time`, `multigraph2graph`, `random_false_edges`, `nodes2embeddings` | **DA PORTARE** — è il valore riusabile più alto |
| `src/utils/general_utils.py` | 11 | `save_object` / `load_object` (pickle) | **RIUSABILE COSÌ** |
| `src/utils/consts.py` | 2 | Costanti `TLP`, `NC` | **RIUSABILE COSÌ** |
| `src/utils/__init__.py` | 2 | Riesporta le utility | **RIUSABILE COSÌ** |

---

## 2. I contratti da rispettare (il punto di aggancio delle nostre feature)

**Formato del grafo**: `networkx` non-multigrafo, attributo `time` su ogni arco.

**Time step**: `times = get_graph_times(graph)`; `pivot_time = get_pivot_time(test_size)` per TLP; `train_time_steps = times[times <= pivot_time]`; ogni grafo è **cumulativo** (`get_graph_T(graph, max_time=t)`).

**Punto da sostituire — inizializzazione** (`_initialize_embeddings`, righe 160-184):

```python
for time in tqdm(times, ...):
    cur_graph_nx = get_graph_T(graph_nx, max_time=time)   # subgraph cumulativo fino a t
    cur_graph_nx = multigraph2graph(cur_graph_nx)
    node2vec = Node2Vec(graph=cur_graph_nx, quiet=True, **n2vargs)
    ntv_model = node2vec.fit()
    ntv = {node: ntv_model[str(node)] for node in cur_graph_nx.nodes()}
    nx.set_node_attributes(graph_nx, ntv, time)           # <-- CONTRATTO: nome attributo = time step
    return graph_nx
```

→ Le nostre feature vanno iniettate esattamente lì: `nx.set_node_attributes(graph, {node: feature_vector}, time)`.

**Allineamento** (`_align_embeddings`, righe 186-213): `scipy.linalg.orthogonal_procrustes`, catena `R_t` verso `Q_{t-1}`; **richiede dimensione costante** in tutti i time step; applicato anche in test.

**Lettura degli embedding**: attributi di nodo via `nodes2embeddings` (`graph_utils.py:211-242`), con padding a zeri se manca il time step; `dimensions = max(shape)` (`dataset_generator.py:47-49`).

**Architettura LSTM** (`_get_model`, righe 79-120):
- TLP: input `(T, 2, D)` → due split (nodo u / nodo v) → `LSTM(128, relu)` **condivisa** → `Concatenate` → `Dense(128, relu)` → `Dense(1)` sigmoide; loss `binary_crossentropy`.
- NC: `LSTM(128)` → `Dense(num_classes)` softmax; loss `categorical_crossentropy`.

**Pivot e negativi**: `get_pivot_time` sceglie il time con rapporto test ≈ 0,2 (scarta `ratio < 0,1`); `random_false_edges` campiona non-archi con `nx.non_edges`, con fallback random.

---

## 3. Cosa è rotto / datato (e cosa facciamo)

| Problema | Effetto | Nostra decisione |
|---|---|---|
| Keras standalone + `fit_generator` / `predict_generator` | non gira su TF 2.6+ | reimplementiamo in PyTorch |
| `nx.read_gpickle` / `nx.write_gpickle` | rimossi in networkx ≥ 2.6 | non li usiamo |
| `nx.non_edges` | rimosso | campionamento negativi scritto da noi |
| `OneHotEncoder().toarray()` | API cambiata | irrilevante (non usiamo NC) |
| Path relativi (`"../data"`) | va lanciato da `src/` | non rilevante |
| Mancano le label per NC | solo TLP eseguibile | ci serve solo TLP |
| Il README è dichiarato "beta" dagli autori | — | non ci fidiamo dei numeri, solo della struttura |

---

## 4. Cosa guardare per primo

1. `src/models/tNodeEmbed.py` — `_initialize_embeddings` (l'unico da sostituire), `_align_embeddings`, `_get_model`.
2. `src/utils/graph_utils.py` — logica temporale, pivot, negativi, lettura embedding.
3. `src/loader/task_loader.py` — costruzione del task TLP.
4. `src/models/task_model.py` — interfaccia train/predict.
5. `src/main.py` + `src/loader/dataset_generator.py` — scheletro end-to-end e batching.
