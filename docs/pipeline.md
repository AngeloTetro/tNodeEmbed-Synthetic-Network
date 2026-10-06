# pipeline.md — la pipeline definitiva

## 1. Impianto (confermato dal dottorando)

- Si **addestra sul dataset reale** e si fa **inferenza sul dataset sintetico**.
- Per ora **solo collegamenti sintetico↔sintetico**.
- L'**unica feature è il `content`** dei post: è l'unica informazione comune ai due dataset (il reale ha gli attributi del post, il sintetico ha gli attributi dell'utente; l'unica cosa equiparabile è il testo).
- Il sintetico **non ha l'accrescimento** del reale (quasi tutti gli utenti sono presenti dal primo batch), quindi va **partizionato in 9 snapshot** per ricreare un'evoluzione della rete analoga a quella reale.

## 2. I due dataset

| | reale | sintetico |
|---|---|---|
| utenti | 25.714 nodi | 10.000 |
| post | 173.475 cumulativi / 154.580 per periodo | 100.474 righe nei batch (100.456 con `id` valido) |
| archi | 824.045 diretti | nessuno |
| tempo | 9 snapshot (2022 → luglio 2025), **split dichiarato 6 training / 1 validation / 2 test** | nessuno: 4 batch con ordine casuale, `id` dei post in sequenza densa (0…100.455) |
| per utente | solo i post (`content`) | 12 colonne di profilo (11 attributi dell'utente + `format_style`, che descrive come è stato scritto il post) + post |

Nessun file originale viene modificato: le cartelle generate sono **nuove** dentro `Esame Angelantonio Fedele Murolo/`.

## 3. Il tempo è la struttura portante: tNodeEmbed

**Ogni snapshot è un time step** (9 nel reale, 9 nel sintetico partizionato). La pipeline temporale è quella del paper, **senza node2vec**, con il contenuto al posto degli embedding statici:

1. **rappresentazioni per time step** — per ogni snapshot `t` e ogni utente, un vettore che rappresenta il contenuto dei suoi post fino a `t` (encoder scelto al §4); in questa variante le rappresentazioni **sostituiscono** la matrice `Q_t` appresa del paper (non c'è una matrice di embedding addestrabile inizializzata dalle feature: le feature *sono* l'input del modello);
2. **allineamento** — Procrustes ortogonale a catena fra snapshot consecutivi (`R_t` stimata in forma chiusa sui nodi comuni e applicata a tutti, nuovi inclusi) come passo di pre-processing; **non** compare come termine esplicito nella loss (anche l'implementazione di riferimento allena con la sola cross-entropy binaria);
3. **LSTM** — la sequenza dei time step di ogni nodo viene compressa nell'embedding temporale;
4. **testa del task** — link prediction: ogni coppia `(u, v)` è rappresentata dai due embedding temporali **e dal loro prodotto elemento per elemento** (`[h_u, h_v, h_u ⊙ h_v]`), con ottimizzazione end-to-end della loss del task (eq. 1–4 del paper).

Il modello si allena sui **6 snapshot di training** del reale, si controlla su **1 di validation** e si misura su **2 di test**; poi si applica al sintetico per generare la rete, **snapshot per snapshot**.

## 4. La feature: il contenuto, e l'encoder

Il contenuto è l'unica feature, quindi l'encoder è la leva principale. Famiglie disponibili (dalla letteratura recente: [rassegna MTEB](https://arxiv.org/html/2406.01607v2), [classifica modelli](https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-march-2026/)): TF-IDF/LSA, embedding densi statici (word2vec/GloVe/FastText), contextual (BERT/SBERT), universal text embeddings (BGE, GTE, E5, mxbai, fino ai modelli LLM-based), sparse neurali (SPLADE++) e late interaction (ColBERT).

**Ambiente di calcolo**: la macchina ha una **NVIDIA GeForce RTX 3060 con 6 GB di VRAM**. Sono quindi praticabili i modelli della fascia piccola e media (MiniLM, BGE-small/base, GTE-small/base, mxbai) e il **fine-tuning di un encoder piccolo** sul segnale del task; i modelli da 7B (e5-mistral, Qwen3-Embedding, LLM2Vec) solo in quantizzazione 4-bit.

Candidati al confronto: `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `bge-base-en-v1.5`, `bge-large-en-v1.5`, `gte-small`, `gte-base`, `GIST-small-Embedding-v0`, `mxbai-embed-large-v1`, più `tf-idf+svd` come riferimento classico; opzionale un encoder **fine-tuned** con loss contrastiva sui collegamenti del reale.

**Come si sceglie**: si addestra la pipeline tNodeEmbed sul reale con ciascun encoder e si confrontano i risultati su validation e test **con i negativi appaiati** (la stessa modalità usata poi nel transfer); il criterio è la media dell'AUC sui tre valori (validation, test1, test2) e l'encoder che la ottiene più alta diventa quello della pipeline. La scelta è guidata dai numeri, non da preferenze. Nel confronto attuale il campo è stretto: `minilm` 0,534 davanti a `gte-base` 0,532 e `bge-base` 0,530. (Con i negativi casuali i valori sono ~0,99 su validation e 0,95–0,97 sui test: non è quella la colonna della scelta.)

## 5. Cosa gira, e cosa produce

| script | cosa fa | output |
|---|---|---|
| `src/content_encoders.py` | interfaccia unica per gli encoder del contenuto | — |
| `src/tnodeembed_real.py` | pipeline tNodeEmbed sul reale, un giro per encoder (training/validation/test) | `results/encoder_comparison.csv` |
| `src/build_synthetic_snapshots.py` | partizione del sintetico in 9 time step: variante "tutti presenti" e variante "arrivo progressivo" | `Esame Angelantonio Fedele Murolo/generated_snapshots/`, `generated_snapshots_growth/` |
| `src/transfer_to_synthetic.py` | inferenza del modello addestrato sul reale: rete sintetico↔sintetico per ogni snapshot | **dentro la cartella della partizione**: `snapshot_t/social_network.edg`, `tnodeembed_embeddings.npz`, più `results/transfer_report.csv` |
| `src/analyze_structure.py <cartella>` | struttura: nodi, archi, densità, gradi (distribuzione, k-core), reciprocità, transitivity e clustering, componenti deboli/forti e gigante, diametro e cammino medio, modularità e comunità | `results/structure_*.csv`, `structure_heavy_*.csv` + figure |
| `src/analyze_temporal.py <cartella>` | evoluzione: delta per snapshot, crescita, densificazione (E ~ N^α), arrivi e turnover, persistenza degli archi (Jaccard), stabilità delle comunità (NMI) | `results/temporal_*.csv` + figure |
| `src/analyze_content.py <cartella>` | contenuto: omofilia degli archi vs coppie casuali, similarità dentro/fuori le comunità, comunità vs attributi | `results/content_*.csv` + figure |
| `src/analyze_profiles.py <cartella>` | profili (solo sulle cartelle generate): classificazione di `political_leaning`, `profession`, `age_interval` da contenuto, da embedding tNodeEmbed e da contenuto+rete | `results/profiles_*.csv` + figure |
| `src/compare_folders.py` | mette a confronto reale e partizioni leggendo i CSV | `results/figures/compare_*.png` |
| `src/graph_io.py` | libreria condivisa di lettura (edge list, post, attributi) | — |

Struttura, evoluzione e contenuto girano **sia sul reale sia su ogni cartella generata**; `analyze_profiles.py` solo sulle cartelle generate (i profili non esistono nel reale) e `compare_folders.py` una volta sola. I risultati numerici stanno in **CSV/TSV** (`results/`), le **figure** le producono gli script di analisi stessi (`results/figures/`). Nessuna relazione: si scrive solo se e quando serve.

## 6. Note, vincoli e limiti noti

- **Semantica dell'arco**: nessun significato specifico da preservare — è **un'interazione fra due utenti**; si può leggere come un follow, ma non è necessariamente un follow.
- **Utenti reali**: **non esistono profili o attributi** oltre ai post; l'unica feature utilizzabile è il `content`. Il dataset reale **non va modificato** in alcun modo.
- **`exact_level_found`** (0–5) nel sintetico deriva dal progetto originale sulla **tossicità** di post e commenti: è irrilevante per questa pipeline e non viene usato.
- **Restrizione del paper**: applicata al reale produce **0 archi** in tutte le transizioni; per questo la valutazione usa l'intera transizione e due modalità di negativi (casuali e appaiati), entrambe riportate nel CSV.
- **Regola di generazione della rete**: a ogni time step ogni utente **che scrive in quel time step** crea `k ~ Poisson(1.2)` nuovi archi uscenti verso gli utenti **già presenti nella rete** — quelli entrati in quello snapshot o in uno precedente: la rete è cumulativa e nessuno esce mai — scegliendo quelli con il punteggio più alto del modello ed escludendo self-loop e archi già presenti; gli archi si accumulano. È una scelta dichiarata: il conteggio non viene dal modello, solo l'ordinamento.
- **Conseguenza nota**: con questa regola la **reciprocità della rete generata è ≈ 0** (le scelte dei due capi sono indipendenti), mentre nel reale è ≈ 0,42. Varianti di conteggio/soglia sono possibili e non sono state ancora confrontate.
- **Encoder**: il confronto copre la variante di contenuto **per periodo**; la variante cumulativa non è stata confrontata sistematicamente. Nessun encoder è stato ancora **fine-tuned** (opzione prevista).
- **Righe scartate**: nei batch sintetici 18 righe non hanno il numero di campi dell'intestazione e vengono ignorate in lettura; per questo i post usati sono 100.456 su 100.474 righe. (Un `id` non valido non fa scartare la riga: riceve un identificatore di ripiego.)
- **Tracciabilità**: il progetto **non è sotto git**, quindi "nessun file originale modificato" è verificato per ispezione del codice (le scritture avvengono solo in `results/` e nelle cartelle `generated_*`).

## 7. Convenzioni

- File `.md` in italiano; ogni altro file (script, output, CSV, figure) **in inglese**.
- `Esame Angelantonio Fedele Murolo/` e `tNodeEmbed/` sono **read-only** (le cartelle `generated_*` sono aggiunte nostre).
- Seed fissi ovunque; ambiente `uv` (torch CUDA per la RTX 3060).

## 8. Artefatti prodotti

| artefatto | contenuto |
|---|---|
| `results/encoder_comparison.csv` | confronto degli encoder sul reale (val/test, entrambe le modalità di negativi, e il numero di archi che soddisfano la restrizione del paper) |
| `Esame Angelantonio Fedele Murolo/generated_snapshots/`, `generated_snapshots_growth/` | le due partizioni del sintetico: `snapshot_t/posts_current_snapshot.tsv`, `snapshot_t/posts_incremental.tsv`, `snapshot_t/social_network.edg` (rete generata), `users_by_birth.csv`, `tnodeembed_embeddings.npz` |
| `results/transfer_report.csv` | per partizione: AUC del modello sul reale, nodi/archi/densità/reciprocità della rete generata, sovrapposizione di vocabolario reale↔sintetico |
| `results/structure_*.csv`, `structure_heavy_*.csv` | metriche strutturali per snapshot (e metriche pesanti sull'ultimo) per reale e partizioni |
| `results/temporal_*.csv` | delta per snapshot, persistenza degli archi (Jaccard), stabilità delle comunità (NMI) |
| `results/content_*.csv`, `content_attributes_*.csv` | omofilia del contenuto e, dove esistono i profili, omofilia per attributo |
| `results/accretion_*.csv` | per ogni transizione, gli archi nuovi classificati per estremi (nuovo/già presente) e il grado dell'estremo già presente (`src/check_real_accretion.py`) |
| `results/profiles_*.csv` | classificazione dei profili dagli embedding tNodeEmbed e dal contenuto |
| `results/figures/*.png` | figure prodotte dagli script di analisi e dal confronto (`compare_*.png`) |
