# CMDC.md — memoria del progetto per una nuova sessione

*Documento di handoff: spiega il progetto, i file, i processi e le trappole già incontrate a chi
riprende il lavoro in una nuova chat. Non sostituisce i documenti del progetto — li riassume e dice
dove guardare: **`FINAL.md`** (il piano e l'esito), **`docs/pipeline.md`** (il tecnico),
**`docs/relazione.md` + PDF** (il divulgativo), **`docs/nota_dottorando.md`** (la sintesi per il
dottorando), **`docs/project.md`** (il progetto in due minuti).*

---

## 1. Cos'è il progetto, e i vincoli

Progetto d'esame di Big Data. Si costruisce una **rete sociale sintetica temporale** che imiti un
dataset reale, usando **tNodeEmbed**, e la si confronta col reale.

Vincoli concordati con il dottorando (non negoziabili):

- si **addestra sul reale** e si fa **inferenza sul sintetico**; solo collegamenti sintetico↔sintetico,
  nessun aggancio al reale;
- **un passaggio = uno snapshot = un time step**: 9 nel reale, 9 nel sintetico (che va ripartito);
- l'**unica informazione comune** ai due dataset è il testo dei post (`content`, oltre a `id` e
  `account_id`): il reale ha campi del post, il sintetico attributi dell'utente. Gli attributi del
  sintetico servono **solo in valutazione**, mai come feature (sarebbe leakage);
- **struttura ≠ contenuto**: la struttura si usa (gradi, arrivo, attività), il reale non va mai
  modificato.

## 2. I dati

| | reale | sintetico |
|---|---|---|
| utenti | 25.714 (nodi) | 10.002 (10.000 nominali) |
| post | 173.475 cumulativi / 154.580 per periodo | 100.456 (100.474 righe, 18 scartate) |
| archi | 824.045 diretti, 9 snapshot 2022 → 2025_07 | **nessuno**, e **nessuna data** |
| split | 6 training / 1 validation (`2025_07_01_20_val`) / 2 test | il tempo lo imponiamo noi |

Cartella: `Esame Angelantonio Fedele Murolo/real_snapshots/` (3 file per snapshot:
`social_network.edg` a due colonne di **id utente**, `posts_current_snapshot.tsv` = solo il periodo,
`posts_incremental.tsv` = cumulativo) e `synthetic_snapshots/` (4 batch, ~9.700 utenti ciascuno con
sovrapposizione). I `profiles_*` e `content_attributes_*` usano le label del sintetico.

## 3. Il tempo: il criterio degli snapshot

`src/build_synthetic_snapshots.py`. Il sintetico ha solo un `id` denso (0…100.455): è l'unico ordine
di fabbrica. **Due cose vengono dal reale, misurate**:

- **curva di arrivo** (quota di utenti che entra a ogni passo): 41,8% · 14,2% · 11,0% · 7,2% · 7,8%
  · 5,1% · 1,0% · 7,7% · 4,2% — utenti ordinati per **primo post**;
- **forma dell'attività** (*quando* si scrive): i post di ogni utente sono distribuiti sui passi
  della sua vita campionando la distribuzione reale dei post per utente attivo per periodo
  (media 3,7 · mediana 2 · p90 9), con le raffiche **collocate a caso** (compaiono periodi pieni e
  silenziosi).

**Nostro e dichiarato**: ordine di arrivo (dagli `id`), ordine dentro il passo (per `id`), variante
di controllo "tutti presenti". **Non** copiamo la curva aggregata dei post per snapshot del reale
(gli snapshot hanno durata diversa — un anno, tre mesi, venti giorni, tre giorni — quindi quei
conteggi misurano "attività × lunghezza finestra").

Partizioni definitive: `generated_snapshots_v2` (tutti presenti) e
`generated_snapshots_growth_v2` (arrivo progressivo). Le `generated_snapshots*` senza `_v2`
(attività uniforme) sono il **controllo storico**.

## 4. La pipeline

1. **Contenuto** (`src/content_encoders.py`, `src/tnodeembed_real.py`) → encoder `all-MiniLM-L6-v2`,
   **ogni post codificato da solo** e vettori **mediati per utente** (`--pooling mean_of_posts`).
   Per-post batte la concatenazione (0,534 → 0,549) perché la concatenazione veniva tagliata al
   limite di token (per gli utenti più attivi sopravviveva il 28% del testo).
2. **Modello temporale** (`src/transfer_to_synthetic.py`, `src/tnodeembed_real.py`): tNodeEmbed
   **senza node2vec** — Procrustes a catena, LSTM, testa `[h_u, h_v, h_u⊙h_v]`, addestrata sulle
   transizioni di training.
3. **Generazione della rete**, due famiglie ufficiali:
   - **V8** (`src/generate_structural.py`): modello di formazione degli **archi** fittato sul reale
     (logistica su contenuto, grado relativo, "utente nuovo", arco inverso, vicini in comune,
     attività, recenza, passo), poi **campionamento con retroazione** (lo stato si ricalcola a ogni
     passo). Nessuna soglia, nessun conteggio imposto.
   - **V9** (`src/generate_arrival.py`): l'unità è l'**evento di arrivo**. Sul reale si stimano k
     (archi per evento), la quota uscente, la scelta dei bersagli (grado, novità, attività,
     contenuto, co-occorrenza) e la quota di coppie ricambiate; si genera arrivo per arrivo con k
     riscalato sulla popolazione (`--exponent 1`).
   Entrambi accettano `--content {real,permuted,off}` (**A** testo autentico, **B** testo permutato
   fra utenti, **C** senza testo) e scrivono `tnodeembed_embeddings.npz` nella cartella generata.
4. **Analisi** (su reale e su ogni rete): `analyze_structure.py`, `analyze_temporal.py`,
   `analyze_content.py`, `analyze_profiles.py`, `check_real_accretion.py`, `analyze_homophily.py`,
   `variant_summary.py`, `compare_folders.py`.

## 5. I risultati

| rete | archi | grado medio | grado max | reciprocità | modularità | transitivity | clustering | diametro |
|---|---|---|---|---|---|---|---|---|
| **reale** | 824.045 | **32,05** | **1.290** | **0,418** | **0,275** | **0,034** | **0,080** | **6** |
| **V9** | 280.298 | 28,30 | 324 | 0,304 | 0,106 | 0,0209 | 0,043 | 6 |
| V8 | 129.321 | 12,93 | 102 | 0,003 | 0,152 | 0,003 | 0,005 | 9 |

- **V9 è il risultato più avanzato** e supera V8 su tutte le proprietà relazionali (reciprocità,
  scala dei gradi, coda, triangoli, clustering, diametro). Gli scarti residui sono 0,25–0,9× e hanno
  **una causa unica**: la preferenza per i nodi più connessi resta sottostimata nel trasferimento da
  25.714 utenti reali a 10.000 sintetici (Rischio 1 del piano).
- **Il contenuto non decide chi si collega a chi**: quattro controlli A/B/C concordi (V8 e V9).
- **Il 2,13 di omofilia del reale è in gran parte struttura**: contro non-archi appaiati per grado
  scende a 1,51, a 1,29 col grado esatto ±10%; gli estremi degli archi hanno grado medio 403,6
  contro 61,7.
- **Il testo conosce l'utente, la struttura i collegamenti**: dal contenuto si predice lo
  schieramento politico (AUC 0,827), la professione (0,867), l'età (0,690); gli embedding tNodeEmbed
  molto peggio (0,61 / 0,55 / 0,57), perché sono addestrati sui legami.

## 6. I risultati negativi (dichiarati, non nascosti)

- il **contenuto** non sposta la rete (A/B/C indistinguibili); l'ablazione per AUC è satura;
- **fine-tuning** dell'encoder e **feature dei nodi addestrabili**: peggiorano (0,519 e 0,512 contro
  0,534);
- **la `Q_t` completa** del paper non entra nei 6 GB: non provata;
- **il ramo "i post come archi" è chiuso**: il testo di un singolo post non separa le coppie vere
  (AUC 0,515–0,517 contro 0,5166 della media), i post di una coppia reciproca non si somigliano più
  del caso, chi scrive di più non riceve più archi; la parte strutturale (co-interazione sullo stesso
  post) non è verificabile perché il dataset non registra **quale** post è agganciato a un arco;
- **la leva "più negativi nel fit"** non cambia la coesione e costa 5×.

## 7. Le direzioni aperte

1. **La preferenza per l'hub** (l'unica leva residua): come trasferire meglio i percentili di grado
   fra reti di dimensioni diverse — da qui passano modularità, coda dei gradi e quota new→new.
2. **La dipendenza fra i bersagli dello stesso evento** (chiuderebbe la parte triadica).
3. La **parte strutturale dei post-come-archi**, solo se un giorno arrivasse il file delle
   interazioni (post id + utente).

## 8. La mappa dei file

- **piano e esito**: `FINAL.md` (§0 in una pagina, §4 i comandi, §10 i risultati)
- **tecnico**: `docs/pipeline.md` (script, scelte, limiti) · **progetto in due minuti**: `docs/project.md`
- **divulgativo**: `docs/relazione.md` → `docs/relazione.pdf` (rigenerato con
  `pandoc + xelatex`, evitando `≈` e gli apici esponenti che il font non ha)
- **dottorando**: `docs/nota_dottorando.md` (e `docs/conversazione.md`, lo scambio originale)
- **script** in `src/`: `graph_io.py` (I/O condiviso), `content_encoders.py`, `tnodeembed_real.py`,
  `transfer_to_synthetic.py`, `build_synthetic_snapshots.py`, `measure_source_activity.py`,
  `generate_structural.py` (V8), `generate_arrival.py` (V9), `apply_edge_rules.py` (v1–v5),
  `apply_v6.py`, `generate_inference.py` (v7), `finetune_encoder.py`, `analyze_structure.py`,
  `analyze_temporal.py`, `analyze_content.py`, `analyze_profiles.py`, `check_real_accretion.py`,
  `analyze_homophily.py`, `variant_summary.py`, `compare_folders.py`;
  chiusi col ramo post-come-archi: `build_edge_posts.py`, `edge_content_test.py`
- **risultati** in `results/`: `v8_report.csv`, `v9_report.csv`, i `*_coefficients.csv`,
  `variant_summary.csv` (la tabella unica), `structure_*`, `structure_heavy_*`, `temporal_*`,
  `content_*`, `content_attributes_*`, `accretion_*`, `profiles_*`, `homophily_control.csv`,
  `compare_folders`, e le figure in `results/figures/`
- **reti generate** in `Esame Angelantonio Fedele Murolo/generated_snapshots*_v8*/` e `..._v9*/`
  (ignorate da git, rigenerabili)

## 9. Convenzioni e ambiente

- **seed fissi** ovunque; le reti sono **riproducibili**: i generatori iterano gli archi in ordine
  ordinato (l'ordine di iterazione dei set di stringhe non è fissato in Python e faceva derivare il
  volume di circa il 2% fra un run e l'altro), quindi ripetere i comandi dà gli stessi numeri;
- **lingue**: script, CSV e figure in **inglese**; i `.md` e la conversazione in **italiano**;
- **macchina**: una RTX 3060 da 6 GB (il fine-tuning di `bge-base` richiede `--batch 8`);
- **git**: ramo di lavoro `levers/posts-as-edges` (= `main`, allineati); `Esame .../`, `models/`,
  `results/_*.txt` sono in `.gitignore`; commit con trailer
  `Co-authored-by: CommandCodeBot <noreply@commandcode.ai>`;
- **metrica**: una rete generata **non si valuta con un AUC** (nel sintetico non ci sono archi veri)
  ma confrontando statistiche e distribuzioni; l'AUC serve solo sul reale (fit e diagnostica). Le
  distanze fra reti sono la prova, l'AUC saturo no.

## 10. Le trappole già incontrate (non ripeterle)

1. **Il bacino dei negativi deve coincidere con quello dei candidati della generazione.** In V9 i
   negativi erano campionati solo fra i presenti già esistenti: la variabile "la controparte è nuova"
   separava da sola positivi e negativi e la rete generata finiva per avere il 99,95% di archi fra
   co-arrivati (reale: 8,1%), con modularità 0,854 contro 0,275. Corretto il pool: new→new 17%,
   transitività 0,0203.
2. **Misurare la composizione della rete generata prima di toccare il modello**: un controllo da due
   minuti sull'accrescimento (`check_real_accretion.py`) ha risolto quello che sembrava un problema
   di forma del modello.
3. **Il riscalamento di k sulla popolazione va lineare** (`--exponent 1`): l'esponente 1,79 stimato
   log-log, applicato fuori dal range dei dati, fa collassare il volume a un terzo.
4. **Nei cicli Python non richiamare funzioni che scorrono tutti gli archi**: `population(...)` dentro
   il ciclo sugli eventi significava miliardi di operazioni e un run di fatto infinito.
5. **La ridirezione dell'output in `cmd` è bufferizzata**: il file di console resta vuoto fino alla
   fine. I log live sono `results/_v8_run.txt` e `results/_v9_run.txt` (scritti con `flush`).
6. **`cmd` non ha heredoc**: per i commit usare più `-m` (o `-F file`), mai `<<`.
7. **PDF**: `pandoc docs/relazione.md -o docs/relazione.pdf --pdf-engine=xelatex -V geometry:margin=2.5cm -V fontsize=10pt`
   — evitare `≈` e `⁻⁵` (usare `~`, `circa`, "6,4 su 100.000") perché nel font spariscono.

## 11. I comandi chiave

Tutti i comandi sono in **`FINAL.md` §4**. I più usati: partizione del sintetico
(`build_synthetic_snapshots.py --activity from-real`), generazione V9 e V8 (con `--content`), il
controllo A/B/C, `analyze_*` su reale e su ogni cartella, `variant_summary.py`. Da qui in poi, la
riga di comando non serve per capire il progetto: bastano `FINAL.md` §10 e `docs/relazione.md` §6.
