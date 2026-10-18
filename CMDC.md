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
sovrapposizione: **misurato**, ognuno copre tutto l'intervallo degli `id` — mediana intorno a
50.000 — quindi **non sono periodi**, e il quarto batch ha 2 righe: `results/native_batches.csv`).
I `profiles_*` e `content_attributes_*` usano le label del sintetico.

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
  scala dei gradi, coda, triangoli, clustering, diametro). Gli scarti residui **non sono una
  sottostima dell'hub**: la legge di scala del reale (nove snapshot, grado massimo contro utenti,
  esponente 1,445, R² 0,986) prevede ~339 a 10.000 utenti e V9 arriva a 328 (**97%**), e il rapporto
  169/437 è il rapporto fra le popolazioni (0,385). Restano la **forma** della coda (a pari dimensione
  −3,33 contro −1,69: hub relativamente meno estremi) e la **coesione**. Artefatti: `src/tail_scaling.py`,
  `results/tail_scaling_real.csv`, `results/tail_scaling_summary.csv`.
- **Il confronto onesto con tutte le famiglie** (`src/variant_scorecard.py` → `results/variant_scorecard.csv`): su **7 metriche confrontabili V9 è la più vicina al reale solo sul grado medio**. La densità migliore è di V8, transitivity e clustering di v6b, la reciprocità di v6a/v6b (0,420, **imposta**) e grado massimo e modularità di v2. V9 vince la **contemporaneità**, non la classifica metrica per metrica. La misura che mancava, la **pendenza di coda** (CCDF del grado uscente, ultimo decile): reale **−2,35** (esponente **3,35**), v2 −2,25, v6b −2,09, **V9 −3,33 (4,33)**, v8 −3,63 → la coda di V9 è **troppo leggera**, quella di v2/v6b appena troppo pesante. Il **volume** non è un difetto: archi = grado medio × utenti, e alla scala reale V9 sta a **0,88** (V8 0,40, v2/v6b 1,29–1,32).
- **Le due partizioni, provate entrambe su V9**: `_v2` ("tutti presenti", **ufficiale**) contro `_growth_v2` (ingresso progressivo, curva di arrivo del reale). Quasi equivalenti — distanza relativa su otto metriche **3,79 contro 4,39**, che diventa 3,26 contro 3,29 togliendo la quota di co-arrivati (12,4% contro 17,0%, reale 8,1%) — con la progressiva migliore sulla reciprocità (0,3224 contro 0,3035) e l'ufficiale su volume, grado medio, grado massimo e coesione. Artefatti: `generated_snapshots_growth_v2_v9/`, `results/structure_generated_snapshots_growth_v2_v9.csv`, `results/accretion_generated_snapshots_growth_v2_v9.csv`.
- **La ripetibilità ha un confine, dichiarato**: il **generatore** è deterministico (il rilancio del comando documentato dà 280.299 archi contro 280.298, con gli stessi 8.293 eventi di arrivo, la stessa quota ricambiate 0,4074 e la stessa reciprocità 0,3035), l'**addestramento** no: i pesi variano al terzo decimale fra processi. Le tabelle si citano quindi come **un'unica esecuzione**. Artefatti: `results/v9_check_coefficients.csv` contro `results/v9_coefficients.csv`.
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
- **i negativi "misti"** (30% coppie invertite, 30% difficili sul grado, 40% uniformi — l'unica proposta esterna con un effetto meccanico plausibile): provati in controllo e **peggiorano** la rete — transitivity 0,0115 contro 0,0209, clustering 0,0305 contro 0,043, co-arrivati 18,4% contro 12,4%, grado dell'estremo già presente 104 contro 169 (reale 437), reciprocità 0,300 contro 0,3035. I pesi cambiano molto (novità +0,61 contro +1,22, grado −0,21 contro +1,02) ma la **reciprocità non si muove**: in V9 la genera la tabella delle coppie ricambiate, non il classificatore. La configurazione documentata è il default (`--negatives uniform`, cartella di controllo `generated_snapshots_v2_v9_strat`, coefficienti in `results/v9_strat_coefficients.csv`).

## 7. Le direzioni aperte

1. **Non più il trasferimento dell'hub** (misurato: a pari dimensione il grado massimo di V9 è al 97%
   di quello che la legge di scala del reale prevede, e il rapporto fra gli hub è il rapporto fra le
   popolazioni), ma la **concentrazione**: la coda resta più ripida (−3,33 contro −1,69 a pari
   dimensione) e la coesione sotto (transitività 0,0209 contro 0,0337). È qui che punta il
   campionamento congiunto dentro l'evento.
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
  `analyze_homophily.py`, `variant_summary.py`, `variant_scorecard.py` (scorecard + pendenza di
  coda), `consolidate_native.py` (consegna nella forma dei batch nativi), `compare_folders.py`;
  chiusi col ramo post-come-archi: `build_edge_posts.py`, `edge_content_test.py`
- **risultati** in `results/`: `v8_report.csv`, `v9_report.csv`, i `*_coefficients.csv`,
  `variant_summary.csv` (la tabella unica), `structure_*`, `structure_heavy_*`, `temporal_*`,
  `content_*`, `content_attributes_*`, `accretion_*`, `profiles_*`, `homophily_control.csv`,
  `variant_scorecard.csv`/`.md` (confronto onesto + pendenza di coda), `native_batches.csv`,
  `v9_strat_coefficients.csv`, `v9_check_coefficients.csv`, `compare_folders`, e le figure in
  `results/figures/`
- **reti generate** in `Esame Angelantonio Fedele Murolo/generated_snapshots*_v8*/` e `..._v9*/`
  (ignorate da git, rigenerabili)

## 9. Convenzioni e ambiente

- **seed fissi** ovunque; le reti sono **riproducibili per il generatore**: i generatori iterano gli archi in ordine ordinato (l'ordine di iterazione dei set di stringhe non è fissato in Python e faceva derivare il volume di circa il 2% fra un run e l'altro) e nessun ramo di flag deve consumare estrazioni in più nel percorso di default. L'**addestramento** invece non è riproducibile al bit: i pesi variano al terzo decimale fra processi, quindi il rilancio del comando documentato dà **280.299** archi invece di 280.298 (stessi 8.293 eventi, stessa reciprocità 0,3035). Le tabelle si citano come **un'unica esecuzione**;
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
6. **`cmd` non ha heredoc**: per i commit usare più `-m` (o `-F file`), mai `<<`. Attenzione anche al
   `;`: in `cmd` **non** separa i comandi (si usa `&&`) e finisce fra gli argomenti del programma.
7. **PDF**: `pandoc docs/relazione.md -o docs/relazione.pdf --pdf-engine=xelatex -V geometry:margin=2.5cm -V fontsize=10pt`
   — evitare `≈` e `⁻⁵` (usare `~`, `circa`, "6,4 su 100.000") perché nel font spariscono. Le stesse
   opzioni valgono per `paper.md` → `paper.pdf` e `paper_en.md` → `paper_en.pdf`.
8. **Nessuna sottocartella dentro una cartella dataset**: `graph_io.snapshots_folder` legge **ogni**
   sottocartella come uno snapshot, quindi una cartella di appoggio creata lì dentro rompe tutte le
   analisi (successo con `native_batches/`, poi spostata nella radice come `native_4_*.edg`).
9. **`results/v9_coefficients.csv` (e `v8_*`) è unico e viene riscritto a ogni run**: per un
   esperimento salvare i coefficienti a parte (`v9_strat_coefficients.csv`, `v9_check_coefficients.csv`)
   e poi `git checkout -- results/v9_coefficients.csv`, altrimenti si perde il fit dell'esecuzione
   documentata.
10. **I 4 batch nativi del sintetico non sono periodi** (ognuno copre tutto l'intervallo degli `id`,
    ~9.500 utenti condivisi su ~9.700, il quarto ha 2 righe): non esiste un consolidamento "per
    periodi" dei 9 passi, solo uno per appartenenza degli utenti — e il numero di run non cambia il
    fatto.
11. **Confrontare metriche fra reti di dimensioni diverse richiede la scala.** Il grado massimo
    (324 contro 1.290) e il grado medio dell'estremo già presente (169 contro 437) sembravano
    sottostime di V9 e sono invece il rapporto fra le popolazioni (0,388 e 0,385 contro 0,389). Il
    metro giusto è la legge interna del reale (grado massimo contro utenti: esponente 1,445, R² 0,986,
    `src/tail_scaling.py`), e per la **forma** della coda il riferimento è il reale **al primo
    snapshot** (10.755 utenti, pendenza −1,69). Lo stesso vale per densità e volume: non si
    confrontano fra popolazioni diverse.

## 11. I comandi chiave

Tutti i comandi sono in **`FINAL.md` §4**. I più usati: partizione del sintetico
(`build_synthetic_snapshots.py --activity from-real`), generazione V9 e V8 (con `--content`), il
controllo A/B/C, `analyze_*` su reale e su ogni cartella, `variant_summary.py`. Due aggiunte:
`uv run python src/variant_scorecard.py` (scorecard di tutte le varianti con la pendenza di coda) e
`uv run python src/consolidate_native.py <cartella>` (consegna nella forma dei 4 batch nativi). Da
qui in poi, la riga di comando non serve per capire il progetto: bastano `FINAL.md` §10 e
`docs/relazione.md` §6.
