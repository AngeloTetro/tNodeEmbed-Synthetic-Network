# FINAL.md — il piano definitivo del progetto

*Documento di riferimento: cosa facciamo, in che ordine, con quali comandi e perché. Da copiare nella root del progetto come `FINAL.md` appena si esce dalla modalità piano.*

---

## 0. In una pagina

1. Si addestra sul **dataset reale** e si fa **inferenza sul dataset sintetico**; l'unica informazione **comune** ai due è il **testo dei post**.
2. **Un passaggio = uno snapshot = un time step**: 9 nel reale (dati), 9 nel sintetico (ripartito da noi in due varianti).
3. Il **contenuto** viene rappresentato con un encoder (scelto coi numeri), **ogni post codificato da solo** e i vettori mediati per utente.
4. Il modello temporale è **tNodeEmbed senza node2vec**: le rappresentazioni del contenuto sono l'input della LSTM (allineamento di Procrustes a catena come pre-processing), testa di link prediction `[h_u, h_v, h_u⊙h_v]`.
5. Il suo **punteggio di coppia** diventa **una variabile** di un **modello di formazione degli archi fittato sul reale**, che include lo **stato della rete** (grado relativo, arrivo, reciprocità, vicini in comune, recenza).
6. La rete sintetica si genera **passo per passo con retroazione**: probabilità calcolate sullo stato corrente, archi **campionati**, stato aggiornato, si ripete. Volume, distribuzione dei gradi e hub **emergono** dal modello: nessuna statistica del reale viene imposta.
7. Si valuta il **grafo** (non solo l'AUC) contro il reale e contro le varianti storiche, e si **quantifica il contributo del contenuto** con un'ablazione (modello con e senza punteggio di contenuto).
8. Tutto è misurato, riproducibile con seed fissi, e documentato in `docs/pipeline.md` (tecnico) e `docs/relazione.md` + `docs/relazione.pdf` (divulgativo, in italiano).

---

## 1. Obiettivo, vincoli, linguaggi

**Obiettivo**: costruire la rete fra gli **utenti sintetici** con un modello addestrato sul **reale**, e studiarne struttura ed **evoluzione** nel tempo confrontandola con quella reale.

**Vincoli dichiarati (dottorando e nostro accordo)**

- Si addestra sul reale, si fa inferenza sul sintetico; solo collegamenti **sintetico ↔ sintetico** (nessun aggancio al reale; il "fallback" dei nodi isolati non serve).
- **Niente node2vec**: le feature iniziali dei nodi sono i **vettori del contenuto** (il paper lo permette, §3.1).
- **L'unica colonna comune** ai due dataset (oltre a `id` e `account_id`) è il `content` dei post: il reale ha solo campi **del post** (lingua, conteggi di reazioni/risposte, timestamp), il sintetico ha **attributi dell'utente** (genere, religione, orientamento politico, età, professione…) più il testo. Poiché il modello si addestra sul reale e si applica al sintetico, come feature può entrare solo ciò che esiste da entrambe le parti: il contenuto. Gli attributi del sintetico servono solo in **valutazione**. Il dataset reale **non viene mai modificato**.
- L'**arco** è un'interazione fra due utenti (leggibile come follow, senza semantica specifica da preservare).
- Il sintetico, che non ha tempo, viene **ripartito in 9 time step** in due varianti.

**Linguaggi**: ogni file creato/generato è **in inglese**; **`*.md` e la conversazione con me in italiano**.

**Macchina**: una sola GPU, **RTX 3060 con 6 GB** (i modelli e i lotti vanno scelti di conseguenza: `bge-base` in fine-tuning richiede `--batch 8`).

---

## 2. I due dataset

| | **reale** | **sintetico** |
|---|---|---|
| utenti | 25.714 | 10.000 |
| post | 173.475 cumulativi / 154.580 per periodo | 100.456 utilizzabili (18 righe scartate su 100.474) |
| archi | 824.045 diretti | nessuno |
| tempo | 9 snapshot (2022 → 2025_07), split **6/1/2** (`2025_07_01_20_val`, `2025_07_21_28_test1`, `2025_07_29_31_test2`) | 4 batch senza ordine: il tempo lo imponiamo noi (`id` denso 0…100.455) |
| per utente | solo il testo dei post | 11 attributi di profilo + post |

**Partizioni del sintetico** (generate, mai modificando gli input): `generated_snapshots/` (tutti presenti dal primo passo) e `generated_snapshots_growth/` (arrivo progressivo, curva del reale: 41,8% → 14,2% → 11% → … → 100%).

**Fatti del reale che guidano il modello** (misurati, `results/accretion_*.csv`, `structure_*.csv`, `content_*.csv`):
- il **92%** degli archi nuovi coinvolge un utente appena entrato; archi **già presente → già presente**: **0 su 707.604**;
- gli utenti nuovi si collegano a nodi con grado medio **437** contro una media di **64**;
- **reciprocità 0,418**; grado uscente medio **32,05**; grado massimo **1.290**;
- **omofilia del contenuto 2,13** contro coppie a caso — ma è una misura **in gran parte strutturale** (§10): contro non-archi con la stessa posizione (grado, volume di testo, arrivo) scende a 1,29–1,37, e il residuo lessicale è piccolo (AUC ≈ 0,58), su una rappresentazione che il generatore non usa.

### 2.1 Il tempo del sintetico: come è stato costruito, e cosa resta una nostra scelta

Il sintetico **non ha tempo**, quindi il tempo lo imponiamo noi. Le regole attuali (`src/build_synthetic_snapshots.py`) sono queste:

| scelta | da dove viene | è dei dati o nostra? |
|---|---|---|
| numero di passi (9) | il reale ha 9 snapshot | **dato** |
| curva di arrivo (quota di utenti che entrano a ogni passo: 0,418 → 0,142 → 0,110 → … → 0,042) | **misurata sul reale** | **dato** |
| ordine di arrivo (chi entra prima) | utenti ordinati per il loro **primo post** (`id`) | **nostra assunzione** (che gli `id` riflettano l'ordine di generazione) |
| post per passo di ogni utente | **distribuzione reale dei post per utente attivo** (raffiche collocate a caso nella vita dell'utente) | **dato** (resta nostro solo l'ordine interno, per `id`) |
| variante `generated_snapshots` | tutti gli utenti presenti dal passo 1 | **nostra** (è il controllo senza accrescimento) |
| variante `generated_snapshots_growth` | arrivi con la curva del reale | **dato + nostra assunzione** (vedi sopra) |

Questo è **l'elemento che ancora imponiamo con la nostra costruzione**, e non è un dettaglio: nel reale il **92%** degli archi nuovi coinvolge un utente appena entrato, quindi lo *scenario* (quando gli utenti esistono e quanto scrivono a ogni passo) determina in buona parte la crescita della rete generata.

**Decisione presa: l'attività del sintetico si deriva dal reale.** In concreto:

1. si misurano sul reale **la curva di attività per passo** (quanti post scrive, in proporzione, l'insieme degli utenti attivi a ogni snapshot) e **la distribuzione dei post per utente attivo per periodo** (media 3,7, mediana 2, p90 9, coda fino a **767**: attività "a raffica", con periodi interi di silenzio);
2. i post di ogni utente sintetico vengono distribuiti sui suoi passi di vita campionando la **distribuzione reale dei post per utente attivo**, con le raffiche collocate **a caso** dentro la vita dell'utente. I totali per passo **non** seguono la curva aggregata del reale: quella curva viene misurata ma **non usata**, perché gli snapshot reali hanno durata disomogenea e quei conteggi misurano "attività × lunghezza della finestra";
3. resta **nostro solo l'ordine interno a ciascun passo** (per `id`), ed è dichiarato.

**Conseguenza importante**: gli utenti alternano passi attivi e passi silenziosi (come nel reale), quindi la variabile *"la sorgente ha scritto in questo periodo"* **diventa informativa** — con l'attività uniforme era inutile, perché tutti erano sempre attivi, e a decidere *quando* nasceva un arco restavano solo gli arrivi. La cronologia della rete generata può così crescere "a ondate" invece che in modo lineare.

Le due partizioni restano — `generated_snapshots_v2` (tutti presenti) e `generated_snapshots_growth_v2` (arrivo progressivo) — ma **in entrambe l'attività viene dal reale**: restano come controllo di sensibilità sull'**arrivo**, che è l'altro elemento strutturale. (Le `generated_snapshots*` senza suffisso `_v2`, ad attività uniforme, sono il controllo storico.)

**Cautela dichiarata**: dal reale prendiamo una **forma**, non un calendario (il sintetico ha `id`, non date), e i totali di post per utente restano quelli del dataset sintetico.

---

## 3. L'architettura definitiva, strato per strato

### 3.1 Strato 1 — Rappresentazione del contenuto

- Encoder scelto **coi numeri** (protocollo tNodeEmbed completo, negativi appaiati, media val+test): `all-MiniLM-L6-v2` (0,534) davanti a `gte-base` (0,532) e `bge-base` (0,530).
- **Codifica per post**: ogni post viene codificato **da solo** e i vettori di un utente vengono **mediati** (`encode_grouped`, modalità `mean_of_posts`). Motivo misurato: la vecchia concatenazione veniva tagliata al limite di token dell'encoder (per gli utenti più attivi sopravviveva il **28%** del testo).
- Miglioramento misurato: MiniLM **0,534 → 0,549**, bge-base **0,530 → 0,540**, tf-idf **0,505 → 0,524**; nel transfer 0,541 → 0,552.
- L'encoder si fitta **solo sul contenuto di training** (nessuna perdita di informazione dal test).

### 3.2 Strato 2 — Il modello temporale (tNodeEmbed, senza node2vec)

1. **Rappresentazioni per passo**: una matrice utenti × dimensione per ogni snapshot (contenuto dal passo 1 a t per la variante cumulativa; il periodo corrente per quella "per periodo").
2. **Allineamento**: Procrustes ortogonale a catena fra snapshot consecutivi (pre-processing, non termine di loss).
3. **LSTM**: legge la sequenza dei 9 passi di ogni utente → embedding temporale.
4. **Testa**: punteggio per coppia da `[h_u, h_v, h_u ⊙ h_v]`, addestrata con cross-entropy binaria sulle transizioni di training del reale, early stopping sulla validation; **il test non si tocca** fino alla valutazione.
- Feature dei nodi **congelate** (la variante addestrabile è stata provata e **peggiora**: 0,512 contro 0,534).

### 3.3 Strato 3 — Il modello di formazione degli archi (V8, **il metodo del piano**; V9 in §10)

**Unità di decisione**: la coppia ordinata (sorgente, destinazione) fra utenti presenti al passo t, esclusi self-loop e coppie già collegate.

**Prima di scrivere il codice, una misura decide il disegno.** Sul reale si misura la quota di **archi nuovi la cui sorgente ha scritto nel periodo** (nuovo script `src/measure_source_activity.py`, pochi secondi). Se è ~100%, "la sorgente deve essere attiva" è un **fatto dei dati** e va messo nella definizione dei candidati; se è molto meno, l'attività resta una **variabile** che il modello pesa da sé. Non lo decidiamo a priori: lo decidono i dati.

**Variabili per coppia** (tutte calcolate sullo stato al passo t−1):

| variabile | perché |
|---|---|
| **punteggio di contenuto** (tNodeEmbed) | è il segnale del testo: sceglie *fra chi* |
| **grado relativo** del destinatario (percentile nella popolazione) | attaccamento preferenziale, in forma **trasferibile** fra popolazioni di dimensioni diverse |
| grado relativo della sorgente | attività di chi crea il collegamento |
| **destinazione/sorgente nuova** (nessun arco prima di t) | il 92% degli archi nuovi coinvolge un entrante |
| **arco inverso esistente** a t−1 | reciprocità (0,418 nel reale) |
| **vicini in comune** (log1p) | chiusura triadica ("amico dell'amico") |
| sorgete ha scritto nel periodo | nel reale chi crea l'arco è attivo |
| **recenza** (passi dall'ultimo arco ricevuto) | il tempo come stato, non solo come indice |
| passo temporale (normalizzato) | tendenza temporale residua |

**Modello**: **regressione logistica** su variabili standardizzate, fittata sulle **transizioni di training del reale** (positivi = archi nuovi; negativi = coppie campionate dallo stesso universo di candidati, con negativi appaiati). I **coefficienti sono il risultato scientifico**: dicono quanto pesa la struttura e quanto il contenuto. Controllo di robustezza: un modello ad alberi (gradient boosting) sulle stesse variabili, per la parte non lineare.

**Correzione della prevalenza**: le probabilità vengono riportate alla scala reale (le coppie collegate sono ~6,4 su 100.000) con uno spostamento dell'intercetta, come in V7.

**Decisione e generazione**:
- **campionamento**: ogni coppia candidata diventa un arco con la *sua* probabilità corretta → **il volume emerge**, non viene scelto da noi;
- **nessuna soglia**, quindi **nessun artefatto** di calcolo;
- **iterazione con retroazione**: a ogni passo le variabili (gradi, novità, vicini in comune, recenza) sono ricalcolate sullo **stato della rete generata**, poi si campiona e si scrive `snapshot_t`.
- La decisione è **solo il campionamento** (scelta confermata); la soglia al tasso reale resta documentata solo in v7, come confronto storico.

**Il contenuto entra due volte, e in modo controllato**: (a) nelle rappresentazioni che addestrano tNodeEmbed, (b) come variabile nel modello di formazione degli archi. La sua importanza si **misura** (vedi §6), non si postula.

### 3.4 Strato 4 — Le varianti di confronto (tenute, non cancellate)

| variante | cos'è | perché resta |
|---|---|---|
| v1–v5 | regole a conteggio deciso a mano (`k ~ Poisson(1,2)`, quota reale, accrescimento, reciprocità aggiunta) | documentano il percorso e mostrano cosa succede senza modello |
| v6a/v6b | soglia calibrata sul reale (grado finale / crescita per snapshot), reciprocità forzata a 0,42 | "cosa serve per **somigliare** al reale": impone le statistiche |
| v7a–v7f | modello di formazione degli archi con soglia (blocchi/globale, encoder base/fine-tuned) | "cosa **predice** il modello" in forma minima: è il predecessore di V8 |
| **v8** | modello di formazione degli archi con campionamento e retroazione | **il metodo del piano: la rete ufficiale** |
| **v9** | evento di arrivo: quanti archi porta ogni utente nuovo, dove vanno, quanto è ricambiato | **il risultato più avanzato**: supera V8 su reciprocità (0,303 contro 0,003), scala dei gradi (28,2 contro 12,9) e coda (324 contro 102), con triangoli e clustering nell'ordine giusto (§10) |

La rete **ufficiale del piano** è V8; **V9** (§10) è la variante più avanzata e la supera su tutte le proprietà relazionali. v6 e v7 restano nella tabella di confronto — servono a mostrare la differenza fra **imparare** il meccanismo (v8, v9), **imporre** le statistiche del reale (v6) o applicare un modello di solo contenuto con una soglia (v7).

### 3.5 Strato 5 — Analisi e valutazione

- **Struttura**: nodi, archi, densità, gradi (distribuzione e CCDF), reciprocità, transitivity, clustering, componenti, k-core, modularità e comunità.
- **Evoluzione**: archi nuovi/rimossi per passo, densificazione, arrivi e turnover, persistenza (Jaccard), stabilità delle comunità (NMI).
- **Contenuto**: omofilia del contenuto (ratio collegati/casuali), similarità dentro/fuori le comunità, comunità vs attributi.
- **Accrescimento**: quota "già presente → già presente", grado dell'estremo già presente.
- **Tabella unica** (`results/variant_summary.csv`) con reale + tutte le varianti.
- **Ablazione del contenuto** (§6).

### 3.6 Cosa non imponiamo più, e cosa resta una scelta di costruzione

**Sul meccanismo non imponiamo più nulla.** Il numero di archi, la preferenza per i nodi connessi, la reciprocità, la chiusura triadica e gli effetti temporali sono **fittati sul reale** e poi **emergono** dalla generazione passo per passo: nessun valore viene scelto da noi per far assomigliare la rete al reale. È la differenza con v1–v5 (dove il conteggio era nostro) e con v6 (dove le statistiche del reale venivano imposte).

**Restano, per necessità, delle scelte di costruzione** — non numeri nascosti, ma scelte **dichiarate**:

| scelta | cosa determina |
|---|---|
| **lo scenario temporale del sintetico** (§2.1) | quando ogni utente esiste (curva di arrivo **dal reale**, ordine per `id`) e quanto scrive a ogni passo (forma di attività **dal reale**); resta nostro solo l'ordine interno per `id` e la variante "tutti presenti" |
| l'universo delle coppie candidate | quali coppie *possono* diventare archi a ogni passo |
| l'elenco delle variabili | quali meccanismi il modello *può* imparare (i pesi sono fittati; un meccanismo che non è nell'elenco, il modello non lo può usare) |
| la forma funzionale | regressione logistica con prevalenza corretta (+ controllo non lineare ad alberi) |
| la regola di campionamento | ogni coppia con la sua probabilità, indipendentemente: implica un certo profilo di "quanti archi per utente" |
| encoder e rappresentazione | scelti con i numeri fra le opzioni disponibili |

Il confine è questo: **sul meccanismo non imponiamo nulla; sullo scenario e sulle definizioni scegliamo, e lo dichiariamo.** Dove una statistica può venire dal reale viene dal reale (la curva di arrivo); dove non può — il sintetico non ha tempo — resta un'assunzione esplicita.

---

## 4. Il workflow completo, in ordine

```bash
# 0) PREPARAZIONE
#    partizione definitiva del sintetico: curve di arrivo E forma di attività dal reale
uv run python src/build_synthetic_snapshots.py --activity from-real

# 0b) MISURA CHE DECIDE IL DISEGNO: quota di archi nuovi la cui sorgente ha scritto nel periodo
uv run python src/measure_source_activity.py     # nuovo, pochi secondi -> results/source_activity.csv

# 1) ENCODER: confronto a parità di protocollo (decisione con i numeri)
uv run python src/tnodeembed_real.py --encoders tfidf,minilm,bge-small,gte-small,bge-base,gte-base \
       --content period --negatives random,matched
uv run python src/tnodeembed_real.py --encoders minilm,bge-base,tfidf --content period \
       --negatives matched --pooling mean_of_posts            # codifica per post (la migliore)

# 2) MODELLO TEMPORALE sul reale (tNodeEmbed, senza node2vec) → results/encoder_comparison.csv

# 3) GENERAZIONE DEFINITIVA (V8): modello di formazione degli archi + campionamento + retroazione
uv run python src/generate_structural.py --partition generated_snapshots        --pooling mean_of_posts
uv run python src/generate_structural.py --partition generated_snapshots_growth --pooling mean_of_posts
uv run python src/generate_structural.py --partition <partizione VECCHIA>      --pooling mean_of_posts
#    la terza è il CONTROLLO: isola quanto pesa il cambio di scenario (attività uniforme -> attività dal reale)

# 4) CONTROLLO DEL CONTENUTO: tre modi a parità di modello e di scenario
#    A: testo autentico | B: testo permutato fra utenti | C: senza testo
uv run python src/generate_structural.py --partition generated_snapshots_v2 --pooling mean_of_posts --content real     --variant v8_real
uv run python src/generate_structural.py --partition generated_snapshots_v2 --pooling mean_of_posts --content permuted --variant v8_permuted
uv run python src/generate_structural.py --partition generated_snapshots_v2 --pooling mean_of_posts --content off      --variant v8_off

# 4b) SCOMPOSIZIONE DELL'OMOFILIA DEL REALE: quanta è struttura (grado, volume, arrivo), quanta è scelta
uv run python src/analyze_homophily.py real_snapshots      # -> results/homophily_control.csv

# 5) VARIANTI DI CONFRONTO (già disponibili, si rigenerano solo se serve)
uv run python src/transfer_to_synthetic.py generated_snapshots_growth minilm matched mean_of_posts
uv run python src/apply_edge_rules.py                          # v1-v5
uv run python src/apply_v6.py                                  # v6a, v6b
uv run python src/generate_inference.py --encoder minilm --threshold global --variant v7c --pooling mean_of_posts

# 5b) V9, IL MECCANISMO DI ARRIVO (la variante più avanzata)
uv run python src/generate_arrival.py --partition generated_snapshots_v2 --content real     --variant v9_real
uv run python src/generate_arrival.py --partition generated_snapshots_v2 --content permuted --variant v9_permuted
uv run python src/generate_arrival.py --partition generated_snapshots_v2 --content off      --variant v9_off
#    entrambi i generatori (V8 e V9) scrivono anche tnodeembed_embeddings.npz nella cartella
#    generata, così le analisi dei profili hanno le righe con gli embedding tNodeEmbed

# 6) ANALISI SU OGNI RETE (reale compreso): struttura, evoluzione, contenuto, accrescimento, profili
uv run python src/analyze_structure.py <cartella>
uv run python src/analyze_temporal.py  <cartella>
uv run python src/analyze_content.py   <cartella>
uv run python src/check_real_accretion.py <cartella>
uv run python src/analyze_profiles.py <cartella>       # solo sulle cartelle generate (le label sono lì)

# 7) TABELLE E FIGURE
uv run python src/variant_summary.py
uv run python src/compare_folders.py

# 8) DOCUMENTI
#    docs/pipeline.md (tecnico) + docs/relazione.md -> docs/relazione.pdf (divulgativo)
```

**Note operative**: seed fissi ovunque; log di ogni script in `results/_*.txt` (la redirezione della shell su questa macchina non cattura l'output); la cartella dei dataset è sotto OneDrive → tenere i file "sempre disponibili" per evitare letture vuote; niente `models/` né log di lavoro nel versionamento (`.gitignore`). La partizione definitiva richiede **anche il dataset reale** (per misurare la forma di attività), e dopo averla creata **tutte** le reti usate nel confronto finale vanno rigenerate su di essa (§5).

---

## 5. Risultati già acquisiti (gli ancoraggi)

| misura | valore | fonte |
|---|---|---|
| encoder migliore, negativi appaiati | MiniLM 0,534 → **0,549** con per-post | `encoder_comparison.csv` |
| **profili dell'utente dal testo** (label solo sintetiche, split 80/20) | schieramento **0,827**, professione **0,867**, età **0,690** di AUC — e **aggiungere la rete non cambia nulla** | `profiles_*.csv` |
| **evento di arrivo nel reale** | da **35 a 111** archi per evento secondo la transizione (mediana 6–32), massimo **1.202**, **51–63%** uscenti, ~40% coppie ricambiate | `arrival_events.csv` |
| negativi **casuali** (non è la colonna di scelta) | ~0,99 val / 0,96–0,97 test | `encoder_comparison.csv` |
| fine-tuning dell'encoder | **peggiora** i test (0,519 vs 0,534) | `finetune_report.csv` |
| feature dei nodi addestrabili | **peggiora** (0,512 vs 0,534) | `encoder_comparison.csv` |
| coefficienti del modello (v7) | contenuto **+0,20**; grado +1,6; utente nuovo **+2,28** | `v7_coefficients*.csv` |
| v6 (calibrata) | reciprocità **0,420** esatta; grado 6,4–9,4 (il 32 è irraggiungibile col vincolo di accrescimento) | `v6_report.csv` |
| v7 (inferenza con soglia) | grado medio 3,6–5,8; reciprocità 0,00–0,72 | `v7_report.csv` |
| omofilia del contenuto | reale **2,13** — ma **1,29–1,37** contro controlli appaiati per grado/volume/arrivo: in gran parte struttura (§10); tutte le reti generate **≈ 1,00** | `content_*.csv`, `homophily_control.csv` |

**Nota sulle partizioni**: le reti già generate (riferimento, v1–v7) usano la **vecchia partizione** (attività uniforme, §2.1). Con la partizione definitiva vanno **rigenerate anche le varianti di confronto** usate nella tabella (transfer, v6, v7), così il confronto con V8 è omogeneo: stesso scenario, stesso numero di passi, stessi utenti presenti a ogni passo. In più si genera **V8 anche sulla vecchia partizione**: è il **controllo** che isola quanto pesa il cambio di scenario (attività uniforme → attività dal reale).

**Il risultato da spiegare** (e la ragione di V8): le reti generate finora hanno omofilia del contenuto ≈ 1,00, perché il modello di solo contenuto non vede il meccanismo che nel reale genera gli archi (arrivo e grado). V8 nasce per **imparare quel meccanismo** invece di imporlo.

---

## 6. Il controllo che quantifica il contenuto

**Quale metrica, per quale domanda.** Nella generazione **non esiste un insieme di archi veri** nel sintetico, quindi un AUC sugli archi non ha significato lì: la rete generata si valuta su **distribuzioni e statistiche di rete** per snapshot (gradi, reciprocità, transitività, modularità, omofilia, crescita) confrontate col reale. L'AUC resta la metrica giusta solo dove le etichette esistono, cioè **sul reale**, e per due domande specifiche: (a) il modello ordina meglio un arco nuovo di un non-arco appaiato? (b) una rappresentazione del contenuto separa un arco da un non-arco con la stessa posizione strutturale? Limite da dichiarare: l'AUC di ablation del modello è **saturo** (0,9818 → 0,9816), quindi è uno strumento debole proprio per misurare il contributo del testo — per quello servono i confronti fra reti.

Quattro misure, tutte strumentate:

1. **Sul reale (ranking)**: AUC/AP con negativi appaiati del modello **con** e **senza** il punteggio di contenuto — saturo: 0,9818 → 0,9816.
2. **Sui coefficienti**: il peso del contenuto nel modello fittato, con intervallo fra addestramenti (v7: +0,18 ÷ +0,26; V8: +0,14).
3. **Sulla rete generata, col controllo a testo permutato** (`--content real|permuted|off`): gli stessi profili riassegnati ad altri utenti (stesse statistiche, stessa supervisione, un'unica permutazione per tutti i passi) → se `real` e `permuted` danno la stessa rete, il testo non porta il legame. È il confronto che *misura*; l'AUC satura no.
4. **Omofilia del contenuto con controlli strutturali** (`src/analyze_homophily.py`): la misura 2,13 contro coppie a caso, ripetuta contro non-archi appaiati per grado, volume di testo e passo di arrivo → dice quanta parte dell'omofilia è scelta di somiglianza e quanta è posizione nella rete.

**Esito misurato**: con testo autentico, testo permutato e senza testo le reti sono **indistinguibili** nelle proprietà relazionali, e il 2,13 del reale è **in gran parte struttura** (§10).

---

## 7. Rischi, limiti dichiarati, criteri di accettazione

**Rischio 1 — trasferimento fuori distribuzione.** Il reale ha 25.714 utenti e il sintetico 10.000, con curve di arrivo diverse: un coefficiente stimato sui gradi reali applicato ai gradi sintetici è un'estrapolazione. **Mitigazione**: variabili **relative** (percentili nella popolazione corrente) invece di valori assoluti, e riporto del **volume implicito** del modello (se differisce da quello reale, è un risultato sulla diversa struttura di arrivi, non un parametro da aggiustare).

**Rischio 2 — il contenuto può risultare decorativo.** Se la struttura spiega quasi tutto, il progetto rischia di diventare "un modello strutturale con un termine di contenuto". **Risposta**: la divisione del lavoro è strutturale e va dichiarata — la struttura dà il *meccanismo* (quanti archi, con quale dinamica), il contenuto sceglie *quali coppie* fra quelle ammissibili, ed è l'unica informazione che distingue fra loro i 10.000 utenti sintetici. Il contributo si **misura** (§6).

**Rischio 3 — valutare un modello generativo non si fa con un AUC.** Nel sintetico non ci sono archi veri: la generazione si giudica sulle **distribuzioni e sulle statistiche di rete** per snapshot (gradi, reciprocità, transitività, modularità, omofilia, crescita) contro il reale, e qualche divergenza è fisiologica. L'AUC serve solo dove le etichette esistono, cioè sul reale, e non è un proxy della qualità della rete generata (§6).

**Rischio 4 — lo scenario temporale del sintetico resta in parte una nostra costruzione (§2.1).** Dal reale prendiamo la **curva di arrivo** e, dopo la decisione presa, anche la **forma di attività** (curva per passo + raffiche per utente); restano assunzioni nostre **l'ordine di arrivo** (dedotto dagli `id`) e la variante "tutti presenti". Poiché nel reale il 92% degli archi nuovi coinvolge un entrante, lo scenario resta l'elemento che pesa di più sulla rete generata. **Mitigazione**: dichiararlo come *scenario* e non come dato; le **due varianti** di arrivo come controllo di sensibilità; e il fatto che l'attività non sia più uniforme rende finalmente informativa la variabile "la sorgente ha scritto", che con l'attività costante era inutile.

**Criterio di accettazione (cosa significa "fatto")**:
- la rete V8 esiste, è scritta snapshot per snapshot e **riproducibile** con i comandi di §4;
- le statistiche strutturali (volume, gradi, hub, accrescimento) **emergono** dal modello e si avvicinano al reale **molto più** di v7, senza imposizioni;
- il contributo del contenuto è **quantificato** con l'ablazione;
- la tabella unica e la documentazione sono aggiornate, PDF compreso;
- se qualcosa non riesce, il documento lo dice con i numeri (risultato negativo ma misurato).

---

## 8. Perché questo è il disegno definitivo (e cosa resta fuori)

**Perché**: è l'unico disegno coerente con le prove raccolte in tutto il progetto — la struttura domina la formazione degli archi; il modello di solo contenuto non riproduce la rete (e per avvicinarla abbiamo dovuto *copiare* le statistiche del reale in v6); tutte le altre leve provate (encoder più grandi, fine-tuning, feature addestrabili, codifica per post) **non** cambiano la rete, e sono documentate come tali. V8 è anche ciò che chiedevi: un modello **fittato** sul reale, non una regola scritta a mano.

**Cosa resta fuori, e perché**:
- **node2vec** (escluso per progetto: non sa rappresentare utenti mai visti);
- **la `Q_t` completa del paper** (una riga per utente e snapshot, ~70M parametri): non entra nei 6 GB; l'evidenza dice comunque che la capacità non è il collo di bottiglia;
- **il fine-tuning dell'encoder** (provato, peggiora i test);
- **l'aggancio sintetico ↔ reale** (escluso dal dottorando; non serve);
- **gli attributi del sintetico come feature** (non esistono nel reale: restano solo per le analisi di valutazione, `analyze_profiles.py`);
- **ripetere il confronto fra tutti gli encoder in versione per-post** (fatto su tre: la graduatoria non cambia).

Queste strade sono **documentate e non vengono riaperte** in questo progetto: non c'è una sezione di "estensioni possibili". Se V8 non bastasse, se ne parla dopo, con i numeri di V8 in mano.

---

## 9. Cosa produciamo alla fine

| artefatto | dove |
|---|---|
| reti generate (V8, due partizioni; più le varianti) | `Esame Angelantonio Fedele Murolo/generated_snapshots*_v8/` |
| **rete V9** (il meccanismo di arrivo, tre modalità di contenuto) | `Esame Angelantonio Fedele Murolo/generated_snapshots_v2_v9_*/` |
| **consegna doppia**: i 4 file `.edg` nella forma dei batch nativi (**restrizione per utenti**, non per periodi: i batch non sono slice temporali, §2) | `<cartella V9>/native_4_batch_<0-3>.edg`, `<cartella V9>/native_4.csv`, `results/native_batches.csv` |
| **V8 di controllo sulla vecchia partizione** (misura l'effetto del cambio di scenario) | `<partizione storica>_v8/` |
| modelli, report, coefficienti | `models/`, `results/v8_report.csv`, `results/v8_coefficients.csv`, `results/v9_report.csv`, `results/v9_coefficients.csv` |
| **embedding tNodeEmbed del sintetico** (per le analisi dei profili) | `<cartella generata>/tnodeembed_embeddings.npz`, scritto da **entrambi** i generatori |
| tabelle e figure di confronto | `results/variant_summary.csv`, `results/figures/` |
| analisi complete su reale e su ogni rete | `results/structure_*.csv`, `temporal_*.csv`, `content_*.csv`, `content_attributes_*.csv`, `accretion_*.csv`, `profiles_*.csv` |
| documentazione tecnica | `docs/pipeline.md`, `docs/project.md`, `FINAL.md` |
| relazione divulgativa per me | `docs/relazione.md` + `docs/relazione.pdf` |
| **nota tecnica breve per il dottorando** | `docs/nota_dottorando.md` (2–3 pagine, in italiano: cosa è stato fatto, i numeri, i risultati negativi dichiarati) |
| storia del lavoro | commit su `levers/encoder` (main = fotografia congelata) |

---

## 10. Esito (a fine lavoro)

| rete | archi | densità | grado medio | grado max | reciprocità | modularità | transitivity | omofilia |
|---|---|---|---|---|---|---|---|---|
| **reale** | 824.045 | **0,00125** | **32,05** | **1.290** | **0,418** | 0,275 | 0,034 | **2,13** |
| **V8, tutti presenti** | 129.321 | **0,00129** | 12,9 | 102 | 0,003 | 0,152 | 0,003 | 1,00 |
| **V8, arrivo progressivo** | 159.804 | 0,00160 | 16,0 | 125 | 0,004 | 0,140 | 0,004 | 1,00 |
| V8 controllo (vecchio scenario) | 168.015 | 0,00168 | 16,8 | 128 | 0,004 | 0,131 | 0,003 | 1,00 |
| V8 **ablazione senza contenuto** (`growth_v2`) | 153.100 | 0,00153 | 15,3 | 126 | 0,004 | 0,142 | 0,004 | 1,00 |

**Controllo del contenuto A/B/C** — stesso modello fittato sul reale, stesso scenario, partizione `generated_snapshots_v2`, unica differenza il canale testuale sul lato sintetico:

| modalità | archi | densità | reciprocità | grado max | modularità | transitività | omofilia |
|---|---|---|---|---|---|---|---|
| A — testo autentico | 128.831 | 0,001288 | 0,0031 | 107 | 0,154 | 0,00269 | 1,0009 |
| B — testo permutato fra utenti | 123.294 | 0,001233 | 0,0030 | 96 | 0,159 | 0,00253 | 0,9998 |
| C — senza testo | 124.560 | 0,001246 | 0,0029 | 106 | 0,156 | 0,00262 | 1,0004 |

**V9 — il meccanismo di arrivo** (`src/generate_arrival.py`): l'unità non è la coppia ma l'**evento di arrivo**, cioè un utente che entra nella rete in quella transizione. Misurato sul reale (`results/arrival_events.csv`): ogni evento porta in media da **35 a 111 archi** secondo la transizione (mediana 6–32), con massimo **1.202** — lo stesso ordine del grado massimo reale, quindi l'hub nasce da un *singolo* arrivo e non da molte decisioni indipendenti; il **51–63%** degli archi dell'evento è uscente e il **40%** circa delle coppie è ricambiato dentro lo stesso evento. V9 stima queste tre cose sul reale e genera arrivo per arrivo: k archi per evento, bersagli scelti con un modello (grado log +1,02 e percentile +0,91, "la controparte è nuova" +1,21, co-occorrenza +0,19, contenuto +0,05), coppie ricambiate pescate fra gli uscenti già scelti, e k riscalato linearmente sulla popolazione (`--exponent`, default 1: senza riscalatura il grado medio uscirebbe 80 contro 32).

| rete | archi | densità | grado medio | grado max | reciprocità | modularità | transitività | clustering | omofilia |
|---|---|---|---|---|---|---|---|---|---|
| **reale** | 824.045 | 0,001246 | **32,05** | **1.290** | **0,418** | **0,275** | **0,034** | **0,080** | **2,13** |
| **V9, testo autentico** | 280.298 | 0,002858 | **28,30** | **324** | **0,304** | 0,106 | **0,0209** | **0,043** | 0,999 |
| V9, testo permutato | 274.448 | 0,002803 | 27,73 | 344 | 0,302 | 0,106 | 0,0207 | 0,042 | 1,002 |
| V9, senza testo | 280.292 | 0,002861 | 28,32 | 326 | 0,304 | 0,105 | 0,0211 | 0,044 | 0,999 |
| V8 (coppie indipendenti) | 129.321 | 0,001293 | 12,93 | 102 | 0,003 | 0,152 | 0,003 | 0,005 | 1,0003 |

- **La diagnostica, e il bug che ha trovato**: la quota di archi fra utenti arrivati **nello stesso passo** era **99,95%** in V9 contro l'**8,1%** del reale (`check_real_accretion.py`, sul reale e sulla rete generata). La causa era nel fit: i negativi della scelta venivano campionati **solo fra i presenti già esistenti**, mai fra i co-arrivati, quindi la variabile "la controparte è nuova" separava positivi e negativi *da sola* (coefficiente gonfiato a +1,38) e in generazione la scelta andava quasi tutta sui co-arrivati. Correzione: negativi dallo **stesso insieme dei candidati della generazione** (presenti più nuovi del passo), con il percentile ricalcolato su quell'insieme.
- **Cosa V9 risolve**: la **reciprocità** (0,304 contro 0,003 di V8; era il coefficiente dell'arco inverso di segno sbagliato), la **scala dei gradi** (grado medio 28,3 contro 12,9, con il reale a 32,05), la **coda** (grado massimo 324 contro 102) e — dopo la correzione — anche la **coesione**, che prima sbagliava nel verso opposto: transitività 0,0209 contro 0,107 di prima (reale 0,034), clustering 0,043 contro 0,141 (reale 0,080), diametro 6 come il reale.
- **Il confronto onesto con le altre famiglie** (`src/variant_scorecard.py` → `results/variant_scorecard.csv`): messo accanto a **v2**, **v6a**, **v6b** e **v7c** — non solo a V8 — V9 **non è la più vicina su ogni metrica**, e va detto. La densità resta di **v8** (0,00129 contro 0,00125 del reale; V9 sta a 0,00286), la reciprocità di **v6a/v6b** (0,420 contro 0,4177, ma **imposta**, non imparata), transitivity e clustering di **v6b** (0,0258 e 0,0978 contro 0,0337 e 0,0804), grado massimo e modularità di **v2** (1.008 e 0,357), e il **grado medio è l'unica metrica dove vince V9** (28,30 contro 32,05). Il punto a favore di V9 è che è l'unica a prendere **contemporaneamente** scala dei gradi, coesione e verso della coda **senza imporre statistiche**: v6 e v2 vincono dove le statistiche del reale sono copiate per costruzione.
- **La misura che mancava, la pendenza della coda** (stessa tabella; CCDF del grado uscente sull'ultimo decile, soglia e numero di punti nel CSV): reale **-2,35** (esponente **3,35**), v2 -2,25, v6b -2,09, **V9 -3,33** (esponente **4,33**), v8 -3,63. La coda di V9 è quindi **troppo leggera** e quella di v2/v6b **appena troppo pesante**: il difetto che prima si vedeva solo nel grado massimo (324 contro 1.290) ora è un numero confrontabile.
- **Il volume non è un difetto**: gli archi sono il grado medio per il numero di utenti, quindi i 280.298 di V9 su 9.904 utenti corrispondono a circa **317.000** alla scala del reale (10.000 utenti al grado medio reale), mentre i 824.045 reali sono 32,05 × 25.714. La differenza è la **popolazione**, non la forma della rete: su quel riferimento V9 sta a **0,88**, V8 a 0,40 e v2/v6b a 1,29–1,32.
- **Cosa resta sotto**: dopo la correzione gli errori sono tutti **dello stesso segno e contenuti** (0,25–0,9× del reale): modularità 0,106 contro 0,275, grado massimo 324 contro 1.290, new→new 12,4% contro 8,1%. Il filo comune è la **preferenza per l'hub**, che sottostima — il grado dell'estremo già presente è 169 contro 437 del reale, 0,39× — e viene dal trasferire i percentili di grado da 25.714 utenti reali a 10.000 sintetici: è il Rischio 1 del §7.
- **Una leva scartata**: rifare il fit con 100 negativi per positivo invece di 10 non cambia la coesione e costa cinque volte tanto → configurazione riportata a 10.
- **Un'idea di Gemini, provata e scartata (esito negativo)**: i **negativi misti** — 30% inversioni asimmetriche (la coppia al contrario di un positivo, solo se quell'arco inverso non esiste davvero), 30% difficili sul grado (candidati del decile più connesso del pool) e 40% uniformi — sono l'unica sua proposta con un effetto meccanico plausibile, quindi è stata messa in controllo (`--negatives stratified`, cartella `generated_snapshots_v2_v9_strat`, coefficienti in `results/v9_strat_coefficients.csv`). Peggiora tutto quello che conta: **transitività 0,0115** contro 0,0209, **clustering 0,0305** contro 0,043, archi **fra co-arrivati 18,4%** contro 12,4% (reale 8,1%), **grado dell'estremo già presente 104** contro 169 (reale 437), grado massimo 283 contro 324, reciprocità **0,300** contro 0,3035. I coefficienti invece si spostano molto — "la controparte è nuova" da +1,22 a **+0,61**, "grado logaritmico" da +1,02 a **−0,21**, "uscente" da +0,14 a **+0,47** — ma la **reciprocità non si muove**, ed è il punto: in V9 non la impara il classificatore, la **genera la tabella** delle coppie ricambiate. L'AUC di fit scende da 0,918 a 0,767, ma misura un'altra distribuzione di negativi e quindi non è confrontabile.
- **Riproducibilità**: la variabilità fra run (~2% sul volume) aveva una causa precisa — l'**ordine di iterazione dei set di archi** (stringhe) non è fissato in Python, quindi la sequenza casuale a valle cambiava da un processo all'altro. I generatori ora **iterano gli archi in ordine ordinato** e la ripetizione dello stesso comando dà lo stesso risultato a meno di **qualche unità**: il controllo che rilancia la configurazione documentata trova **280.299** archi invece di 280.298 — un arco su 280 mila — con gli **stessi** 8.293 eventi di arrivo, lo stesso `edges_per_event` (33,799), la stessa quota ricambiate (0,4074) e la **stessa reciprocità** (0,3035). La differenza non sta nel generatore — che è deterministico e, con `--negatives uniform`, non consuma nessuna estrazione in più — ma nei **pesi della rete addestrata**, che variano al terzo decimale fra processi (contenuto 0,047112 in `results/v9_check_coefficients.csv` contro 0,048790 in `results/v9_coefficients.csv`). Per questo le tabelle si citano come **un'unica esecuzione** e non come valori ripetibili al bit. **Tutte le tabelle di questa sezione e le figure vengono da un'unica esecuzione finale** (V8 e V9, tre modalità di contenuto ciascuno, con le stesse analisi), quindi sono confrontabili fra loro e **ricostruibili** con i comandi del §4.
- **Il controllo A/B/C su V9**: testo autentico, permutato e assente danno reti **indistinguibili** anche in questo meccanismo (archi 280.298 / 274.448 / 280.292; reciprocità 0,3035 / 0,3021 / 0,3037; transitività 0,0209 / 0,0207 / 0,0211; modularità 0,106 / 0,106 / 0,105; omofilia 0,999 / 1,002 / 0,999) e, per la quarta volta, l'ordine non è monotono: la conclusione del §6 vale su **entrambi** i meccanismi, quello per coppie e quello per eventi.

- **Raggiunto**: la densità coincide con quella reale (0,00129 contro 0,00125) e il grado medio è a 2–2,5× dal reale, **senza imporre statistiche** (v7 si fermava a 0,0004 e 3,6–5,8). Il piano è stato eseguito come scritto: candidati decisi dalla misura sul reale, attività dal reale, campionamento, retroazione, ablazione.
- **Non raggiunto, dichiarato**: coda dei gradi (102–128 contro 1.290), reciprocità (0,004 contro 0,418), transitivity (0,003 contro 0,034).
- **Il contenuto non porta il legame (esperimento controllato, non più solo ablazione)**: nelle tre modalità A/B/C le proprietà relazionali sono **indistinguibili** (reciprocità 0,0031 / 0,0030 / 0,0029; transitività 0,00269 / 0,00253 / 0,00262; grado massimo 107 / 96 / 106; modularità 0,154 / 0,159 / 0,156) e il volume varia di pochi punti percentuali **senza ordine** (`permuted` sta più vicino a `off` che a `real`). L'ablazione per AUC (0,9818 → 0,9816) va letta come strumento **saturo**: la prova è il confronto fra reti.
- **L'unico effetto del testo è minimo**: l'omofilia passa da 0,9998 (testo permutato) e 1,0004 (senza testo) a **1,0009** con testo autentico — circa lo 0,1% del divario che nel reale è 2,13.
- **La scomposizione del 2,13** (`results/homophily_control.csv`, ultimo snapshot): contro coppie a caso **2,133**; contro non-archi appaiati per grado **1,511**; per grado esatto ±10% **1,290**; per grado+volume di testo 1,367; per volume di testo 1,554. Gli estremi degli archi hanno grado medio **403,6** contro **61,7** e **1.297 caratteri** cumulativi contro una mediana di **323**. Il 2,13 è quindi **in gran parte posizione nella rete** (chi viene collegato è chi è molto connesso e scrive molto, e i profili lessicali dei grandi utenti si somigliano da soli), con un **residuo lessicale piccolo** (AUC ≈ 0,575–0,585). Attenzione a non confondere le rappresentazioni: il residuo vive nei profili **TF-IDF+SVD cumulativi**, non nel canale del generatore (**MiniLM per post**, dove il segnale è nullo) — il 2,13 e l'1,00 del modello non misurano la stessa cosa. I controlli fini sono instabili per passo (in un caso il gruppo appaiato è più simile degli archi: 0,42 contro 0,31), quindi 1,29 va preso come limite inferiore.
- **Il cambio di scenario conta meno del previsto**: ~5% di volume, nessuna metrica strutturale diversa.
- **Il coefficiente negativo dei vicini in comune (−6,1 nell'ultima esecuzione; −4,4 in run precedenti)** è spiegato: i negativi sono coppie a caso e gli archi veri coinvolgono entranti, che non hanno vicini in comune.
- **Il ramo "i post come archi" è chiuso con esito negativo** (racconto completo in `docs/pipeline.md` §6): il testo di un singolo post non separa le coppie collegate da quelle appaiate (AUC 0,515–0,517 contro 0,5166 della media), i post di una coppia reciproca non si somigliano più del caso, e chi scrive più post non riceve più archi. La parte strutturale dell'idea (co-interazione sullo stesso post) non è verificabile: il dataset non registra *quale* post è agganciato a ogni arco. Artefatti: `src/build_edge_posts.py`, `src/edge_content_test.py`, `results/edge_*_report.csv`, `results/reciprocity_content.csv`.
- **Cosa servirebbe per andare oltre** (la direzione che V9 ha già iniziato con l'evento di arrivo): la **dipendenza fra i bersagli scelti dentro lo stesso evento**, la **preferenza per l'hub** — l'unica leva residua, sottostimata nel trasferimento — e una rappresentazione del contenuto che agisca **a livello di coppia** invece che di singolo utente.
