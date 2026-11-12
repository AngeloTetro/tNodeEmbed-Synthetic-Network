# pipeline.md — la pipeline definitiva

## 1. Impianto

- Si **addestra sul dataset reale** e si fa **inferenza sul dataset sintetico**.
- Solo collegamenti sintetico↔sintetico.
- L'**unica feature comune ai due dataset è il `content`** dei post: il reale ha attributi del post,
  il sintetico attributi dell'utente, e l'unica cosa equiparabile è il testo.
- Il sintetico **non ha l'accrescimento** del reale, quindi va **partizionato in 9 snapshot** per
  ricreare un'evoluzione analoga. Due partizioni provate: **ingresso progressivo** con la curva di
  arrivo del reale (ufficiale) e **tutti presenti dal primo passo** (controllo).

## 2. I due dataset

| | reale | sintetico |
|---|---|---|
| utenti | 25.714 nodi | 10.000 |
| post | 173.475 cumulativi / 154.580 per periodo | 100.474 righe nei batch (100.456 con `id` valido) |
| archi | 824.045 diretti | nessuno |
| tempo | 9 snapshot (2022 → luglio 2025), split **6 training / 1 validation / 2 test** | nessuna data: 4 batch con ordine casuale, `id` in sequenza densa; il tempo lo imponiamo noi (9 snapshot) |
| per utente | solo i post (`content`) | 11 attributi dell'utente, più `format_style` (del *post*, non dell'utente) e i post |

Nessun file originale viene modificato: le cartelle `generated_*` sono aggiunte nostre.

**I 4 batch nativi non sono periodi** (`src/consolidate_native.py` → `results/native_batches.csv`):
`batch_0/1/2` hanno 33.068 / 33.068 / 34.318 post, coprono **tutto** l'intervallo degli id
(median ~ 49,500-51,600) e condividono ~9.500 utenti su ~9.700; `batch_3` ha 2 righe. Sono tre terzi
della **stessa** popolazione. Da qui la **consegna doppia**: i 9 file `.edg` (uno per passo) e i
`native_4_batch_<i>.edg`, cioè la rete **ristretta agli utenti** di quel batch (93-97% degli archi),
più `native_4.csv`. È una restrizione per utenti, non una suddivisione per periodi. Il quarto file è
**vuoto per costruzione**: `batch_3` contiene due righe, quindi nessun utente.

## 3. Il tempo è la struttura portante: tNodeEmbed

Ogni snapshot è un time step (9 nel reale, 9 nel sintetico). La pipeline è quella del paper, **senza
node2vec**, con il contenuto al posto degli embedding statici:

1. **rappresentazioni per passo** — per ogni snapshot e ogni utente, il vettore del contenuto dei suoi
   post fino a quel passo (encoder scelto al §4); sostituiscono la matrice `Q_t` appresa del paper;
2. **allineamento** — Procrustes ortogonale a catena fra snapshot consecutivi, stimato sui nodi comuni;
3. **LSTM** — comprime la sequenza dei passi nell'embedding temporale del nodo;
4. **testa di coppia** — `[h_u, h_v, h_u ⊙ h_v]`, addestrata end-to-end sulla link prediction.

Il modello si allena sulle 5 transizioni di training, si controlla su 1 e si misura su 2; poi si
applica al sintetico **congelato** (encoder, Procrustes, LSTM e testa non si riaddestrano).

## 4. Il contenuto e l'encoder

Nove candidati, **sei eseguiti**: `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `bge-base-en-v1.5`,
`gte-small`, `gte-base`, più `tf-idf+svd` come riferimento classico. `bge-large`, `GIST-small` e
`mxbai` non sono stati eseguiti.

**Come si sceglie**: pipeline completa sul reale con ciascun encoder, **negativi appaiati**, media
dell'AUC su validation e due test. Vince **MiniLM (0,534)**, davanti a `gte-base` (0,532) e
`bge-base` (0,530); con i negativi casuali i valori sono ~0,99 in validation e 0,95-0,96 nei test,
ma non è quella la colonna della scelta.

**Per post, non concatenato (misurato)**: unire i post in una stringa faceva tagliare il testo al
limite di token (MiniLM: 256), con il 28% dei token sopravvissuti nei periodi più attivi e perdita
concentrata sugli hub. `--pooling mean_of_posts` codifica ogni post e media: MiniLM 0,534 → **0,549**,
bge-base 0,530 → 0,540, tf-idf 0,505 → 0,524.

**Leve scartate** (dettaglio in `archive/storico.md`): fine-tuning contrastivo dell'encoder (0,519
contro 0,534), feature dei nodi addestrabili (0,512 contro 0,534), `Q_t` completo del paper (~70M
parametri, non entra in 6 GB), "post come arco" (AUC 0,515-0,517).

## 5. La generazione: V9, eventi di arrivo

L'unità non è la coppia ma **l'utente che entra**. Nasce da un fatto misurato: nel reale **nessun
arco nuovo collega due utenti già presenti** (`results/accretion_real_snapshots.csv`, quota
"vecchio→vecchio" 0,0000% in tutte le transizioni), quindi ogni arco appartiene all'evento del suo
estremo nuovo. Un evento porta in media da **35 a 111** archi secondo la transizione (mediana 6-32,
massimo **1.202**), il **51-63%** è uscente e ~**40%** delle coppie è ricambiato dentro l'evento.

La generazione, per ogni passo e per ogni entrante:

1. **k** archi dall'evento, campionato dalla tabella reale dei k normalizzati e riscalato su
   `popolazione^esponente` (`--exponent 1`, configurazione documentata);
2. la **direzione** (uscente/entrante) con la quota reale del passo;
3. il **partner** con una logistica a sette variabili, fittata sul reale: `content_score` (uscita
   della testa di coppia), `partner_degree_log`, `partner_percentile`, `partner_new`,
   `partner_active`, `outgoing`, `partner_co_occurrence`. Pesi: contenuto **+0,05**, novità
   **+1,22**, grado log **+1,02**, percentile **+0,91**, co-occorrenza +0,19, uscente +0,14,
   attività −0,03;
4. le coppie **ricambiate** pescate fra gli uscenti già scelti, con la tabella reale per direzione e
   classe di grado.

Lo stato (gradi, percentile, chi è nuovo, chi ha scritto, vicinato) si ricalcola **a ogni passo**
sulla rete appena generata.

**Numeri della rete ufficiale** (ultimo snapshot, testo autentico): 9.897 nodi con archi, 271.646
archi, densità 0,002774, grado medio **27,45** (reale 32,05), mediano 10 (4), massimo **274**
(1.290), reciprocità **0,3224** (0,4177), transitività **0,0212** (0,0337), clustering 0,0420
(0,0804), modularità 0,1074 (0,2751), omofilia del contenuto 1,00 (2,13), diametro 6.

**Controllo "tutti presenti"**: 9.901 nodi, 280.298 archi, grado medio 28,31, massimo 328,
reciprocità 0,3035, transitività 0,0209, clustering 0,0435, modularità 0,1030, omofilia 0,999,
diametro 6.

**Perché la progressiva è ufficiale**, e cosa costa: nel reale il 41,8% degli utenti finali è già
collegato al primo snapshot e il restante 58% entra durante il periodo, quindi una rete che nasce deve
avere arrivi. Il prezzo è dichiarato: grado medio 27,45 contro 28,31, massimo 274 contro 328, archi
fra utenti arrivati nello stesso passo **17,0%** contro 12,4% (reale 8,1%), e primo passo con il
**15,7%** dei nodi collegati contro il 41,8% del reale. Le due reti restano equivalenti sulla
distanza relativa dal reale (3,84 contro 3,89 su sette metriche, definizione in
`src/scorecard.py`), e invertendo la normalizzazione l'ordine si inverte: non è un criterio.

**Cosa si consegna, e cosa no.** Delle tre modalità di contenuto si consegna solo quella con **testo
autentico**, per la rete ufficiale e per il controllo: quattro cartelle (i due dataset originali, la partizione e la rete ufficiali); il controllo e' in
`archive/dataset/`. Le reti con testo permutato e contenuto azzerato sono un **controllo**, non un
risultato: i loro numeri restano nei CSV (`results/*_v9_permuted.csv`, `*_v9_off.csv`) e si rigenerano
in pochi minuti con `--content permuted` e `--content off`.

## 6. Cosa gira, e cosa produce

| script | cosa fa | output |
|---|---|---|
| `src/content_encoders.py` | interfaccia unica per gli encoder | — |
| `src/tnodeembed_real.py` | pipeline tNodeEmbed sul reale, un giro per encoder | `results/encoder_comparison.csv` |
| `src/build_synthetic_snapshots.py` | partizione del sintetico in 9 passi (tutti presenti / arrivo progressivo) | `synthetic_all_present_posts/`, `synthetic/` |
| `src/generate_arrival.py` | **V9**: fit sul reale e generazione per eventi di arrivo (`--content {real,permuted,off}`, `--exponent`, `--closure`, `--negatives`) | `results/network_report.csv`, `results/coefficients_<cartella>.csv`, `tnodeembed_embeddings.npz` |
| `src/analyze_structure.py <cartella>` | nodi, archi, densità, gradi, k-core, reciprocità, transitività, clustering, componenti, diametro, modularità | `results/structure_*.csv`, `structure_heavy_*.csv` + figure |
| `src/analyze_temporal.py <cartella>` | delta per passo, crescita, densificazione, arrivi, persistenza (Jaccard), stabilità delle comunità (NMI) | `results/temporal_*.csv` + figure |
| `src/analyze_content.py <cartella>` | omofilia del contenuto, similarità dentro/fuori le comunità, comunità vs attributi | `results/content_*.csv`, `content_attributes_*.csv` + figure |
| `src/analyze_homophily.py [cartella]` | scompone l'omofilia: casuale, grado, grado ±10%, arrivo+grado, volume di testo, grado+volume | `results/homophily_control_<cartella>.csv` |
| `src/analyze_profiles.py <cartella>` | profili (solo sintetico): `political_leaning`, `profession`, `age_interval` da contenuto, da embedding e da contenuto+rete | `results/profiles_*.csv` + figure |
| `src/check_real_accretion.py <cartella>` | archi nuovi per estremi (nuovo/presente) e grado dell'estremo già presente | `results/accretion_*.csv` |
| `src/scorecard.py` | reale contro ufficiale e controllo, con distanza relativa dichiarata e pendenza di coda | `results/scorecard.csv`, `.md` |
| `src/tail_scaling.py` | legge di scala del reale (grado massimo contro utenti) e confronto a pari dimensione | `results/tail_scaling_real.csv`, `tail_scaling_summary.csv` |
| `src/measure_arrivals.py` | misura degli eventi di arrivo del reale: archi per evento, quota uscente, ricambiati | `results/arrival_events.csv` |
| `src/consolidate_native.py <cartella>` | consegna nei 4 file nella forma dei batch nativi | `results/native_batches.csv`, `<cartella>/native_4_batch_<i>.edg` |
| `src/compare_folders.py` | confronto grafico fra cartelle | `results/figures/compare_*.png` |
| `src/plot_pipeline.py` | schema della pipeline | `results/figures/paper_pipeline.png` |
| `src/graph_io.py` | libreria di lettura (edge list, post, attributi) | — |

Le famiglie abbandonate (regole a mano, v6, v7, V8) hanno gli script in `archive/src/` e i numeri in
`archive/storico.md`.

## 7. Note, vincoli e limiti noti

- **Quale metrica, per quale domanda**: la rete generata **non si valuta con un AUC** — nel sintetico
  non ci sono archi veri — ma con statistiche di rete per snapshot contro il reale. L'AUC resta
  corretta sul reale: fit del modello e diagnostica del contenuto. L'AUC di ablation è **satura**
  (0,9818 → 0,9816) e non misura il contributo del testo; per quello valgono il controllo a testo
  permutato e la scomposizione dell'omofilia.
- **Restrizione del paper**: applicata al reale produce **0 archi** in tutte le transizioni; per
  questo la valutazione usa l'intera transizione e due modalità di negativi (casuali e appaiati).
- **Negativi casuali contro appaiati**: i primi sono facili (~0,99 in validation), i secondi sono la
  prova onesta (0,51-0,57) e sono la colonna di scelta. Gli appaiati hanno stesso mittente, stessa
  classe di arrivo e grado vicino (prossimità in una lista ordinata per arrivo e grado).
- **Il contenuto non decide chi si collega a chi**: le tre reti ufficiali (testo vero, permutato,
  azzerato) sono indistinguibili — archi 271.646 / 261.680 / 271.640, reciprocità 0,3224 / 0,3229 /
  0,3227, omofilia ~1,00. Il termine di contenuto pesa +0,05 contro +1,22 della "controparte nuova".
- **L'omofilia del reale è in gran parte struttura** (`results/homophily_control_real_snapshots.csv`):
  contro coppie a caso **2,133**; contro non-archi appaiati per grado **1,511**; per grado ±10%
  **1,290**; per volume di testo 1,554; per grado+volume 1,367. Gli estremi degli archi hanno grado
  medio 403,6 contro 61,7 e 1.297 caratteri contro una mediana di 323. Sulla rete ufficiale lo
  stesso protocollo dà **1,001 / 1,000 / 0,980 / 1,001 / 1,014**
  (`results/homophily_control_synthetic_progressive.csv`): nessuna omofilia di contenuto.
- **Legge di scala**: il reale prevede ~339 di grado massimo a 10.000 utenti (esponente 1,445,
  R² 0,986). La rete ufficiale sta a **274 (81%)**, il controllo a **328 (97%)**. È il costo più
  visibile della partizione progressiva.
- **Coda**: pendenza CCDF sull'ultimo decile −3,98 (ufficiale), −3,33 (controllo), −2,349 (reale, che
  al primo snapshot con 10.755 utenti sta a −1,69). A pari dimensione la coda generata è **più
  ripida**: gli hub sono meno estremi del vero.
- **Il primo passo della progressiva è più vuoto del reale** (15,7% dei nodi collegati contro 41,8%).
  Non è un errore di misura: la legge di k si applica alla popolazione presente, che al primo passo è
  solo quella entrata. La legge è però fittata su popolazioni fra 10.755 e 21.082, quindi a 10.000 e
  sotto è un'**estrapolazione**. La riga "nodi collegati al primo passo" del reale misura uno stock
  preesistente, non un passo di nascita, e non va usata come criterio.
- **Riproducibilità**: i generatori iterano gli archi in ordine ordinato; il generatore è
  deterministico (stessi eventi, stessa quota di ricambiati), l'addestramento no: i pesi variano al
  terzo decimale e il volume di ~10 archi su 271.646 (0,003%). Le tabelle vengono da un'unica
  esecuzione.
- **Bug trovato dalla diagnostica, tenuto come metodo**: la prima versione campionava i negativi del
  fit solo fra i presenti, quindi "la controparte è nuova" separava le classi da sola (peso +1,38) e
  il 99,95% degli archi finiva fra co-arrivati; corretto con negativi dallo stesso pool della
  generazione, la transitività è passata da 0,107 a 0,0209.
- **Scenario del sintetico**: curva di arrivo e forma di attività vengono dal reale. La *curva
  aggregata dei post* del reale **non** è utilizzabile (gli snapshot hanno durata diversa: da un anno
  a tre giorni) e copiarla concentrava le raffiche sugli stessi passi.
- **Utenti reali**: nessun profilo né attributo oltre ai post. Il dataset reale non va modificato.
  `exact_level_found` nel sintetico deriva da un progetto sulla tossicità e non si usa.
- **Righe scartate**: 18 righe dei batch sintetici non hanno il numero di campi dell'intestazione.
- **Tracciabilità**: ramo `levers/posts-as-edges`, `main` allineato, tag `consegna-progetto`.

## 8. Convenzioni

- File `.md` in italiano; script, output, CSV e figure **in inglese**.
- `Esame Angelantonio Fedele Murolo/` e `tNodeEmbed/` sono read-only (le `generated_*` sono nostre).
- Seed fissi; ambiente `uv`.
