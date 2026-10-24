# Archivio dei tentativi precedenti

Questo file documenta tutto ciò che è stato provato prima della configurazione finale. Nessuna
delle reti qui descritte viene consegnata: i numeri restano in questo documento e i CSV originali
in `archive/results/`, gli script che le rigenerano in `archive/src/`.

Il progetto finale usa **una sola pipeline, V9**, su due partizioni:

- **ufficiale** — V9 con ingresso progressivo (`generated_snapshots_growth_v2_v9`), scelta per
  realismo degli arrivi;
- **controllo** — V9 con tutti gli utenti presenti dal primo passo (`generated_snapshots_v2_v9_real`).

Tutti i numeri di questo file sono sull'ultimo snapshot e, salvo dove indicato, sulla vecchia
partizione ad attività uniforme: sono storici, non vanno confrontati riga per riga con le reti
finali.

## 1. Famiglie di generazione

La domanda che le distingue è una sola: **da dove nasce il numero di archi**.

| variante | meccanismo | perché è stata abbandonata | archi | grado medio | grado max | reciprocità | fonte |
|---|---|---:|---:|---:|---:|---|---|
| riferimento (solo contenuto) | punteggio di tNodeEmbed su tutte le coppie + soglia | nessun hub, nessuna reciprocità, coesione fuori scala; l'omofilia resta 1,00 | 107.511 | 10,75 | 28 | 0,0011 | `transfer_report.csv` |
| v1 | soglia quantilica con il vincolo di ~1,2 archi per utente attivo | il conteggio è deciso a mano | 104.672 | 10,47 | 183 | 0,0037 | `edge_rules_report.csv` |
| v2 | grado uscente di ogni utente campionato dalla distribuzione reale | copia la statistica invece di impararla; massimo 2,9× la legge di scala | 414.985 | 41,50 | 1.008 | 0,0017 | `edge_rules_report.csv` |
| v3 | v1 + inversione forzata fino a reciprocità 0,42 | reciprocità imposta, hub degenere | 138.582 | 13,86 | 5.959 | 0,4480 | `edge_rules_report.csv` |
| v4 | vincolo di accrescimento, bersagli ∝ grado precedente | troppo rada, reciprocità quasi nulla | 82.860 | 8,29 | 24 | 0,00017 | `edge_rules_report.csv` |
| v5 | v2 + v3 + v4 insieme | "la migliore regola a mano": centra la reciprocità per costruzione | 427.056 | 42,71 | 998 | 0,4946 | `edge_rules_report.csv` |
| v6a | bisezione sul quantile per centrare il grado medio finale reale (32,05) | statistica imposta, non appresa; transitività 0,00015 | 64.422 | 6,44 | 1.885 | 0,4200 | `v6_report.csv` |
| v6b | bisezione per riprodurre la curva di crescita reale (×0,389) | stessa cosa: il conteggio viene dal reale per costruzione | 390.737 | 42,38 | 1.709 | 0,4200 | `v6_report.csv` |
| v7a | decisione bayesiana: arco se p > 0,5 | con un prior di 6,4 coppie collegate ogni 100.000 non produce **nessun** arco | 0 | — | 0 | — | `v7_report.csv` |
| v7b | tasso reale applicato per blocchi | la soglia per blocco rompe la simmetria (u,v)/(v,u) | 61.560 | 6,16 | 77 | 0,0158 | `v7_report.csv` |
| v7c | soglia globale sul tasso reale | resta un modello di solo contenuto: troppo rada | 57.558 | 5,76 | 920 | 0,183 | `v7_report.csv` |
| v7d | v7b con encoder fine-tuned | il fine-tuning non passa il criterio di tenuta | 61.588 | 6,16 | 76 | 0,0023 | `v7_report.csv` |
| v7e | v7d con soglia globale | hub degenere: un utente collegato a 9.999 | 57.559 | 5,76 | 9.999 | 0,7209 | `v7_report.csv` |
| V8 | modello pairwise a 11 feature, campionamento, correzione del prior, stato ricalcolato | superata da V9: reciprocità 0,0031, transitività 0,0027, massimo 102 | 129.321 | 12,93 | 102 | 0,0031 | `v8_report.csv` |
| V8 A/B/C | testo vero / testi scambiati / contenuto azzerato | controllo: le tre reti sono indistinguibili | 128.831 / 123.294 / 124.560 | 12,88 / 12,33 / 12,46 | 107 / 96 / 106 | 0,0031 / 0,0030 / 0,0029 | `v8_report.csv` |

Riferimento per tutte: rete reale, ultimo snapshot, 824.045 archi, grado medio 32,05, massimo
1.290, reciprocità 0,4177, transitività 0,0337, clustering 0,0804, modularità 0,2751, diametro 6.

## 2. Leve provate dentro la famiglia finale

| leva | cosa è stato provato | esito | numeri |
|---|---|---|---|
| fine-tuning dell'encoder | perdita contrastiva su archi di training, negativi appaiati, temporali e in-batch, early stopping su un AUC proxy | **respinto**: peggiora anche in validation | MiniLM 0,534 → 0,519 di media val+test; bge-base 0,530 → 0,523 (`finetune_report.csv`, `encoder_comparison.csv`) |
| feature dei nodi addestrabili | proiezione condivisa inizializzata dal testo, addestrata con il task (versione ridotta del Q_t del paper) | **respinto**: overfitta | 0,512 contro 0,534 (test2 0,476) |
| Q_t completo del paper | una riga per utente e snapshot, ~70M parametri | **mai eseguito**: non entra in 6 GB | — |
| "post come arco" | il contenuto del singolo post come relazione fra due utenti | **ramo chiuso**: il testo di un post non separa le coppie collegate | AUC 0,515–0,517 contro 0,5166 della media; reciprocità instabile 0,18–0,66 (`edge_posts_report.csv`) |
| negativi misti nel fit | 30% inversioni asimmetriche, 30% grado alto, 40% uniformi | **respinto**: peggiora coesione e co-arrivi | transitivity 0,0115 contro 0,0209; co-arrivi 18,4% contro 12,4% (`v9_strat_coefficients.csv`) |
| numero di negativi nel fit | 100 per positivo invece di 10 | **respinto**: nessun guadagno, costo ×5 | modularità 0,856 contro 0,854; transitivity 0,114 contro 0,107 |
| chiusura triadica | frazione delle altre controparti invece del flag binario | **non adottata**: il coefficiente va a rumore e la rete non cambia | +0,194 → +0,051; 280.310 archi contro 280.298 (`v9_closure_coefficients.csv`) |
| esponente della scala | 1,79 (pendenza fittata sul reale) invece di 1 | **scartato**: densità centrata ma grado massimo dimezzato | 122.608 archi, massimo 156 contro 328 attesi, transitivity 0,00998 (`v9_slope_coefficients.csv`) |
| pooling dei post | media dei vettori per-post invece di un testo concatenato | **adottato** (unico miglioramento): il troncamento tagliava il 72% dei token degli utenti più attivi | MiniLM 0,534 → 0,549; bge-base 0,530 → 0,540 |
| soglia contro campionamento | tagliare le coppie sopra una soglia (famiglia v6–v7) contro campionare senza soglia (V8, V9) | **campionamento adottato** | v7c 5,76 di grado medio contro 28,30 di V9 |
| partizioni temporali | tutti presenti dal primo passo contro ingresso progressivo | la progressiva è diventata **ufficiale** per realismo degli arrivi; la tutti-presenti è il **controllo** | v. documento di pipeline |

## 3. Esperimenti mai eseguiti

- Tre encoder candidati mai lanciati: `bge-large`, `GIST-small`, `mxbai-embed-large`.
- Baseline statica node2vec (solo ultimo snapshot): prevista dal contratto del paper, mai portata.
- Task cold-start iniziale (dare a un entrante metà dei suoi archi e prevedere l'altra metà): esiste
  solo nella conversazione di progetto, senza CSV.
- Il confronto `content=period` contro `content=cumulative`: tutte le righe di
  `encoder_comparison.csv` sono `period`, la variante cumulativa non è mai stata misurata.

## 4. Numeri da non riusare

Documentati qui perché circolano in vecchi documenti e vanno corretti prima di citarli:

- **"distanza 3,79 contro 4,39"** fra le due partizioni: non si riproduce dalla scorecard
  archiviata, e le colonne `dist_*` di quel CSV non sono omogenee (densità su scala ×10 in
  entrambe le righe, transitivity e omofilia ×10 solo su una). Ricalcolata con una distanza
  relativa simmetrica sulle stesse metriche dà 3,82 e 3,84: le due partizioni sono di fatto in
  parità.
- **grado massimo della V9 "tutti presenti"**: 324 nelle tabelle, 328 in `tail_scaling_summary.csv`,
  e la percentuale sulla legge di scala è 95% in un documento e 97% negli altri.
- `variant_summary.csv` etichetta la run `v9_strat` come "riferimento (solo contenuto)".
- `v9_report.csv` contiene circa venticinque righe, comprese le run difettose precedenti alla
  correzione dei negativi (985.258 archi, reciprocità 0,6431) prive di etichetta.
