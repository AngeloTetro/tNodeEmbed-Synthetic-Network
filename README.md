# Ricostruire una rete sociale dal testo dei post

Il progetto ricostruisce la **rete di collegamenti** di una popolazione di cui si osserva soltanto il
**testo dei post**. Il meccanismo con cui nasce un legame si impara da una rete sociale reale e si applica
a una popolazione sintetica che non ha archi e non ha date.

Il risultato non è una previsione di archi: è un **generatore**. Dato il testo dei post e l'ordine con cui
gli utenti entrano nella piattaforma, produce una rete diretta con le stesse proprietà statistiche della
rete reale (grado, reciprocità, transitività, clustering, modularità) senza imporre nessuno di quei numeri.

![La pipeline](results/figures/pipeline_diagram.png)

---

## Il risultato in numeri

Ultimo snapshot, rete reale contro le due reti generate dalla stessa pipeline su due partizioni temporali
del sintetico.

| metrica | reale | progressiva | tutti presenti |
|---|---:|---:|---:|
| utenti con almeno un arco | 25.714 | 9.897 | 9.901 |
| archi | 824.045 | 271.646 | 280.298 |
| grado medio | 32,05 | 27,45 | 28,31 |
| grado massimo | 1.290 | 274 | 328 |
| reciprocità | 0,4177 | 0,3224 | 0,3035 |
| transitività | 0,0337 | 0,0212 | 0,0209 |
| clustering medio | 0,0804 | 0,0420 | 0,0435 |
| modularità | 0,2751 | 0,1074 | 0,1030 |
| diametro | 6 | 6 | 6 |
| distanza relativa dal reale | — | **3,84** | 3,90 |

La rete **progressiva** è quella presentata: gli utenti entrano nel tempo seguendo la curva di arrivo
misurata sul reale. Quella con **tutti presenti dal primo passo** è il confronto. Sulle sette metriche
confrontabili le due sono equivalenti, e invertendo la normalizzazione l'ordine si inverte: la scelta è
concettuale, non metrica. Una rete che nasce ha degli arrivi.

Sul contenuto il risultato va in direzione opposta all'intuizione, ed è la parte più interessante: il
testo **identifica** l'utente (dal testo si predice l'orientamento politico con AUC 0,827 e la professione
con 0,867), ma **non decide con chi si collega**. Il termine di contenuto pesa +0,05 nel modello di scelta
del partner, contro +1,22 di "la controparte è nuova" e +1,02 del logaritmo del grado. L'omofilia del
contenuto che sembra contraddire questa lettura (2,13 sul reale) scende a 1,29–1,55 quando il confronto
tiene fermo il grado: è in gran parte posizione nella rete.

---

## La pipeline

1. **Rappresentazione del contenuto.** Ogni post è codificato singolarmente da un encoder di frase (sei
   confrontati; `all-MiniLM-L6-v2`, codifica per post) e i vettori di un utente sono mediati in una
   rappresentazione per utente e per istantanea. Il testo è l'unica colonna presente in entrambi i dataset.
2. **Modello temporale.** `tNodeEmbed` **senza node2vec**: catena di allineamenti di Procrustes fra
   istantanee consecutive, LSTM su nove passi, testa di coppia su `[h_u, h_v, h_u ⊙ h_v]`. node2vec è
   stato tolto perché si inizializza osservando i collegamenti, mentre qui ogni arco nuovo tocca un utente
   che entra: proprio dove servirebbe vedere, un modello strutturale è cieco.
3. **Generatore per eventi di arrivo.** L'unità non è la coppia ma **l'utente che entra**: si porta dietro
   `k` archi (tabella appresa sul reale e riscalata sulla popolazione), la direzione e la quota di coppie
   ricambiate; il partner si sceglie con una regressione logistica a sette variabili (grado, percentile di
   grado, novità, attività, direzione, co-occorrenza, punteggio di contenuto) più rumore di Gumbel, per
   campionare invece di prendere il massimo. Nessuna soglia e nessun conteggio imposto: volume, gradi e
   poli emergono.
4. **Valutazione.** Le statistiche di rete sono la misura primaria; l'AUC è diagnostica e vive solo sul
   reale, perché su una rete generata non ci sono archi veri da prevedere.

---

## Repository

```
dataset/                          i due dataset e le due reti generate (sei cartelle)
├── real_snapshots/                    reale: 9 istantanee, post e archi
├── synthetic_snapshots/               sintetico originale: 4 batch nativi
├── synthetic_progressive_posts/       ingresso: post distribuiti su 9 passi
├── synthetic_all_present_posts/       idem, tutti presenti dal primo passo
├── synthetic_progressive/             uscita: archi generati, embedding, batch nativi
└── synthetic_all_present/             idem, versione di confronto
src/
├── paths.py                       percorsi condivisi da tutti gli script
├── lib/                           lettura dati, encoder, tNodeEmbed, pipeline temporale
├── pipeline/                      partizioni, generazione, consolidamento
├── analysis/                      struttura, temporalità, contenuto, omofilia, profili, scorecard
└── figures/                       le figure
results/
├── tables/                        33 CSV citati dal documento
└── figures/                       17 PNG, tutte rigenerate da src/figures/
document.pdf                       il documento finale, in inglese
```

Ogni cartella di dati espone la stessa interfaccia, `snapshot_i/social_network.edg` più i due file dei
post. È così che lo stesso codice di analisi gira sulla rete reale e su quelle generate, senza sapere da
dove vengono i testi.

---

## Riprodurre

Serve [uv](https://docs.astral.sh/uv/) e Python ≥ 3.11. L'addestramento gira su CPU.

```bash
uv sync      # ambiente

# 1. le due partizioni del sintetico (curva di arrivo e attività misurate sul reale)
uv run python -m src.pipeline.build_synthetic_snapshots --activity from-real

# 2. le due reti (testo autentico; le varianti permuted/off sono controlli)
uv run python -m src.pipeline.generate_arrival \
    --partition synthetic_progressive_posts --output synthetic_progressive
uv run python -m src.pipeline.generate_arrival \
    --partition synthetic_all_present_posts --output synthetic_all_present

# 3. le analisi, per ogni cartella che interessa
uv run python -m src.analysis.analyze_structure  synthetic_progressive
uv run python -m src.analysis.analyze_temporal   synthetic_progressive
uv run python -m src.analysis.analyze_content    synthetic_progressive
uv run python -m src.analysis.analyze_homophily  synthetic_progressive
uv run python -m src.analysis.analyze_profiles   synthetic_progressive

# 4. tabelle e figure
uv run python -m src.analysis.scorecard
uv run python -m src.analysis.tail_scaling
uv run python -m src.figures.compare_folders
uv run python -m src.figures.plot_partitions
uv run python -m src.figures.plot_homophily_controls
```

Tutti i comandi girano dalla radice del progetto. Le figure dei dati si rigenerano dal codice; le tabelle
in `results/tables/` sono l'output di questi script.

---

## Cose da sapere sui numeri

- **Non riproducibilità al bit.** Il generatore è deterministico (stessi eventi di arrivo, stessi archi per
  evento), ma i pesi del modello addestrato variano al terzo decimale fra processi: una ripetizione dà
  271.655 archi invece di 271.646, cioè 9 su 271 mila.
- **La scelta dell'encoder è un pareggio.** MiniLM, GTE-base e BGE-base stanno entro 0,002, sotto la
  variabilità fra processi (~0,005). Va letta come parità, non come vittoria.
- **La coda dei gradi è più ripida del reale** a parità di dimensione: i poli generati sono meno estremi
  (274 contro 1.290). Il documento la misura con la legge di scala del reale e la dichiara fra i limiti.
- **Il primo passo della partizione progressiva è più vuoto del reale** (15,7% di nodi collegati contro
  41,8%). Quel 41,8% è uno stock preesistente, non un passo di nascita: non è un errore da correggere.
- **Distanza dal reale.** Definità una volta in `src/analysis/scorecard.py` come
  `|valore − reale| / |reale|`, sommata sulle sette metriche confrontabili (densità, grado medio, grado
  massimo, reciprocità, modularità, transitività, clustering). Archi e nodi sono esclusi perché dipendono
  dalla scala, l'omofilia perché tutte le reti generate stanno a 1,00.

---

## Documento

`document.pdf` è il documento finale, in inglese, su schema CRISP-DM: business understanding, dati,
preparazione, modellazione, valutazione, conclusioni. Nove tabelle, cinque figure, e i limiti dichiarati:
le cifre vengono da una singola esecuzione e non portano intervalli di confidenza.
