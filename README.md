# Dalla rete reale alla rete sintetica: tNodeEmbed sul testo dei post

Il progetto ricostruisce la **rete di collegamenti** di una popolazione di cui si osserva soltanto il
**testo dei post**. Il meccanismo di formazione dei legami si impara da una rete sociale reale
(25.714 utenti, 824.045 archi, 9 snapshot) e si applica a un dataset sintetico (10.000 utenti,
nessun arco, nessuna data) ripartito in 9 passi temporali.

Due reti consegnate, stessa pipeline, nominate secondo la partizione da cui nascono:

| rete | partizione del sintetico | ruolo |
|---|---|---|
| `synthetic_progressive` | ingresso progressivo, con la curva di arrivo del reale | **la rete presentata** |
| `synthetic_all_present` | tutti gli utenti presenti dal primo passo | **confronto** |

La scelta della progressiva è **concettuale**, non metrica: nel reale il 58% degli utenti finali entra
durante il periodo osservato, e una rete che nasce deve avere arrivi. Sulle metriche le due sono
equivalenti (distanza relativa dal reale 3,84 contro 3,90 su sette metriche); la progressiva paga su
grado medio (27,45 contro 28,31), grado massimo (274 contro 328) e quota di archi fra utenti
arrivati nello stesso passo (17,0% contro 12,4%, reale 8,1%).

## I tre compiti

1. **Link prediction sul reale** con tNodeEmbed: rappresentazioni dei nodi dal testo (MiniLM, media
   dei post), Procrustes a catena, LSTM, testa di coppia `[h_u, h_v, h_u⊙h_v]`. Serve a scegliere
   l'encoder (6 confrontati, negativi appaiati) e a misurare quanto il contenuto separi un arco da un
   non-arco. Esito: il contenuto da solo sta sul caso (AUC 0,51-0,57).
2. **Generazione**: il modello per **eventi di arrivo**. L'unità non è la coppia ma l'utente che
   entra, che si porta dietro k archi (`results/tables/arrival_events.csv`: media 35-111 per transizione,
   massimo 1.202). k viene da una tabella fittata sul reale e riscalata sulla popolazione; i partner si
   scelgono con una logistica a sette variabili (grado, percentile, novità, attività, direzione,
   co-occorrenza, punteggio di contenuto).
3. **Predizione degli attributi** sul sintetico: dai post dell'utente (TF-IDF + SVD, 128 dimensioni)
   si predice `political_leaning` (AUC 0,827), `profession` (0,867), `age_interval` (0,690). Il grafo
   non aggiunge nulla; gli embedding di tNodeEmbed vanno peggio (0,619 / 0,587 / 0,514). Il task esiste
   solo sul sintetico: il reale non ha attributi.

## Dove guardare

| documento | contenuto |
|---|---|
| **`document.pdf`** (sorgente `document.tex`) | **il documento finale, in inglese**, su schema CRISP-DM: business understanding, dati, preparazione, modellazione, valutazione, conclusioni |
| `docs/pipeline.md` | dettaglio tecnico: script, dati, scelte, limiti |
| `docs/relazione.md` (+ PDF) | la guida in italiano: cosa c'è nel documento, perché delle scelte, glossario delle metriche |
| `archive/storico.md` | tutti i tentativi precedenti, con meccanismo e metriche |
| `archive/docs/` | i documenti superati: articolo (it/en), relazione precedente, prompt delle figure |
| `docs/riferimenti/` | figure e note di consultazione |

## Come si rigenera tutto

```bash
# 1. le due partizioni del sintetico (attività e curva di arrivo dal reale)
uv run python -m src.pipeline.build_synthetic_snapshots --activity from-real

# 2. la rete progressiva e quella con tutti presenti (solo testo autentico; le modalità con testo
#    permutato o azzerato sono un controllo, si rigenerano con --content permuted / --content off)
uv run python -m src.pipeline.generate_arrival --partition synthetic_progressive_posts --output synthetic_progressive
uv run python -m src.pipeline.generate_arrival --partition synthetic_all_present_posts --output synthetic_all_present

# 3. le analisi di una rete, per ogni cartella che interessa
uv run python -m src.analysis.analyze_structure synthetic_progressive
uv run python -m src.analysis.analyze_temporal  synthetic_progressive
uv run python -m src.analysis.analyze_content   synthetic_progressive
uv run python -m src.analysis.analyze_homophily synthetic_progressive
uv run python -m src.analysis.analyze_profiles  synthetic_progressive
uv run python -m src.analysis.check_real_accretion synthetic_progressive

# 4. tabelle, figure e consegna
uv run python -m src.analysis.scorecard      # reale, progressiva, tutti presenti
uv run python -m src.analysis.tail_scaling           # legge di scala del reale
uv run python -m src.figures.compare_folders --prefix compare   # figure di confronto
uv run python -m src.figures.plot_pipeline                      # schema della pipeline
uv run python -m src.pipeline.consolidate_native synthetic_progressive
```

Il documento si compila con LaTeX (due passate, per l'indice), la guida in italiano con pandoc:

```bash
xelatex document.tex && xelatex document.tex
pandoc docs/relazione.md -o docs/relazione.pdf --pdf-engine=xelatex -V geometry:margin=2.2cm --resource-path=.
```

## Cosa si consegna

`dataset/` con **sei cartelle**: i due dataset originali
(`real_snapshots/`, `synthetic_snapshots/`), le due **partizioni** del sintetico
(`synthetic_progressive_posts/`, `synthetic_all_present_posts/`) e le due **reti** generate
(`synthetic_progressive/`, `synthetic_all_present/`), ognuna con i 9 file `.edg` (uno per passo), i 4
file nella forma dei batch nativi e gli embedding dei nodi. I numeri sono in `results/tables/*.csv`, le figure
in `results/figures/` e `docs/riferimenti/`.

Non si consegnano, e sono documentati nei CSV: le reti generate con testo permutato e con il contenuto
azzerato (si rigenerano con `--content permuted` e `--content off`), i tentativi precedenti (in
`archive/`, descrizione in `archive/storico.md`) e i documenti superati (`archive/docs/`).

Il riferimento stabile della consegna è il tag **`consegna-progetto`**.

## Tre cose da sapere sui numeri

- **Non riproducibilità al bit**: il generatore è deterministico (stessi eventi, stessi archi per
  evento), ma i pesi della rete addestrata variano al terzo decimale fra processi. Una ripetizione
  della configurazione progressiva dà **271.655** archi invece di 271.646 (9 archi, 0,003%).
- **Coefficienti**: un file per run in `results/tables/coefficients_<cartella>.csv`. Quelli di
  `coefficients_synthetic_progressive.csv` vengono da una ripetizione indipendente della
  stessa configurazione, perché la run che ha generato la rete presentata scriveva su un file unico
  sovrascritto (corretto in `src/pipeline/generate_arrival.py`).
- **Distanza dal reale**: definita una volta in `src/analysis/scorecard.py` come
  `|valore − reale| / |reale|`, sommata sulle sette metriche confrontabili (densità, grado medio,
  grado massimo, reciprocità, modularità, transitività, clustering). Archi e nodi sono esclusi perché
  dipendono dalla scala, l'omofilia perché tutte le reti generate stanno a 1,00.
