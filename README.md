# Dalla rete reale alla rete sintetica: tNodeEmbed sul testo dei post

Il progetto ricostruisce la **rete di collegamenti** di una popolazione di cui si osserva soltanto il
**testo dei post**. Il meccanismo di formazione dei legami si impara da una rete sociale reale
(25.714 utenti, 824.045 archi, 9 snapshot) e si applica a un dataset sintetico (10.000 utenti,
nessun arco, nessuna data) ripartito in 9 passi temporali.

Due reti consegnate, stessa pipeline:

| rete | partizione del sintetico | ruolo |
|---|---|---|
| `synthetic_network` | ingresso progressivo, con la curva di arrivo del reale | **ufficiale** |
| `synthetic_control_network` | tutti gli utenti presenti dal primo passo | **controllo** |

La scelta della progressiva è **concettuale**, non metrica: nel reale il 58% degli utenti finali entra
durante il periodo osservato, e una rete che nasce deve avere arrivi. Sulle metriche le due sono
equivalenti (distanza relativa dal reale 3,84 contro 3,89 su sette metriche); la progressiva paga su
grado medio (27,45 contro 28,30), grado massimo (274 contro 328) e quota di archi fra utenti
arrivati nello stesso passo (17,0% contro 12,4%, reale 8,1%).

## I tre compiti

1. **Link prediction sul reale** con tNodeEmbed: rappresentazioni dei nodi dal testo (MiniLM, media
   dei post), Procrustes a catena, LSTM, testa di coppia `[h_u, h_v, h_u⊙h_v]`. Serve a scegliere
   l'encoder (6 confrontati, negativi appaiati) e a misurare quanto il contenuto separi un arco da un
   non-arco. Esito: il contenuto da solo sta sul caso (AUC 0,51-0,57).
2. **Generazione**: V9, il modello per **eventi di arrivo**. L'unità non è la coppia ma l'utente che
   entra, che si porta dietro k archi (`results/arrival_events.csv`: media 35-111 per transizione,
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
| `FINAL.md` | sintesi di consegna: cosa c'è, cosa dicono i numeri, cosa resta aperto |
| `docs/pipeline.md` | dettaglio tecnico: script, dati, scelte, limiti |
| `docs/relazione.md` (+ PDF) | relazione divulgativa |
| `paper.md` / `paper_en.md` (+ PDF) | articolo in italiano e in inglese |
| `CMDC.md` | memoria di lavoro: stato, comandi, trappole |
| `archive/storico.md` | tutti i tentativi precedenti, con meccanismo e metriche |
| `docs/riferimenti/` | figure e note di consultazione |

## Come si rigenera tutto

```bash
# 1. le due partizioni del sintetico (attività e curva di arrivo dal reale)
uv run python src/build_synthetic_snapshots.py --activity from-real

# 2. la rete ufficiale e il suo controllo (solo testo autentico: le altre modalità sono un controllo,
#    si rigenerano con --variant network_permuted --content permuted, e --variant network_off --content off)
uv run python src/generate_arrival.py --partition synthetic         --variant network --content real
uv run python src/generate_arrival.py --partition synthetic_control --variant network --content real

# 3. le analisi di una rete, per ogni cartella che interessa
uv run python src/analyze_structure.py synthetic_network
uv run python src/analyze_temporal.py  synthetic_network
uv run python src/analyze_content.py   synthetic_network
uv run python src/analyze_homophily.py synthetic_network
uv run python src/analyze_profiles.py  synthetic_network
uv run python src/check_real_accretion.py synthetic_network

# 4. tabelle, figure e consegna
uv run python src/scorecard.py      # reale contro ufficiale e controllo
uv run python src/tail_scaling.py           # legge di scala del reale
uv run python src/compare_folders.py --prefix compare   # figure di confronto
uv run python src/plot_pipeline.py                      # schema della pipeline
uv run python src/consolidate_native.py synthetic_network
```

I PDF (`paper.pdf`, `paper_en.pdf`, `docs/relazione.pdf`) si rigenerano dai rispettivi markdown:

```bash
pandoc paper.md    -o paper.pdf    --pdf-engine=xelatex -V geometry:margin=2.2cm --resource-path=.
pandoc paper_en.md -o paper_en.pdf --pdf-engine=xelatex -V geometry:margin=2.2cm --resource-path=.
pandoc docs/relazione.md -o docs/relazione.pdf --pdf-engine=xelatex -V geometry:margin=2.2cm --resource-path=.
```

## Cosa si consegna

`Esame Angelantonio Fedele Murolo/` con **quattro cartelle**: i due dataset originali
(`real_snapshots/`, `synthetic_snapshots/`), la **partizione** ufficiale (`synthetic/`) e la **rete**
ufficiale (`synthetic_network/`), che contiene i 9 file `.edg` (uno per passo), i 4 file nella forma dei
batch nativi e gli embedding dei nodi. I numeri sono in `results/*.csv`, le figure in
`results/figures/` e `docs/riferimenti/`.

Non si consegnano, e sono documentati nei CSV o nelle tabelle: la **partizione e la rete di controllo**
(archiviate in `archive/dataset/`, numeri in `results/*_synthetic_control*.csv`) e le reti del
**controllo A/B/C** con testo permutato e contenuto azzerato (numeri nella tabella del controllo in
`FINAL.md` e nel paper; si rigenerano con `--variant network_permuted --content permuted` e
`--variant network_off --content off`).

Il riferimento stabile della consegna è il tag **`consegna-progetto`**. I tentativi precedenti sono in
`archive/` (descrizione in `archive/storico.md`, CSV e script originali). Anche la **partizione di
controllo** e la **rete di controllo** stanno in `archive/dataset/` (fuori dal versionamento, sono
centinaia di MB): i loro numeri restano in `results/*_synthetic_control*.csv` e si citano da lì; le reti
si rigenerano con i due comandi del punto 2.

## Tre cose da sapere sui numeri

- **Non riproducibilità al bit**: il generatore è deterministico (stessi eventi, stessi archi per
  evento), ma i pesi della rete addestrata variano al terzo decimale fra processi. Una ripetizione
  della configurazione ufficiale dà **271.655** archi invece di 271.646 (9 archi, 0,003%).
- **Coefficienti**: un file per run in `results/coefficients_<cartella>.csv`. Quelli di
  `coefficients_synthetic_network.csv` vengono da una ripetizione indipendente della
  stessa configurazione, perché la run che ha generato la rete consegnata scriveva su un file unico
  sovrascritto (corretto in `src/generate_arrival.py`).
- **Distanza dal reale**: definita una volta in `src/scorecard.py` come
  `|valore − reale| / |reale|`, sommata sulle sette metriche confrontabili (densità, grado medio,
  grado massimo, reciprocità, modularità, transitività, clustering). Archi e nodi sono esclusi perché
  dipendono dalla scala, l'omofilia perché tutte le reti generate stanno a 1,00.
