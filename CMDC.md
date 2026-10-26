# CMDC — memoria di lavoro

## Stato

Consegna finale: **V9 su partizione progressiva** (ufficiale) e **V9 su "tutti presenti"** (controllo),
ciascuna nelle tre modalità di contenuto. Il resto dei tentativi è in `archive/`. Ultimo tag:
`consegna-progetto`. Ramo di lavoro `levers/posts-as-edges`, `main` allineato.

## Struttura

```
src/                 pipeline e analisi (nomi in inglese)
results/             CSV delle reti conservate + scorecard + figure
docs/                pipeline.md, relazione.md, riferimenti/ (figure e note)
archive/             storico.md, results/ (CSV dismessi), src/ (script dismessi), docs/
Esame .../           dataset (gitignored): real_snapshots, synthetic_snapshots,
                     synthetic_control, synthetic,
                     synthetic_network, synthetic_control_network
                     (le reti permuted/off sono un controllo: CSV in results, si rigenerano)
```

## Comandi che contano

```bash
uv run python src/generate_arrival.py --partition synthetic --variant network --content real
uv run python src/analyze_structure.py synthetic_network
uv run python src/analyze_homophily.py synthetic_network
uv run python src/scorecard.py
uv run python src/consolidate_native.py synthetic_network
```

## Trappole (imparate a spese nostre)

- **Coefficienti**: fino alla pulizia `generate_arrival.py` scriveva sempre su
  `results/v9_coefficients.csv`, quindi ogni run sovrascriveva la precedente. Ora scrive
  `results/coefficients_<cartella>.csv`. Il file della rete ufficiale viene da una ripetizione
  indipendente: dirlo, non spacciarlo per la run consegnata.
- **`network_report.csv`**: era append-only e conteneva ~25 righe, comprese le run difettose precedenti
  alla correzione dei negativi. Ora ha sette righe con etichette esplicite (`ufficiale_*`,
  `controllo_*`). Un nuovo run aggiunge il proprio `variant`: non confonderlo con quelle righe.
- **Omofilia**: `analyze_homophily.py` scriveva su un nome fisso e l'ultima cartella analizzata
  sovrascriveva la decomposizione del reale. Ora scrive `homophily_control_<cartella>.csv`.
- **`snapshots_folder`** legge ogni sottocartella come snapshot: nella cartella di consegna i file
  nativi devono stare piatti (`native_4_batch_<i>.edg`), non in una sottocartella.
- **Dataset sotto OneDrive**: tenere i file "sempre disponibili", altrimenti le letture tornano vuote.
- **La cartella del dataset è in `.gitignore`**: cancellare una rete generata non si recupera da git.
  Gli script per rigenerarla stanno in `archive/src/`, i numeri in `archive/storico.md`.
- **Determinismo**: il generatore è deterministico, l'addestramento no (pesi al terzo decimale,
  ~10 archi su 271.646 fra processi). Non promettere numeri riproducibili al bit.
- **Distanza dal reale**: una sola definizione, in `scorecard.py`, `|v−r|/|r|` su sette
  metriche. Il vecchio "3,79 contro 4,39" non era riproducibile: non usarlo.
- **Log**: `results/_run.txt` viene sovrascritto a ogni run (contiene l'ultima, non quella
  consegnata).

## Convenzioni

- `.md` in italiano; script, CSV, figure e nomi file in inglese.
- `Esame Angelantonio Fedele Murolo/` e `tNodeEmbed/` sono read-only.
- Seed fissi; `uv` per l'ambiente; GPU RTX 3060 6 GB (fine-tuning di bge-base richiede `--batch 8`).
- Non aggiungere commenti al codice oltre a quelli già presenti; niente file temporanei lasciati in
  giro (script usa-e-getta nella scratchpad della sessione).
