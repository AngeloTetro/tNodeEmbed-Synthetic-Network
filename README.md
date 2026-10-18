# Dalla rete reale alla rete sintetica: tNodeEmbed sul testo dei post

Il progetto studia come **ricostruire la rete di collegamenti** di una popolazione di cui si osserva
soltanto il **testo dei post**: si impara il meccanismo di formazione dei legami da una rete sociale
reale (25.714 utenti, 824.045 archi, 9 snapshot) e lo si applica per inferenza a un dataset sintetico
(10.000 utenti, nessun arco, nessuna data) ripartito in 9 passi temporali. La pipeline ufficiale è
**V9, il modello per eventi di arrivo**: l'unità non è la coppia di utenti ma l'utente che entra, che
si porta dietro un gruppo di archi. Numeri, limiti e risultati negativi: **`FINAL.md` §10**.

## Da dove cominciare

| documento | a cosa serve |
|---|---|
| `docs/project.md` | il progetto in due minuti |
| `FINAL.md` | il piano e l'esito (§4 i comandi, §10 i risultati) |
| `docs/relazione.md` + `docs/relazione.pdf` | la relazione divulgativa |
| `docs/pipeline.md` | il dettaglio tecnico: script, scelte, limiti |
| `docs/nota_dottorando.md` | la sintesi tecnica per il dottorando |
| `docs/messaggio.md` | il messaggio di aggiornamento da mandargli |
| `paper.md` / `paper_en.md` (+ PDF) | l'articolo in italiano e in inglese |
| `CMDC.md` | memoria di lavoro: stato, comandi, trappole |
| `docs/riferimenti/` | materiali di consultazione (note su tNodeEmbed, conversazione originale) |

## Come si rigenera tutto

```bash
# 1. le due partizioni del sintetico (attività e curva di arrivo dal reale)
uv run python src/build_synthetic_snapshots.py --activity from-real

# 2. la rete V9 nelle tre modalità del controllo A/B/C
uv run python src/generate_arrival.py --partition generated_snapshots_v2 --content real    --variant v9_real
uv run python src/generate_arrival.py --partition generated_snapshots_v2 --content permuted --variant v9_permuted
uv run python src/generate_arrival.py --partition generated_snapshots_v2 --content off     --variant v9_off

# 3. le analisi di una cartella, poi le tabelle di confronto
uv run python src/analyze_structure.py generated_snapshots_v2_v9_real
uv run python src/analyze_content.py   generated_snapshots_v2_v9_real
uv run python src/check_real_accretion.py generated_snapshots_v2_v9_real
uv run python src/analyze_profiles.py generated_snapshots_v2_v9_real
uv run python src/variant_scorecard.py      # confronto con il reale + pendenza di coda
uv run python src/tail_scaling.py           # legge di scala del reale e confronto a pari dimensione
```

I comandi completi, con le varianti di confronto, sono in `FINAL.md` §4.

## I risultati in una riga

V9 riproduce reciprocità (0,304 contro 0,418), grado medio (28,30 contro 32,05), grado massimo (324
contro 1.290, ma **97%** di quanto la legge di scala del reale prevede a 10.000 utenti), transitività
(0,0209 contro 0,0337) e diametro (6 contro 6). Il contenuto **non** decide chi si collega a chi:
il controllo A/B/C (testo autentico, permutato, assente) dà reti indistinguibili, e l'omofilia 2,13
del reale è in gran parte struttura. Restano aperti la **forma** della coda (a pari dimensione più
ripida: −3,33 contro −1,69) e la **coesione**.

## Cosa si consegna

Le reti in `Esame Angelantonio Fedele Murolo/`: `generated_snapshots_v2_v9_{real,permuted,off}/`
(la consegna), `generated_snapshots_growth_v2_v9/` (controllo di sensibilità), le due partizioni e la
riga di confronto V8 (`..._v2_v8_*`). I numeri sono in `results/*.csv`, le figure in `results/figures/`.
Il riferimento stabile della consegna è il tag **`consegna-progetto`**.
