# project.md — il progetto in due minuti

## Cos'è

Il progetto di esame studia **una rete sociale che evolve nel tempo**. Ci sono due dataset:

- **reale**: 25.714 utenti, 173.475 post, **824.045 collegamenti** (diretti) su **9 snapshot** (2022 → luglio 2025), con lo split dichiarato 6 training / 1 validation / 2 test;
- **sintetico**: 10.000 utenti con 100.456 post e attributi di profilo, **senza collegamenti e senza tempo**, in 4 batch.

L'unica informazione **comune** ai due dataset è il **contenuto dei post**. Gli archi sono **interazioni fra due utenti** (leggibili come un follow, ma senza significato specifico da preservare). Il dataset reale **non va mai modificato**.

## Cosa dobbiamo fare

1. **Addestrare** il modello temporale **tNodeEmbed sul reale** — senza node2vec, partendo dalle rappresentazioni del contenuto — e valutarlo con lo split dichiarato.
2. **Ripartire il sintetico in 9 time step** (il sintetico non ha tempo: glielo imponiamo noi), in due varianti: "tutti presenti" e "arrivo progressivo".
3. **Generare la rete sintetico ↔ sintetico** con il modello addestrato sul reale: nessun aggancio al reale.
4. **Studiare come la rete evolve**, confrontandola con il reale: struttura, evoluzione, contenuto, meccanismo di formazione degli archi.

## Come lo facciamo

| passo | script |
|---|---|
| rappresentare il contenuto (encoder scelto con i numeri) | `src/content_encoders.py`, `src/tnodeembed_real.py` |
| ripartire il sintetico in 9 time step | `src/build_synthetic_snapshots.py` |
| addestrare sul reale e fare inferenza sul sintetico | `src/transfer_to_synthetic.py` |
| generare le varianti: regole a mano (v1–v5), calibrate (v6), per inferenza (v7) | `src/apply_edge_rules.py`, `src/apply_v6.py`, `src/generate_inference.py` |
| analizzare ogni rete (struttura, evoluzione, contenuto, accrescimento) | `src/analyze_*.py`, `src/check_real_accretion.py` |
| tabella e figure di confronto con il reale | `src/variant_summary.py`, `src/compare_folders.py` |

Le **due reti ufficiali** sono affiancate: **v7** ("cosa predice il modello", con un modello di formazione degli archi fittato sul reale) e **v6** ("cosa serve per somigliare al reale", con le statistiche del reale imposte). Documenti: **`docs/pipeline.md`** (tecnico, definitivo) e **`docs/relazione.md`** (spiegato semplice, con il PDF).

**Risultato principale**: la pipeline riproduce *quando* nasce un collegamento (chi entra nella rete, quanto è connesso) ma **non** *fra chi* per via del contenuto — le reti generate hanno omofilia del contenuto ≈ 1,00 contro 2,13 del reale, e nel modello fittato il contenuto pesa **+0,18** contro **+2,28** dell'arrivo di un utente nuovo.

## Annotazione sulle lingue

- **Ogni file** creato, generato o modificato (script, CSV, figure, configurazioni) è **in inglese**.
- **Solo la conversazione** con me è **in italiano**.
- I file **`.md`** sono in **italiano** (sono documenti per me).
