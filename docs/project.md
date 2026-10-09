# project.md — il progetto in due minuti

## Cos'è

Il progetto di esame studia **una rete sociale che evolve nel tempo**. Ci sono due dataset:

- **reale**: 25.714 utenti, 173.475 post, **824.045 collegamenti** (diretti) su **9 snapshot** (2022 → luglio 2025), con lo split dichiarato 6 training / 1 validation / 2 test;
- **sintetico**: 10.000 utenti con 100.456 post e attributi di profilo, **senza collegamenti e senza tempo**, in 4 batch.

L'unica informazione **comune** ai due dataset è il **contenuto dei post**: il reale ha solo campi del post (lingua, conteggi di reazioni/risposte, timestamp) e nessun profilo utente, il sintetico ha **attributi dell'utente** (genere, religione, orientamento politico, età, professione…) ma l'unica cosa che esiste da entrambe le parti è il testo — e quindi l'unica feature utilizzabile, visto che il modello si addestra sul reale e si applica al sintetico. Gli archi sono **interazioni fra due utenti** (leggibili come un follow, senza significato specifico da preservare). Il dataset reale **non va mai modificato**.

## Cosa dobbiamo fare

1. **Addestrare** il modello temporale **tNodeEmbed sul reale** — senza node2vec, partendo dalle rappresentazioni del contenuto — e valutarlo con lo split dichiarato.
2. **Ripartire il sintetico in 9 time step**, prendendo dal reale la curva di arrivo e la forma di attività (quando e quanto si scrive).
3. **Generare la rete sintetico ↔ sintetico** con il modello addestrato sul reale: nessun aggancio al reale.
4. **Studiare come la rete evolve**, confrontandola con il reale: struttura, evoluzione, contenuto, meccanismo di formazione degli archi.

## Come lo facciamo

| passo | script |
|---|---|
| rappresentare il contenuto (encoder scelto coi numeri, codifica **per post**) | `src/content_encoders.py`, `src/tnodeembed_real.py` |
| ripartire il sintetico in 9 time step (arrivo e attività **dal reale**) | `src/build_synthetic_snapshots.py` |
| decidere quali coppie sono candidate (misura sul reale) | `src/measure_source_activity.py` |
| generare la rete: **modello fittato sul reale + campionamento + retroazione** | `src/generate_structural.py` |
| varianti di confronto (regole a mano, calibrate, inferenza con soglia) | `src/apply_edge_rules.py`, `src/apply_v6.py`, `src/generate_inference.py` |
| analizzare ogni rete (struttura, evoluzione, contenuto, accrescimento) | `src/analyze_*.py`, `src/check_real_accretion.py` |
| tabella e figure di confronto con il reale | `src/variant_summary.py`, `src/compare_folders.py` |

La **rete ufficiale è V8**: un modello di formazione degli archi fittato sul reale (contenuto di tNodeEmbed + grado relativo + arrivo + arco inverso + vicini in comune + attività + recenza) e poi **simulato passo per passo, campionando** ogni coppia con la sua probabilità. Nessuna soglia e nessun conteggio deciso da noi: volume e gradi **emergono**. Documenti: **`FINAL.md`** (piano), **`docs/pipeline.md`** (tecnico), **`docs/relazione.md`** + PDF (divulgativo), **`docs/nota_dottorando.md`** (sintesi per il dottorando).

## Risultato principale

**Bene**: V8 è la rete più vicina al reale prodotta dal progetto — densità **0,00129** contro **0,00125** del reale (praticamente identica) e grado uscente medio **12,9–16,8** contro 32,05, **tutto emerso dal modello**, senza imporre statistiche (la versione precedente per soglia si fermava a 0,0004 e a 3,6–5,8).

**Male, e dichiarato**: restano lontane la **coda dei gradi** (massimo 102–128 contro 1.290), la **reciprocità** (0,004 contro 0,418), la **transitivity** (0,003 contro 0,034) e l'**omofilia del contenuto** (1,00 contro 2,13).

**Il contenuto non aggiunge nulla, misurato**: togliendo il punteggio di contenuto dal modello, l'AUC passa da **0,9818 a 0,9816** e la rete cambia del **4%**, con omofilia identica. Il meccanismo che il modello impara è "chi entra e quanto è connesso", non "chi si somiglia".

## Annotazione sulle lingue

- **Ogni file** creato, generato o modificato (script, CSV, figure, configurazioni) è **in inglese**.
- **Solo la conversazione** con me è **in italiano**.
- I file **`.md`** sono in **italiano** (sono documenti per me).
