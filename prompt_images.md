# Prompt per le immagini — da dare a Gemini (o a un altro generatore)

Servono quattro figure che non sono grafici di dati: uno schema di pipeline, due diagrammi concettuali e
una mappa. Le figure dei dati (`compare_*`, `structure_*`, `profiles_*`) le genera già il codice in
`results/figures/` e non vanno rifatte a mano.

Regole comuni a tutti i prompt: **testo in inglese**, font senza grazie leggibile, sfondo bianco,
tavolozza blu (`#1f4e79`) + un solo accento caldo (`#c0392b`), niente ombre, niente 3D, niente icone
decorative, frecce sottili. Formato 16:9 o 4:3 secondo l'indicazione, 300 dpi. Se il generatore sbaglia
le parole, chiedere la stessa immagine "con etichette corte" e riscriverle poi a mano.

---

## 1. `pipeline_schema.png` — schema della pipeline (16:9)

> A clean flat vector diagram of a machine-learning pipeline, left to right, white background, blue
> palette with one red accent. Five rounded rectangles connected by thin grey arrows:
> (1) "Real network" with subtitle "9 snapshots · 25,714 users · 824,045 edges";
> (2) "Text representation" with subtitle "MiniLM, per post, mean per user";
> (3) "tNodeEmbed" with subtitle "Procrustes chain · LSTM · pair head";
> (4) "Arrival-event generator" with subtitle "k edges per newcomer · 7-variable choice";
> (5) "Synthetic network" with subtitle "9 steps · 271,646 edges", highlighted in the red accent.
> Below boxes 1 to 4, a second row of three boxes: "Synthetic partition" -> "9 steps" -> "posts only",
> connected upwards to box 4. Under box 5, a lighter box "Evaluation: network statistics vs the real
> network". All arrows point right except the ones from the partition row pointing up. Minimal, no
> shadows, thin 1.5 px strokes, sans-serif labels, 40 pt title "Pipeline".
> Save as `pipeline_schema.png`.

## 2. `partitions_timeline.png` — le due partizioni (16:9)

> A flat vector comparison of two timelines, one above the other, white background, blue palette.
> Top timeline "All present from step 1": nine equal segments; users are all present from the first
> segment, so the population line is flat at 100% from the start.
> Bottom timeline "Progressive arrival": nine segments with a population curve that rises 42%, 56%,
> 67%, 74%, 82%, 87%, 88%, 96%, 100%.
> Right side: two small captions, "official" next to the bottom one in the red accent, "control" next
> to the top one in grey. A third thin lane under both, "activity shape from the real data", shown as
> irregular bars of varying height per step. Sans-serif labels, no legend box, no shadows.
> Save as `partitions_timeline.png`.

## 3. `families_evolution.png` — dal conteggio imposto al meccanismo appreso (4:3)

> A flat vector diagram, left to right, showing four stages as columns of stacked cards, white
> background, blue palette, red accent on the last column.
> Column 1 "Hand-set counts": cards "threshold", "degree sampled from the real", "forced reciprocity",
> "growth constraint"; caption "the number is ours".
> Column 2 "Calibrated statistics": cards "bisection on mean degree", "growth curve"; caption
> "the number is imposed".
> Column 3 "Threshold on model output": cards "probability cutoff", "global rate"; caption "the number
> is copied".
> Column 4 "Learned and sampled": cards "pairwise model", "arrival events"; caption "the number
> emerges", in the red accent.
> Thin arrows between columns, each labelled with the property that broke (e.g. "no hubs", "no
> reciprocity", "too sparse"). Sans-serif, no shadows, no icons.
> Save as `families_evolution.png`.

## 4. `crispdm_map.png` — mappa delle sezioni (4:3)

> A flat vector diagram mapping the six phases of CRISP-DM to the sections of this report, white
> background, blue palette. Six rounded boxes arranged clockwise in a circle: "1 Business
> understanding", "2 Data understanding", "3 Data preparation", "4 Modeling", "5 Evaluation",
> "6 Conclusions", connected by thin clockwise arrows. Under each box, one short line in smaller grey
> text: "goals, constraints", "two datasets, time axis", "partitions, encoders", "tNodeEmbed, generator",
> "structure, content, profiles", "what holds, what does not". Title "Report structure".
> Sans-serif, thin strokes, no shadows.
> Save as `crispdm_map.png`.

---

## Dove vanno

| file | dove |
|---|---|
| `pipeline_schema.png` | `docs/relazione.md` §4 (Modeling), `paper.md` §4 — figura 1 |
| `partitions_timeline.png` | `docs/relazione.md` §3 (Data preparation), `paper.md` §3 |
| `families_evolution.png` | `docs/relazione.md` §6 (Evaluation), `paper.md` §5 |
| `crispdm_map.png` | `docs/relazione.md` §1, facoltativa nel paper |

Le immagini vanno salvate in `results/figures/` insieme alle altre, così i documenti le citano con lo
stesso percorso.
