# Nota per il dottorando — cosa abbiamo fatto e cosa ha dato

*Sintesi tecnica in italiano. Tutti i numeri vengono da file in `results/`; il piano completo è in `FINAL.md`, i dettagli tecnici in `docs/pipeline.md`.*

## 1. L'impianto, come indicato

- **Si addestra sul dataset reale e si fa inferenza sul sintetico**; i collegamenti generati sono **sintetico ↔ sintetico** (nessun aggancio al reale).
- **node2vec è stato rimosso** da tutta la pipeline: le rappresentazioni iniziali dei nodi sono i **vettori del contenuto** dei post (il paper lo consente, §3.1). Il problema dei nodi mai visti è risolto: la **restrizione del protocollo originale produce 0 archi** proprio perché nel reale ogni arco nuovo coinvolge un entrante — è il sintomo che lei aveva indicato, ed è la ragione per cui la valutazione usa l'intera transizione.
- **Un passaggio = uno snapshot = un time step**: 9 nel reale (dati), 9 nel sintetico (ripartito da noi).
- L'**unica colonna comune** ai due dataset (oltre a `id` e `account_id`) è il `content` dei post: il reale ha solo campi **del post** (lingua, conteggi di reazioni/risposte, timestamp) e nessun attributo di profilo; il sintetico ha **attributi dell'utente** (genere, religione, orientamento politico, età, professione…) più il testo. Poiché il modello si addestra sul reale e si applica al sintetico, come feature può entrare solo ciò che esiste da entrambe le parti: il contenuto. Gli attributi del sintetico servono **solo per la valutazione**, mai come ingresso. Il dataset reale non viene mai modificato.

## 2. Cosa abbiamo costruito

1. **Contenuto**: encoder scelto con i numeri (`all-MiniLM-L6-v2`), con **ogni post codificato singolarmente e i vettori mediati per utente**. Questa codifica è un miglioramento misurato: la concatenazione precedente veniva tagliata al limite di token dell'encoder (per gli utenti più attivi sopravviveva il **28%** del testo), e con la codifica per post l'AUC di protocollo passa da **0,534 a 0,549** (MiniLM), da 0,530 a 0,540 (bge-base), da 0,505 a 0,524 (tf-idf).
2. **Modello temporale**: tNodeEmbed sul reale — Procrustes a catena, LSTM, testa `[h_u, h_v, h_u⊙h_v]` — addestrato sulle 6 transizioni di training, validation per l'early stopping, **test intoccato**.
3. **Rete definitiva (V8)**: un **modello di formazione degli archi fittato sulle transizioni reali** — variabili: punteggio di contenuto, **grado relativo** (percentile, per trasferire fra popolazioni di dimensioni diverse), "l'utente è nuovo", **arco inverso**, **vicini in comune**, attività nel periodo, recenza, passo — e poi **generazione per campionamento con retroazione**: a ogni passo le variabili si ricalcolano sullo stato della rete generata, ogni coppia candidata diventa arco con la sua probabilità. **Nessuna soglia, nessun conteggio deciso da noi.**
4. **Una decisione presa dai dati**: sul reale, **almeno un capo ha scritto nel periodo nel 100%** degli archi nuovi → è un fatto e definisce i candidati; la *sorgente* è attiva solo nel **68–81%** → resta una variabile appresa (`results/source_activity.csv`).
5. **Scenario del sintetico**: curva di arrivo **e** forma di attività (raffiche e silenzi, media 3,7 · mediana 2 · p90 9) prese dal reale. Nota: la **curva aggregata per passo** del reale non è utilizzabile, perché gli snapshot hanno **durata diversa** (un anno, tre mesi, 20 giorni, 8 giorni, 3 giorni) e quei conteggi misurano "attività × lunghezza della finestra".

## 3. Risultati

| rete | archi | densità | grado medio | grado max | reciprocità | modularità | transitivity | omofilia |
|---|---|---|---|---|---|---|---|---|
| **reale** | 824.045 | **0,00125** | **32,05** | **1.290** | **0,418** | 0,275 | 0,034 | **2,13** |
| **V8, tutti presenti** | 129.321 | **0,00129** | 12,9 | 102 | 0,003 | 0,152 | 0,003 | 1,00 |
| **V8, arrivo progressivo** | 159.804 | 0,00160 | 16,0 | 125 | 0,004 | 0,140 | 0,004 | 1,00 |
| V8 controllo (vecchio scenario) | 168.015 | 0,00168 | 16,8 | 128 | 0,004 | 0,131 | 0,003 | 1,00 |
| V8, ablazione **senza contenuto** (`growth_v2`) | 153.100 | 0,00153 | 15,3 | 126 | 0,004 | 0,142 | 0,004 | 1,00 |

**Controllo del contenuto A/B/C** (stesso modello fittato sul reale e stesso scenario, partizione `generated_snapshots_v2`; l'unica differenza è il canale testuale sul lato sintetico):

| modalità | archi | densità | reciprocità | grado max | modularità | transitivity | omofilia |
|---|---|---|---|---|---|---|---|
| A — testo autentico | 128.859 | 0,001289 | 0,0029 | 100 | 0,154 | 0,00269 | 1,0017 |
| B — testo permutato fra utenti | 123.310 | 0,001233 | 0,0030 | 111 | 0,158 | 0,00248 | 1,0000 |
| C — senza testo | 124.154 | 0,001242 | 0,0029 | 98 | 0,158 | 0,00259 | 0,9995 |

- **La densità coincide con quella reale** (0,00129 contro 0,00125) e il grado medio è a 2–2,5 volte dal reale: con la versione precedente, basata su una soglia, si arrivava a 3,6–5,8 di grado medio e a una densità 3-4 volte più bassa. **Nessuno di questi valori è imposto**: emergono dal campionamento.
- **V6 (calibrata)** centra volume e reciprocità **per costruzione** (0,420 esatta): è utile come confronto, ma è la differenza fra *imporre* e *imparare*.
- **V9, il passo successivo: l'unità diventa l'evento di arrivo.** Misurando il reale si vede che un utente nuovo non porta con sé un arco ma **un gruppo di archi** — in media 29–83 secondo il periodo, con punte di **1.421**, cioè lo stesso ordine del grado massimo reale — e che circa il 40% di quelli è ricambiato dentro lo stesso arrivo. V9 stima sul reale queste quantità e genera arrivo per arrivo. Il risultato è che anche le proprietà relazionali si avvicinano: **reciprocità 0,303** (V8: 0,003; reale 0,418), grado medio **28,2** (V8: 12,9; reale 32,05), grado massimo **324** (V8: 102; reale 1.290), transitività 0,0203 (V8: 0,003; reale 0,034), diametro 6 come il reale. Gli scarti che restano sono piccoli e tutti nella stessa direzione (V9 è un po' più "piatta"), e la causa è unica: la preferenza per i nodi più connessi resta sottostimata nel passaggio da 25.714 utenti reali a 10.000 sintetici. Anche su V9 il controllo a testo permutato conferma che il contenuto non sposta la rete.
- **Il testo conosce l'utente, la struttura conosce i suoi collegamenti.** Dal solo contenuto dei post si predice lo schieramento politico (AUC **0,827**), la professione (**0,867**) e la fascia d'età (**0,690**) — label che esistono solo nel sintetico, usate perciò solo in valutazione — e **aggiungere la rete non cambia nulla**. Viceversa, per i collegamenti la struttura pesa 20–40 volte il testo. È l'immagine speculare dello stesso fatto, e regge anche su V9.
- **Il cambio di scenario sposta poco**: stesso volume entro il **5%** e nessuna metrica strutturale diversa. (Me l'aspettavo più incisivo: è un esito, non una conferma.)

## 4. Risultati negativi, dichiarati

1. **Il contenuto non porta il legame.** Col controllo **A/B/C** (testo autentico / **testo permutato fra utenti** / senza testo, stesso modello e stesso scenario) le tre reti sono **indistinguibili** nelle proprietà relazionali: reciprocità 0,0029 / 0,0030 / 0,0029, transitivity 0,0025 / 0,0025 / 0,0026, grado massimo 100 / 111 / 98, modularità 0,154 / 0,158 / 0,158; il volume varia di pochi punti percentuali **senza ordine** (`permuted` sta più vicino a `off` che a `real`). L'omofilia resta a 1,0017 / 1,0000 / 0,9995 (il reale è 2,13). Coerente con: i pesi fittati (contenuto **+0,11–0,14** contro **+3,3** dell'utente nuovo), l'AUC con **negativi appaiati** sul reale (0,49–0,57, cioè ~caso), e il fatto che il testo **predice bene gli attributi** del sintetico (schieramento politico AUC 0,83–0,87) — ma quella non è l'informazione che spiega i collegamenti. **Nota di metodo**: valutare una rete generata non si fa con un AUC (nel sintetico non ci sono archi veri) ma confrontando distribuzioni e statistiche di rete; l'ablazione per AUC (**0,9818 → 0,9816**) è **satura** e da sola non proverebbe nulla.
2. **V8 non riproduce tre proprietà**: coda dei gradi (max 102–128 contro 1.290), **reciprocità** (0,004 contro 0,418) e **transitivity** (0,003 contro 0,034). Nel modello fittato il coefficiente dell'arco inverso è lievemente negativo e quello dei **vicini in comune è −4,4**: la ragione è che i negativi sono coppie estratte a caso e gli archi veri coinvolgono spesso entranti, che non hanno vicini in comune. È una proprietà della competizione scelta e spiega la transitivity bassa.
3. **Il fine-tuning dell'encoder non passa**: migliora la validation ma peggiora **entrambe** le finestre di test (0,519 contro 0,534 MiniLM; 0,523 contro 0,530 bge-base). La supervisione è troppo debole.
4. **Le feature dei nodi addestrabili** (una proiezione inizializzata dal contenuto e addestrata con la loss del compito) **peggiorano**: 0,512 contro 0,534. La `Q_t` completa del paper (una riga per utente e snapshot, ~70M parametri) non entra nei 6 GB della scheda disponibile: non l'abbiamo provata.
5. **L'omofilia del 2,13 è in gran parte struttura, non scelta.** Contro coppie a caso è **2,133**; contro non-archi appaiati per grado (classi larghe) **1,511**; per grado esatto ±10% **1,290**; per grado+volume di testo 1,367; per volume di testo 1,554. Gli estremi degli archi hanno grado medio **403,6** contro **61,7** e **1.297 caratteri** cumulativi contro una mediana di **323**; la somiglianza interna per decile di grado sale da 0,079 a 0,291. Resta un **residuo lessicale piccolo** (AUC ≈ 0,58) che vive nei profili **TF-IDF+SVD cumulativi**, non nel canale del generatore (**MiniLM per post**, dove il segnale è nullo): il 2,13 e l'1,00 del modello non misurano la stessa cosa. Script: `src/analyze_homophily.py` → `results/homophily_control.csv`.

## 5. In una riga

La pipeline riproduce **la struttura di base** (densità e gradi) **imparando** il meccanismo dal reale e senza imporre statistiche; con l'unità di V9 — l'evento di arrivo — riproduce nell'ordine giusto anche **reciprocità, triangoli e coda dei gradi** (0,303 · 0,0203 · 324 contro 0,418 · 0,034 · 1.290 del reale), restando un po' più "piatta" del reale, con un'unica causa: la preferenza per i nodi più connessi è sottostimata nel trasferimento da 25.714 utenti reali a 10.000 sintetici. Sul contenuto il progetto ha due risultati controllati: il canale testuale **non sposta la rete** (A/B/C indistinguibili) e l'omofilia **2,13** del reale è **in gran parte posizione nella rete**, non scelta di somiglianza.
