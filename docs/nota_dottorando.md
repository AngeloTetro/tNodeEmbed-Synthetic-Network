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
| V8, ablazione **senza contenuto** | 153.100 | 0,00153 | 15,3 | 126 | 0,004 | 0,142 | 0,004 | 1,00 |

- **La densità coincide con quella reale** (0,00129 contro 0,00125) e il grado medio è a 2–2,5 volte dal reale: con la versione precedente, basata su una soglia, si arrivava a 3,6–5,8 di grado medio e a una densità 3-4 volte più bassa. **Nessuno di questi valori è imposto**: emergono dal campionamento.
- **V6 (calibrata)** centra volume e reciprocità **per costruzione** (0,420 esatta): è utile come confronto, ma è la differenza fra *imporre* e *imparare*.
- **Il cambio di scenario sposta poco**: stesso volume entro il **5%** e nessuna metrica strutturale diversa. (Me l'aspettavo più incisivo: è un esito, non una conferma.)

## 4. Risultati negativi, dichiarati

1. **Il contenuto non aggiunge capacità né cambia la rete.** Nell'ablazione (stesso modello, punteggio di contenuto azzerato): AUC di validation **0,9818 → 0,9816**, rete **+4%** in volume, **omofilia identica** (1,003). Coerente con: i pesi fittati (contenuto **+0,11** contro **+3,3** dell'utente nuovo), l'AUC con **negativi appaiati** (0,49–0,57, cioè ~caso), l'omofilia delle reti generate **1,00 contro 2,13** del reale, e il fatto che il testo **predice bene gli attributi** del sintetico (schieramento politico AUC 0,83–0,87) — ma quella non è l'informazione che spiega i collegamenti.
2. **V8 non riproduce tre proprietà**: coda dei gradi (max 102–128 contro 1.290), **reciprocità** (0,004 contro 0,418) e **transitivity** (0,003 contro 0,034). Nel modello fittato il coefficiente dell'arco inverso è lievemente negativo e quello dei **vicini in comune è −4,4**: la ragione è che i negativi sono coppie estratte a caso e gli archi veri coinvolgono spesso entranti, che non hanno vicini in comune. È una proprietà della competizione scelta e spiega la transitivity bassa.
3. **Il fine-tuning dell'encoder non passa**: migliora la validation ma peggiora **entrambe** le finestre di test (0,519 contro 0,534 MiniLM; 0,523 contro 0,530 bge-base). La supervisione è troppo debole.
4. **Le feature dei nodi addestrabili** (una proiezione inizializzata dal contenuto e addestrata con la loss del compito) **peggiorano**: 0,512 contro 0,534. La `Q_t` completa del paper (una riga per utente e snapshot, ~70M parametri) non entra nei 6 GB della scheda disponibile: non l'abbiamo provata.

## 5. In una riga

La pipeline riproduce **la struttura di base** (densità e gradi) **imparando** il meccanismo dal reale e senza imporre statistiche; **non** riproduce le proprietà che dipendono dalle relazioni (reciprocità, triangoli, hub) né quelle che dipendono dal testo (omofilia), e la ragione è misurata: il meccanismo che i dati rivelano è "chi entra e quanto è connesso", mentre il contenuto — così rappresentato — non aggiunge capacità di previsione.
