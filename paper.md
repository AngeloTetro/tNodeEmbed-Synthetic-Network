# Dal testo dei post alla rete degli utenti: generazione di una rete sociale temporale per trasferimento di tNodeEmbed

**Angelantonio Fedele Murolo**

*Affiliazione: nessuna.*

---

## Abstract

Costruire la rete di collegamenti di una popolazione di cui si possiede soltanto il testo prodotto — senza alcun collegamento osservato e senza date — richiede di imparare il meccanismo di formazione dei legami da una rete diversa e di applicarlo per inferenza. Questo lavoro studia il problema su due dataset: una rete sociale reale con 25.714 utenti, 824.045 archi diretti e 173.475 post distribuiti su 9 snapshot (2022 – luglio 2025) e un dataset sintetico di 10.000 utenti con 100.456 post, privo di archi e di date, che ripartiamo in 9 passi temporali. L'unica informazione comune ai due dataset è il testo dei post. Il modello temporale è tNodeEmbed nella variante senza node2vec, inizializzata con rappresentazioni del contenuto: le rappresentazioni per snapshot, un allineamento di Procrustes a catena e una LSTM producono un embedding per utente, e una testa di coppia ne stima la probabilità di legame. Poiché la restrizione del protocollo di valutazione originale non ammette alcun arco di test sulla rete reale, la valutazione usa l'intera transizione con negativi appaiati e casuali. Dai coefficienti appresi nasce un generatore fittato sul reale, per campionamento con retroazione e senza conteggi imposti: un modello per **eventi di arrivo**, in cui l'unità è l'utente che entra. La rete consegnata si genera sulla partizione a **ingresso progressivo** — la più fedele al fatto che nel reale il 58% degli utenti entra durante il periodo osservato — e una seconda rete sulla partizione "tutti presenti" resta come controllo. La rete ufficiale riproduce la rete reale nell'ordine di grandezza giusto su reciprocità (0,3224 contro 0,4177), grado medio (27,45 contro 32,05), grado massimo (274 contro 1.290), transitività (0,0212 contro 0,0337), clustering (0,042 contro 0,080) e diametro (6 contro 6), restando un po' più "piatta" del reale. Il residuo che la misura attribuisce alla **forma** della coda e alla coesione, non all'hub: il controllo raggiunge il **97%** di ciò che la legge di scala stimata sul reale (esponente 1,445 sui suoi nove snapshot) prevede a 10.000 utenti, la rete ufficiale l'**81%**, e a pari dimensione la coda resta più ripida (−3,98 contro −1,69).

**Parole chiave:** reti temporali; node embedding; generazione di grafi; omofilia; trasferimento fra domini; tNodeEmbed.

---

## 1. Introduzione

Molte domande su un sistema sociale richiedono la sua rete di interazioni, ma i dati disponibili spesso contengono soltanto il testo che le persone producono: nessun collegamento osservato, nessuna data. Se un secondo dataset possiede la rete ma un testo diverso, si apre una possibilità precisa — **imparare il meccanismo di formazione dei legami sul dataset con la rete e applicarlo, per inferenza, al dataset senza rete** — a condizione di trovare un canale informativo presente in entrambi. Questo lavoro affronta esattamente questo problema: dato un corpus di post di 10.000 utenti senza alcun arco, costruire la loro rete sociale e studiarne la struttura e l'evoluzione nel tempo, usando come unica informazione condivisa con il dataset di riferimento il testo dei post.

Il problema ha tre difficoltà. La prima è **l'inizializzazione di nodi mai visti**: i metodi classici di embedding di rete, come node2vec, non sanno rappresentare un utente che non ha ancora collegamenti, mentre nella rete reale che studiamo nessun arco nuovo collega due utenti già presenti. La seconda è **il trasferimento fra popolazioni di dimensioni diverse**: il dataset che porta la struttura ha 25.714 utenti, quello da popolare 10.000, quindi le grandezze legate al grado vanno espresse in forma relativa. La terza è **la valutazione**: nel dataset senza archi non esistono collegamenti veri da indovinare, e un modello generativo di rete non si giudica con una misura di ranking.

Il contributo è quadruplo. (i) Adattiamo **tNodeEmbed** al nostro caso rimuovendo node2vec e usando le rappresentazioni del contenuto dei post come punto di partenza: è una possibilità prevista dall'impostazione del metodo, che ammette un qualunque embedding statico come inizializzazione. (ii) Deriviamo dai coefficienti appresi sul dataset reale un **generatore fittato** per campionamento con retroazione e senza conteggi imposti, in cui l'unità non è la coppia ma l'**evento di arrivo**, cioè l'ingresso di un utente, che porta con sé un gruppo di archi. (iii) Mettiamo a punto una **valutazione controllata** basata su statistiche e distribuzioni di rete, sul controllo del contenuto con testo autentico, testo permutato fra utenti e contenuto azzerato, e sulla scomposizione dell'omofilia con controlli di grado. (iv) Riportiamo come risultati i **risultati negativi**: il contenuto non sposta la rete, il fine-tuning dell'encoder e le feature dei nodi addestrabili peggiorano.

La sintesi dei risultati è la seguente. Il generatore per eventi di arrivo riproduce la rete reale nell'ordine di grandezza giusto: reciprocità 0,3224 (reale 0,4177), grado medio 27,45 (reale 32,05), grado massimo 274 (reale 1.290), transitività 0,0212 (reale 0,0337), clustering 0,042 (reale 0,080), diametro 6 come il reale. Gli scarti residui sono contenuti e tutti dello stesso segno; la loro causa non è una sottostima dell'hub — il grado massimo del controllo è al 97% di quanto la legge di scala prevede — ma la **forma** della coda e la **coesione**. Sul lato del contenuto i risultati sono netti e in parte contrari all'intuizione: il testo conosce l'utente (dal contenuto si predice lo schieramento politico con AUC 0,827, la professione con 0,867, la fascia d'età con 0,690) ma **non i suoi collegamenti**; l'omofilia del contenuto della rete reale, 2,13, è in gran parte posizione nella rete e non scelta di somiglianza.

---

## 2. Lavori correlati

**Embedding di nodi su grafi temporali.** tNodeEmbed (Singer, Guy e Radinsky, IJCAI 2019) è il riferimento di questo lavoro: costruisce una rappresentazione per ogni snapshot, allinea le rappresentazioni di snapshot consecutivi e ne apprende l'evoluzione con una rete ricorrente, con una loss di coppia per il compito di link prediction. L'impostazione generale dell'embedding di nodi è quella di node2vec (Grover e Leskovec, KDD 2016), che tNodeEmbed usa come inizializzazione statica; poiché node2vec richiede l'osservazione dei collegamenti, non è applicabile a utenti mai visti. La letteratura propone altre famiglie di modelli temporali: walk temporali su reti a tempo continuo (Nguyen e altri, WWW 2018), auto-attenzione su quartieri e tempo (Sankar e altri, WSDM 2020), reti convoluzionali i cui parametri evolvono nel tempo (Pareja e altri, AAAI 2020), architetture ricorrenti sulle transizioni (Goyal e altri, *Knowledge-Based Systems* 2020), e il processo di chiusura triadica come meccanismo esplicito (Zhou e altri, AAAI 2018). Il quadro è riassunto da Kazemi e altri (*JMLR* 2020). Rispetto a questi lavori, il nostro problema non è prevedere archi futuri in una rete nota, ma **popolare di archi una rete di cui non si osserva alcun collegamento**, trasferendo i parametri appresi altrove.

**Generazione di grafi.** I modelli classici — Barabási–Albert (1999) per l'attaccamento preferenziale, Watts–Strogatz (1998) per il piccolo mondo, i grafi di Kronecker (Leskovec e altri, *JMLR* 2010) per la struttura ricorsiva — generano topologie realistiche ma non sono fittati sul meccanismo di un dataset specifico né usano contenuto. I modelli generativi neurali recenti, GraphRNN (You e altri, ICML 2018) e NetGAN (Bojchevski e altri, ICML 2018), imparano a riprodurre distribuzioni di grafi e sono valutati confrontando statistiche e distribuzioni, che è lo stesso principio di valutazione che adottiamo; non sono però pensati per trasferire una struttura da una popolazione a un'altra priva di archi. Le leggi di evoluzione dei grafi reali (densificazione e riduzione dei diametri; Leskovec, Kleinberg e Faloutsos, KDD 2005) offrono i riferimenti empirici con cui leggiamo la crescita delle nostre reti.

**Rappresentazione del contenuto.** Le rappresentazioni del testo che usiamo appartengono alla famiglia degli encoder di frase addestrati per similarità (Sentence-BERT; Reimers e Gurevych, EMNLP 2019). La scelta fra di essi è guidata da un benchmark indipendente dal nostro compito (MTEB; Muennighoff e altri, EACL 2023) e dalla disponibilità di modelli compatti (BGE; Xiao e altri, SIGIR 2024).

**Proprietà delle reti e misure.** L'omofilia (McPherson, Smith-Lovin e Cook, 2001) e la chiusura triadica (Granovetter, 1973) sono i fenomeni che misuriamo; la struttura di comunità è identificata con l'algoritmo di Louvain (Blondel e altri, 2008). La cella ricorrente che aggrega la sequenza temporale è la LSTM (Hochreiter e Schmidhuber, 1997) e l'allineamento fra sistemi di coordinate è l'analisi di Procrustes generalizzata (Gower, 1975).

**Il margine che occupiamo.** Nessuno di questi lavori affronta la combinazione che ci interessa: imparare il meccanismo di formazione degli archi da una rete reale **e applicarlo per generazione a una popolazione che possiede soltanto testo**, mostrando con controlli espliciti quanto del risultato venga dal contenuto e quanto dalla struttura.

---

## 3. Dati

I due dataset sono stati **forniti per il progetto dal dottorando che lo ha seguito**. Il primo, che chiamiamo **reale**, contiene utenti, i loro post e i collegamenti fra loro, distribuiti su **9 snapshot** da gennaio 2022 a luglio 2025: 25.714 utenti, 824.045 archi **diretti**, 173.475 post cumulativi (154.580 conteggiati per periodo). Lo split dei periodi è dichiarato: 6 snapshot di training, 1 di validation, 2 di test. Per ogni utente il dataset reale contiene **solo campi del post** (il testo, la lingua, conteggi di reazioni e risposte, il timestamp): **non esistono attributi di profilo**. Il secondo dataset, che chiamiamo **sintetico**, contiene 10.000 utenti con **100.456 post** utilizzabili (su 100.474 righe, 18 scartate in lettura per un numero di campi diverso dall'intestazione), **nessun arco e nessuna data**, più 11 attributi di profilo dell'utente (fra cui schieramento politico, professione e fascia d'età) che nel dataset reale non esistono.

Poiché il modello si addestra sul dataset reale e si applica al sintetico, come feature può entrare soltanto ciò che esiste da entrambe le parti: l'**unica colonna comune**, oltre agli identificatori, è il **testo dei post** (`content`). Gli attributi del sintetico non vengono mai usati come ingresso e servono **solo in valutazione**. La rete reale non viene modificata in alcun modo: la rete sintetica è un artefatto nuovo, e i suoi collegamenti sono esclusivamente **da utente sintetico a utente sintetico**.

**Tabella 1 — I due dataset.**

| | reale | sintetico |
|---|---|---|
| utenti | 25.714 | 10.000 |
| post | 173.475 cumulativi / 154.580 per periodo | 100.456 utilizzabili |
| archi (diretti) | 824.045 | nessuno |
| tempo | 9 snapshot, 2022 → 2025-07; split 6/1/2 | nessuno: 4 batch in ordine casuale |
| informazione per utente | solo il testo dei post | 11 attributi di profilo + testo |
| colonna comune | **`content`** | **`content`** |

*Origine: `results/structure_real_snapshots.csv`, `results/native_batches.csv`, `results/encoder_comparison.csv`.*

**La convenzione del tempo nel sintetico.** Il sintetico non ha date; l'unico ordine disponibile è l'identificatore denso dei post, che assumiamo riflettere l'ordine di generazione. Lo ripartiamo in **9 passi**, pari al numero di snapshot reali, con due regole misurate sul reale. Primo, la **curva di arrivo**, cioè la quota di utenti che entra a ogni passo, ordinando gli utenti per il loro primo post: **0,418 · 0,142 · 0,110 · 0,072 · 0,078 · 0,051 · 0,010 · 0,077 · 0,042**. Secondo, la **forma dell'attività**, cioè quando un utente scrive: i post di ciascun utente sono distribuiti sui passi della sua vita campionando la distribuzione reale dei post per utente attivo per periodo (**media 3,7, mediana 2, p90 9**), con le raffiche collocate **a caso**, così che compaiono sia periodi pieni sia periodi silenziosi. **Non** usiamo la curva aggregata dei post per snapshot del reale, perché gli snapshot reali hanno durata disomogenea (da un anno a pochi giorni) e quei conteggi misurano "attività × lunghezza della finestra". Restano scelte dichiarate l'ordine di arrivo (dedotto dagli identificatori) e l'ordine interno a ciascun passo.

*Origine: curva di arrivo e distribuzione dell'attività misurate sul reale da `src/build_synthetic_snapshots.py`.*

**Le due partizioni.** La partizione a **ingresso progressivo** applica la curva di arrivo; quella "tutti presenti" fa esistere ogni utente dal primo passo e tiene solo la forma dell'attività. Nel reale il 41,8% degli utenti finali è già collegato al primo snapshot e il restante 58% entra durante il periodo osservato: la prima partizione è quindi la più fedele, e la seconda resta come **controllo**.

---

## 4. Metodo

![Schema del metodo: dal corpus reale e da quello sintetico alla rete generata](results/figures/paper_pipeline.png)

**Figura 1** — Schema della pipeline. Il modello temporale si addestra sul dataset reale; il dataset sintetico viene ripartito in 9 passi con la curva di arrivo e la forma di attività misurate sul reale; la rete viene generata passo per passo per campionamento con retroazione. File: `results/figures/paper_pipeline.png`.

### 4.1 Rappresentazione del contenuto

L'unico ingresso testuale è il testo dei post. Ogni post è codificato **singolarmente** da un encoder di frase e i vettori di un utente sono **mediati** in una rappresentazione per utente e per snapshot. Abbiamo scelto l'encoder con un protocollo di confronto invece che a priori: si addestra l'intera pipeline temporale sul reale con ciascun encoder e si confrontano le AUC sulle transizioni di validation e di test. La colonna della decisione è quella con i **negativi appaiati** (vedi §4.2).

**Tabella 2 — Scelta dell'encoder sul dataset reale** (AUC su validation / test1 / test2; media = val+test; testo del periodo).

| encoder | AUC negativi appaiati (val / test1 / test2) | media | AUC negativi casuali | archi che soddisfano la restrizione del protocollo originale |
|---|---|---|---|---|
| tf-idf + SVD | 0,489 / 0,521 / 0,505 | 0,505 | 0,986 / 0,931 / 0,941 | 0 / 0 / 0 |
| **all-MiniLM-L6-v2** | 0,571 / 0,521 / 0,510 | **0,534** | 0,995 / 0,954 / 0,961 | 0 / 0 / 0 |
| BGE-small | 0,541 / 0,517 / 0,519 | 0,526 | 0,996 / 0,958 / 0,967 | 0 / 0 / 0 |
| GTE-small | 0,529 / 0,514 / 0,507 | 0,516 | 0,996 / 0,956 / 0,963 | 0 / 0 / 0 |
| BGE-base | 0,540 / 0,517 / 0,535 | 0,530 | 0,996 / 0,960 / 0,968 | 0 / 0 / 0 |
| GTE-base | 0,543 / 0,523 / 0,531 | 0,532 | 0,996 / 0,959 / 0,969 | 0 / 0 / 0 |

*Origine: `results/encoder_comparison.csv`. L'ultima colonna è il numero di archi di test ammessi dalla restrizione del protocollo originale in ogni finestra.*

La codifica **per post** migliora tutti gli encoder rispetto alla concatenazione dei post di un utente in un'unica stringa, che l'encoder tagliava al proprio limite di token: la media val+test passa da 0,534 a **0,549** per MiniLM, da 0,530 a **0,540** per BGE-base, da 0,505 a **0,524** per tf-idf. La pipeline adotta **all-MiniLM-L6-v2** con codifica per post e vettori mediati; l'encoder è addestrato soltanto sul contenuto di training.

*Origine: `results/encoder_comparison.csv`, righe `*|mean_of_posts|frozen`.*

**Quanto è stretto il confronto.** Le differenze fra i primi tre encoder (0,534, 0,532, 0,530) sono **dentro** la variabilità fra processi: rilanciando lo stesso encoder si ottengono valori che oscillano di circa 0,005 fra un processo e l'altro, perché l'ordine di iterazione degli insiemi di archi non è fissato in Python e cambia il campionamento dei negativi. La scelta va quindi letta come una **parità fra tre encoder** (MiniLM, GTE-base, BGE-base) nettamente sopra tf-idf (0,505), non come una graduatoria. È un limite dichiarato, non un dettaglio: la metrica di selezione non ha la risoluzione per distinguere il primo dal terzo.

### 4.2 Modello temporale

Il modello temporale è **tNodeEmbed senza node2vec**. Per ogni snapshot `t` disponiamo di una matrice utenti × dimensione costruita dal contenuto (§4.1); un **allineamento di Procrustes ortogonale a catena** fra snapshot consecutivi porta le rappresentazioni in un sistema di coordinate comune, stimato sui nodi presenti in entrambi gli snapshot e applicato a tutti, nuovi inclusi; una **LSTM** legge la sequenza dei 9 passi di ciascun utente e la riassume in un embedding temporale; una **testa di coppia**, alimentata da `[h_u, h_v, h_u * h_v]` (embedding del primo utente, embedding del secondo e prodotto elemento per elemento), produce il punteggio di legame, ottimizzato con entropia incrociata binaria sulle transizioni di training. Il test non viene toccato fino alla valutazione.

La rimozione di node2vec non è un'omissione ma una scelta motivata dai dati: node2vec si inizializza osservando i collegamenti, quindi non può rappresentare un utente che compare per la prima volta, e su questo dataset **nessun arco nuovo collega due utenti già presenti**. Applicando la **restrizione del protocollo di valutazione originale** (test limitato a coppie fra utenti già osservati) si ottengono **0 archi di test in tutte le transizioni** (Tabella 2, ultima colonna): la restrizione, corretta in generale, è vuota qui. La valutazione usa quindi **l'intera transizione** (tutti gli archi nuovi del periodo), con due modalità di negativi: **casuali** e **appaiati**, dove ogni coppia vera è accoppiata a una coppia falsa con lo stesso primo utente e una destinazione con storia di arrivo e grado simili. La modalità appaiata è la colonna della decisione, perché impedisce al modello di "imbrogliare" riconoscendo l'utente nuovo dalla sola sequenza.

### 4.3 Generazione della rete

Dai punteggi del modello temporale non nasce automaticamente una rete: servono decisioni su quali coppie diventano archi. Le prendiamo **fittando il meccanismo sul dataset reale** e applicandolo per generazione al sintetico.

**Unità: l'evento di arrivo.** Il punto di partenza è una misura. Sul reale **nessun arco nuovo collega due utenti già presenti** (quota "già presente → già presente" 0,0000% in tutte le transizioni): ogni arco appartiene quindi all'evento del suo estremo nuovo. Un evento porta in media **35–111 archi** secondo la transizione (mediana 6–32), con un massimo di **1.202**, cioè lo stesso ordine del grado massimo reale: l'hub nasce da un *singolo* arrivo e non dalla somma di decisioni indipendenti. Il **51–63%** degli archi dell'evento è uscente e circa il **40%** delle coppie è ricambiato dentro l'evento.

**Il generatore.** Si stimano sul reale tre cose e si generano **arrivo per arrivo**: quanti archi porta un arrivo (`k`, campionato da una tabella fittata sul reale e riscalato sulla popolazione); la direzione di ciascun arco (quota uscente del passo); il bersaglio, con una regressione logistica su sette variabili — punteggio di contenuto della testa di coppia, `log(1+grado)`, percentile di grado, "la controparte è nuova", "la controparte ha scritto nel periodo", direzione, co-occorrenza con i partner già scelti nell'evento. Le coppie ricambiate si pescano fra gli uscenti già scelti, con una tabella reale per direzione e classe di grado. Lo stato della rete (gradi, percentili, novità, attività, vicinato) si ricalcola **a ogni passo**; nessuna soglia e nessun conteggio è imposto.

*Origine: le quantità dell'arrivo sono in `results/arrival_events.csv`; i coefficienti in `results/coefficients_synthetic_network.csv`; i conteggi finali in `results/network_report.csv`.*

**La partizione consegnata.** Tutte le cifre di questo articolo si riferiscono alla rete generata sulla partizione a **ingresso progressivo**. La stessa pipeline eseguita sulla partizione "tutti presenti" è il **controllo** e viene riportata accanto, perché la scelta fra le due non è metrica ma concettuale (§5.2).

### 4.4 Protocollo di valutazione e gerarchia delle metriche

Una **rete generata non si valuta con un AUC**: nel dataset sintetico non esistono archi veri da indovinare, quindi una misura di ranking non ha oggetto. Adottiamo una gerarchia esplicita.

1. **Primarie — statistiche e distribuzioni di rete per snapshot**, confrontate con il reale sullo snapshot finale: distribuzione dei gradi (media, massimo, CCDF), reciprocità, transitività e clustering medio, modularità (Louvain), diametro, composizione dell'accrescimento.
2. **Secondarie — il contenuto**, con la natura del legame: omofilia del contenuto con controlli di grado, e controllo A/B/C (testo autentico, testo permutato fra utenti, contenuto azzerato).
3. **Diagnostiche, solo sul reale** — AUC e average precision con negativi appaiati e casuali, usate per scegliere l'encoder e per fittare il modello. L'ablazione per AUC sul reale è **satura** (0,9818 → 0,9816) e da sola non misura il contributo del testo: la prova è il confronto fra reti (punto 2).
4. **Non valida** — l'AUC come misura di qualità di una rete generata.

---

## 5. Risultati

### 5.1 Il confronto centrale

**Tabella 3 — Ultimo snapshot: reale, rete ufficiale e controllo.**

| rete | archi | nodi | densità | grado medio | grado max | reciprocità | modularità | transitività | clustering | diametro | omofilia |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **reale** | 824.045 | 25.714 | 0,00125 | 32,05 | 1.290 | 0,4177 | 0,275 | 0,0337 | 0,080 | 6 | 2,13 |
| **ufficiale** (progressiva) | 271.646 | 9.897 | 0,00277 | 27,45 | 274 | 0,3224 | 0,1074 | 0,0212 | 0,0420 | 6 | 1,0009 |
| controllo (tutti presenti) | 280.298 | 9.901 | 0,00286 | 28,31 | 328 | 0,3035 | 0,1030 | 0,0209 | 0,0435 | 6 | 0,9993 |

*Origine: `results/structure_*.csv`, `results/structure_heavy_*.csv`, `results/content_*.csv`; distanze in `results/scorecard.csv`.*

La rete ufficiale porta **nell'ordine giusto** reciprocità (0,3224 contro 0,4177), scala dei gradi (27,45 contro 32,05), transitività (0,0212 contro 0,0337), clustering (0,042 contro 0,080) e diametro (6 contro 6). Il controllo fa leggermente meglio su grado medio, grado massimo e clustering, e leggermente peggio su densità, reciprocità, modularità e transitività: sulle sette metriche confrontabili vince 4 a 3, e la distanza relativa dal reale, definita come `|valore − reale| / |reale|` e sommata sulle sette, è **3,84** per l'ufficiale e **3,89** per il controllo. Cambiando normalizzazione l'ordine si inverte: le due partizioni sono **equivalenti** su questo piano.

![Confronto fra reale, ufficiale e controllo: struttura](results/figures/compare_structure.png)

**Figura 2** — Struttura per snapshot: reale, rete ufficiale e controllo (densità, reciprocità, modularità). File: `results/figures/compare_structure.png`.

![Confronto fra reale, ufficiale e controllo: crescita](results/figures/compare_growth.png)

**Figura 3** — Nodi e archi per snapshot. File: `results/figures/compare_growth.png`.

![Confronto fra reale, ufficiale e controllo: coda dei gradi](results/figures/compare_degree_ccdf.png)

**Figura 4** — Distribuzione del grado uscente (CCDF), snapshot finale. File: `results/figures/compare_degree_ccdf.png`.

### 5.2 Perché la partizione progressiva, e cosa costa

Nel reale il **41,8%** degli utenti finali è già collegato al primo snapshot e il restante **58%** entra durante il periodo osservato. Una rete che nasce deve avere arrivi: la partizione progressiva riproduce la curva di arrivo per costruzione, quella "tutti presenti" no. Il costo è dichiarato e misurato.

**Tabella 4 — Le due partizioni, ultimo snapshot.**

| metrica | reale | ufficiale | controllo | più vicino |
|---|---:|---:|---:|---|
| densità | 0,001246 | 0,002774 | 0,002860 | ufficiale |
| grado medio | 32,05 | 27,45 | 28,31 | controllo |
| grado massimo | 1.290 | 274 | 328 | controllo |
| reciprocità | 0,4177 | 0,3224 | 0,3035 | ufficiale |
| modularità | 0,2751 | 0,1074 | 0,1030 | ufficiale |
| transitività | 0,0337 | 0,0212 | 0,0209 | ufficiale |
| clustering | 0,0804 | 0,0420 | 0,0435 | controllo |
| diametro | 6 | 6 | 6 | pari |
| pendenza di coda (ultimo decile) | −2,349 | −3,980 | −3,332 | controllo |
| archi fra utenti arrivati nello stesso passo | 8,1% | 17,0% | 12,4% | controllo |

*Origine: `results/scorecard.csv`, `results/tail_scaling_summary.csv`, `results/accretion_*.csv`.*

Il costo più visibile è su due righe: gli archi fra co-arrivati salgono dal 12,4% al **17,0%** (reale 8,1%), perché con l'ingresso progressivo ci sono più entranti disponibili a ogni passo e il peso positivo su "la controparte è nuova" spende più archi fra coetanei; e il grado massimo scende a **274**, l'**81%** di quanto la legge di scala del reale prevede a 10.000 utenti, contro il 97% del controllo. Il primo passo è inoltre più vuoto del reale (15,7% dei nodi collegati contro 41,8%), perché `k` si riscala sulla popolazione presente e al primo passo è solo quella entrata.

Quest'ultimo confronto va però letto con cautela: il 41,8% del reale **non è un passo di nascita**, è uno stock preesistente (il primo snapshot contiene 10.755 utenti e 116.441 archi creati prima dell'osservazione). Confrontarlo con la prima generazione di una rete che nasce è un confronto fra oggetti diversi, e non è un criterio.

**Un'estrapolazione da dichiarare.** La legge che riscala `k` è fittata su popolazioni fra 10.755 e 21.082 utenti; al primo passo della partizione progressiva vale `P ~ 4.100`, quindi è un'estrapolazione sotto il range di fit. Non abbiamo dati reali sotto i 10.000 utenti per convalidarla.

![Confronto fra le partizioni: omofilia](results/figures/compare_homophily.png)

**Figura 5** — Omofilia del contenuto per snapshot: reale, ufficiale e controllo. File: `results/figures/compare_homophily.png`.

### 5.3 Il controllo sul contenuto

Il contenuto è l'unica informazione che distingue fra loro i 10.000 utenti del sintetico, quindi il suo ruolo va misurato, non postulato. Con lo **stesso modello fittato sul reale e lo stesso scenario**, generiamo tre reti che differiscono per il solo canale testuale sul lato sintetico: **A** testo autentico, **B** testo permutato fra utenti (gli stessi profili di testo riassegnati ad altre persone, così le statistiche del testo restano identiche), **C** contenuto azzerato nel modello di scelta.

**Tabella 5 — Controllo A/B/C sulla rete ufficiale (partizione progressiva).**

| modalità | archi | densità | grado medio | grado max | reciprocità | modularità | transitività | clustering | omofilia |
|---|---|---|---|---|---|---|---|---|---|
| A — testo autentico | 271.646 | 0,002774 | 27,45 | 274 | 0,3224 | 0,1074 | 0,0212 | 0,0420 | 1,0009 |
| B — testo permutato | 261.680 | 0,002668 | 26,42 | 248 | 0,3229 | 0,1102 | 0,0207 | 0,0392 | 0,9995 |
| C — contenuto azzerato | 271.640 | 0,002777 | 27,47 | 275 | 0,3227 | 0,1066 | 0,0213 | 0,0421 | 1,0010 |

*Origine: `results/structure_synthetic_network{,_permuted,_off}.csv` e i rispettivi `structure_heavy_*` e `content_*`. Le due reti di controllo non fanno parte della consegna: i loro numeri sono in quei CSV e si rigenerano con `--content permuted` e `--content off`.*

Le tre reti sono **indistinguibili** nelle proprietà relazionali: il volume varia del 3,7% e **senza ordine** (la rete col testo permutato sta più vicino a quella azzerata che a quella col testo autentico), e reciprocità, transitività, clustering, modularità e diametro coincidono entro la rumorosità fra addestramenti. L'unico effetto visibile del testo è minimo: l'omofilia passa da 1,0010 (contenuto azzerato) a 1,0009 (testo autentico), contro il 2,13 del reale. **Il contenuto non sposta la rete.**

Questo risultato è coerente con i pesi appresi, su variabili standardizzate: il punteggio di contenuto pesa **+0,05** contro **+1,22** della novità della controparte e **+1,02 / +0,91** di grado e percentile; "ha scritto nel periodo" pesa **−0,03** e la co-occorrenza **+0,19**. Il modello discrimina bene sulla transizione di validation (AUC 0,918), ma lo fa **con la struttura**: *chi entra* e *quanto è connesso*, non *cosa scrive*.

*Origine: `results/coefficients_synthetic_network*.csv`, `results/network_report.csv`.*

### 5.4 Scomposizione dell'omofilia

Il numero che sembrava contraddire §5.3 è l'omofilia del contenuto della rete reale: **2,13** sullo snapshot finale, cioè i due capi di un arco si somigliano più del doppio di due utenti estratti a caso. La misura standard confronta però ogni arco con due utenti **qualsiasi**, mentre gli estremi degli archi non sono utenti qualsiasi: nel reale l'estremo già presente ha grado medio **403,6** contro **61,7**, e la somiglianza lessicale fra due utenti **cresce con il grado** (0,079 nel decile di grado più basso, 0,291 nel più alto). Se il confronto si fa con **non-archi della stessa posizione**, l'omofilia cambia volto.

**Tabella 6 — Omofilia del contenuto del reale contro controlli strutturali (ultimo snapshot).**

| confronto | rapporto | AUC archi vs controllo |
|---|---|---|
| coppie **a caso** (misura standard) | 2,133 | 0,702 |
| non-archi con lo **stesso grado** (classi larghe) | 1,511 | 0,610 |
| non-archi con grado esatto **±10%** | 1,290 | 0,575 |
| non-archi con lo stesso **volume di testo** | 1,554 | 0,648 |
| non-archi con grado e **volume di testo** simili | 1,367 | 0,585 |
| — somiglianza interna per decile di grado: 1° decile 0,079 → 10° decile 0,291 | | |

*Origine: `results/homophily_control_real_snapshots.csv`. I controlli più stretti (grado esatto ±10%) instabilizzano il campione (737.101 coppie), quindi 1,29 va letto come limite inferiore.*

Più della metà del divario sparisce appena si confrontano utenti comparabili: **il 2,13 è in gran parte posizione nella rete**, non scelta di somiglianza. Sulla rete ufficiale lo stesso protocollo dà **1,001 / 1,000 / 0,980 / 1,001 / 1,014**: nessuna omofilia di contenuto. Resta un **residuo lessicale piccolo** nel reale (rapporto 1,29; AUC 0,58), che vive nei profili lessicali usati per questa misura e non nel canale del generatore (per-post, dove il segnale è nullo): il 2,13 del reale e l'1,00 delle reti generate **non misurano la stessa cosa**.

*Origine: `results/homophily_control_synthetic_network.csv`.*

### 5.5 I profili

Nel dataset sintetico gli attributi dell'utente sono etichette note, quindi possiamo chiedere se il testo sappia *chi è* un utente. Addestriamo una regressione logistica sull'80% degli utenti (split stratificato per utente, seed fisso) e misuriamo l'AUC macro una-contro-tutte sul 20% restante, confrontando quattro rappresentazioni.

**Tabella 7 — Predizione degli attributi del profilo (AUC macro, ultimo snapshot, rete ufficiale).**

| etichetta (classi) | solo contenuto | contenuto + rete | embedding temporale | embedding + rete |
|---|---|---|---|---|
| schieramento politico (7) | **0,827** | 0,826 | 0,619 | 0,626 |
| professione (41) | **0,867** | 0,865 | 0,587 | 0,577 |
| fascia d'età (8) | **0,690** | 0,683 | 0,514 | 0,505 |

*Origine: `results/profiles_synthetic_network.csv`.*

Il **contenuto conosce l'utente**, e aggiungere la rete non cambia nulla; gli embedding temporali, addestrati a prevedere i *legami*, perdono il profilo. È **l'immagine speculare di §5.3**: *chi è* un utente sta nel testo, *con chi si collega* sta nella struttura. Il legame, infatti, dipende da variabili di stato — l'ingresso dell'utente e il suo grado — che il testo non contiene. Il task esiste solo sul sintetico: il reale non ha attributi.

![Classificazione dei profili](results/figures/compare_profiles.png)

**Figura 6** — Prestazioni di classificazione dei profili, rete ufficiale e controllo. File: `results/figures/compare_profiles.png`.

### 5.6 Risultati negativi

Presentiamo come risultati anche le strade che non hanno funzionato, perché delimitano il contributo. Le configurazioni e le metriche complete sono in `archive/storico.md`.

- **Il contenuto non porta il legame.** Controllo A/B/C (§5.3) indistinguibile; AUC con negativi appaiati sul reale nell'intorno del caso; peso appreso del contenuto un ordine di grandezza sotto quello della struttura.
- **Il fine-tuning dell'encoder non passa.** Addestrato con loss contrastiva sugli archi reali e negativi appaiati e temporali, peggiora **anche la validation** oltre alle due finestre di test: la media val+test scende da 0,534 a **0,519** per MiniLM e da 0,530 a **0,523** per BGE-base. La supervisione è troppo debole e l'encoder si adatta al periodo che vede.
- **Le feature dei nodi addestrabili peggiorano.** Rendere addestrabile la proiezione inizializzata dal contenuto abbassa la media val+test da 0,534 a **0,512** sulla codifica concatenata: aggiungere capacità porta all'overfitting. La versione completa della matrice appresa del paper (~70M parametri) non entra nella memoria disponibile.
- **Il campionamento dei negativi "misto" peggiora la rete.** Sostituendo i negativi uniformi con una miscela 30/30/40: transitività 0,0115 contro 0,0209, clustering 0,0305 contro 0,043, archi fra co-arrivati 18,4% contro 12,4%. I pesi cambiano molto — novità della controparte +0,61 contro +1,22, grado logaritmico −0,21 contro +1,02 — ma la reciprocità **non si muove**: la genera la tabella delle coppie ricambiate, non il classificatore.
- **"Il post come arco" non funziona.** Il testo del singolo post non separa le coppie collegate da quelle appaiate (AUC 0,515–0,517 contro 0,5166 della media) e i post di una coppia reciproca non si somigliano più del caso. La parte strutturale dell'idea non è verificabile: il dataset non registra quale post sia agganciato a ogni arco.
- **L'ablazione per AUC è satura e non serve.** Togliendo il punteggio di contenuto dal modello, l'AUC sul reale passa da 0,9818 a 0,9816: una misura così alta non distingue i due casi, ed è per questo che la prova del §5.3 è il confronto fra reti.

*Origine: `results/encoder_comparison.csv`, `results/finetune_report.csv`; per le configurazioni scartate, `archive/storico.md`.*

---

## 6. Discussione

**Cosa spiega la struttura, e cosa non spiega il contenuto.** Le misure convergono su una lettura sola. Il meccanismo che genera i legami in questo sistema è **relazionale e sequenziale**: gli archi nascono quando entra un utente, vanno verso utenti più connessi della media, si concentrano in pochi arrivi di grande dimensione e in parte vengono ricambiati subito. I pesi appresi (novità e grado dominanti, contenuto con un ordine di grandezza in meno) e i controlli A/B/C dicono la stessa cosa. Il testo, invece, contiene in modo netto *chi è* un utente — il contenuto predice attributi di profilo con AUC fino a 0,867 — ma **non** *con chi si collega*: il canale che spiega l'identità non spiega la relazione. Il secondo risultato, meno intuitivo, è che l'omofilia del contenuto della rete reale non contraddice questa lettura: 2,13 è in gran parte posizione nella rete (grado), e il residuo attribuibile alla somiglianza è piccolo (1,29) e non vive nel canale usato dal generatore.

**Quanto pesa il trasferimento fra reti di dimensioni diverse.** Per trasferire una preferenza di grado da 25.714 utenti a 10.000 esprimiamo il grado in **percentili**, e il grado medio dell'estremo già presente esce 153 nella rete ufficiale contro 437: sembrava una sottostima, ma è vicino al **rapporto fra le popolazioni** (0,35 contro 0,385). Lo stesso vale per il massimo: il grado massimo cresce con la popolazione secondo una legge che si stima **sul reale stesso** (i suoi nove snapshot: esponente **1,445**, R² 0,986), e a 10.000 utenti quella legge prevede **~339**; il controllo arriva a **328** (97%), la rete ufficiale a **274** (81%). Quello che il trasferimento **non** riproduce è la **forma** della coda — a pari dimensione la pendenza del decile alto è −3,98 (ufficiale) e −3,33 (controllo) contro −1,69 del reale — e la **coesione** (transitività 0,021 contro 0,034, clustering 0,042 contro 0,080). Anche la riscalatura del volume è una scelta di bersaglio e non un errore: la configurazione adottata fa portare al sintetico la densità matura del reale, mentre l'alternativa dettata dalla legge del reale fa coincidere la densità ma dimezza la transitività e abbassa il massimo a 0,48 del previsto.

**Minacce alla validità.** *Validità di costrutto.* Le statistiche di rete che usiamo sono proxy: la modularità dipende dall'algoritmo di comunità scelto, e la misura standard di omofilia è confusa dal grado, come mostrato al §5.4. Per questo affianchiamo sempre alle statistiche i controlli (grado, volume di testo). *Validità interna.* Due assunzioni restano nostre: l'ordine di arrivo nel sintetico (dedotto dagli identificatori) e l'ordine interno al passo; le abbiamo rese variabili di controllo con due partizioni, ma non possono essere verificate. Il loro confronto dice che la scelta pesa poco sulle metriche relazionali (distanza somma 3,84 contro 3,89), ma pesa su due aspetti dichiarati: la quota di archi fra co-arrivati (17,0% contro 12,4%, reale 8,1%) e il grado massimo (81% contro 97% del valore atteso dalla legge di scala). La partizione progressiva è stata preferita per **fedeltà al meccanismo di nascita** — nel reale il 58% degli utenti entra durante l'osservazione — e non per il vantaggio metrico, che non c'è. Inoltre il trasferimento fra popolazioni di dimensioni diverse è un'estrapolazione, che abbiamo mitigato con variabili relative, e la legge che riscala `k` è applicata sotto il range su cui è fittata. Infine i coefficienti appresi oscillano leggermente fra addestramenti (il contenuto intorno a +0,05), quindi vanno letti come stime. *Validità esterna.* Il risultato riguarda **un** dataset reale e **un** dataset sintetico, entrambi forniti per il progetto; non sappiamo se il meccanismo appreso generalizzi ad altre piattaforme o domini, e non abbiamo un secondo dataset con archi per verificarlo. *Validità di conclusione.* Le tabelle principali provengono da un'**unica esecuzione finale** dei modelli con le stesse analisi, quindi sono confrontabili fra loro ma non hanno intervalli di confidenza; i controlli fini sul contenuto sono instabili per passo. Il confronto fra encoder, in particolare, ha una risoluzione inferiore al margine fra il primo e il terzo classificato (§4.1).

---

## 7. Conclusioni

Abbiamo costruito la rete di collegamenti di una popolazione di cui si osserva soltanto il testo, imparando il meccanismo di formazione dei legami da una rete reale e applicandolo per inferenza, senza mai agganciare i due dataset. Il modello per eventi di arrivo riproduce la struttura della rete reale nell'ordine di grandezza giusto su reciprocità (0,3224 contro 0,4177), grado medio (27,45 contro 32,05), grado massimo (274 contro 1.290), transitività (0,0212 contro 0,0337), clustering (0,042 contro 0,080) e diametro (6 contro 6). Rispetto alle famiglie a soglia e a conteggio imposto il vantaggio è di **contemporaneità**, non di supremazia su ogni singola metrica: sulle sette metriche confrontabili prende scala dei gradi, reciprocità, modularità e transitività **senza imporre statistiche alle reti**; la sua coda resta **più ripida** di quella reale a pari dimensione (−3,98 contro −1,69). Resta un po' più "piatta" del reale, e i due punti su cui il lavoro può continuare sono la **forma** della coda e la **coesione**.

Sul contenuto il lavoro produce due risultati controllati. Primo: **il contenuto non decide chi si collega a chi** — con testo autentico, testo permutato fra utenti e contenuto azzerato le reti sono indistinguibili. Secondo: il numero che sembrava dire il contrario, l'omofilia 2,13 della rete reale, è **in gran parte posizione nella rete**, e scende a 1,51 e 1,29 con controlli di grado. Il testo conosce l'utente (attributi di profilo con AUC fino a 0,867) ma non i suoi legami, che dipendono dall'ingresso dell'utente nella finestra osservata e dal suo grado.

Le direzioni che restano aperte sono precise: una forma di trasferimento che conservi la **coda** dei gradi e non solo la posizione relativa; la **dipendenza fra i bersagli scelti dentro lo stesso evento di arrivo**, che chiuderebbe la parte triadica; e una rappresentazione del contenuto che agisca **a livello di coppia** invece che di singolo utente, l'unico modo in cui il testo potrebbe ancora entrare nella formazione dei legami.

---

## Riproducibilità e disponibilità dei dati

Tutte le cifre di questo articolo provengono da file in `results/` e da un'**unica esecuzione finale** dei modelli, con seed fissi e con le stesse analisi applicate alla rete reale e a ogni rete generata. I generatori iterano gli archi in **ordine ordinato**, perché l'ordine di iterazione dei set di stringhe non è fissato in Python: senza questa precauzione il volume della rete poteva variare di circa il 2% fra un processo e l'altro. La ripetibilità ha però un confine, che misuriamo e dichiariamo: il **generatore** è deterministico — rilanciando il comando documentato si riottengono gli **stessi** 7.776 eventi di arrivo, lo stesso numero di archi per evento (34,934), la stessa quota di coppie ricambiate (0,4435) e la stessa reciprocità (0,3224) — mentre l'**addestramento** non è riproducibile al bit, perché i pesi della rete variano al terzo decimale fra processi e il volume passa da 271.646 a **271.655** archi, cioè 9 archi su 271 mila (0,003%). La stessa variabilità colpisce il confronto fra encoder, dove il margine fra il primo e il terzo è di 0,002 in una variabilità di circa 0,005: per questo il confronto va letto come una parità fra tre encoder. La scelta dell'encoder, il fitting del generatore e le analisi strutturali, di contenuto, di accrescimento e dei profili sono descritti per intero nel codice sorgente; gli artefatti numerici sono in `results/*.csv` e le figure in `results/figures/`. La rete reale non è mai stata modificata; le reti generate sono artefatti nuovi.

I due dataset non sono ridistribuiti qui. Poiché i dataset non sono pubblici, la riproducibilità piena richiede l'accesso agli stessi input; i parametri, i seed e le procedure sono comunque documentati in modo da poter riapplicare il metodo a dati equivalenti.

---

## Riferimenti

1. Singer, U., Guy, I., Radinsky, K. (2019). *Node Embedding over Temporal Graphs.* In Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI), 4605–4612. arXiv:1903.08889.
2. Grover, A., Leskovec, J. (2016). *node2vec: Scalable Feature Learning for Networks.* In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 855–864. arXiv:1607.00653.
3. Nguyen, G. H., Lee, J. B., Rossi, R. A., Ahmed, N. K., Koh, E., Kim, S. (2018). *Continuous-Time Dynamic Network Embeddings.* In Companion Proceedings of the Web Conference (WWW), 969–976. doi:10.1145/3184558.3191526.
4. Sankar, A., Wu, Y., Gou, L., Zhang, W., Yang, H. (2020). *DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks.* In Proceedings of the 13th ACM International Conference on Web Search and Data Mining (WSDM), 519–527. arXiv:1812.09430.
5. Pareja, A., Domeniconi, G., Chen, J., Ma, T., Toyoda, T., Holzer, M., Karypis, G. (2020). *EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs.* In Proceedings of the AAAI Conference on Artificial Intelligence, 34(04), 5363–5370. arXiv:1902.10191.
6. Goyal, P., Chhetri, S. R., Canedo, A. (2020). *dyngraph2vec: Capturing network dynamics using dynamic graph representation learning.* Knowledge-Based Systems, 187, 104816. arXiv:1809.02657.
7. Kazemi, S. M., Goel, R., Jain, K., Kobyzev, I., Sethi, A., Forsyth, P., Poupart, P. (2020). *Representation Learning for Dynamic Graphs: A Survey.* Journal of Machine Learning Research, 21(70), 1–73.
8. Zhou, L., Yang, Y., Ren, X., Wu, F., Zhuang, Y. (2018). *Dynamic Network Embedding by Modeling Triadic Closure Process.* In Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).
9. You, J., Ying, R., Ren, X., Hamilton, W. L., Leskovec, J. (2018). *GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models.* In Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, 5708–5717. arXiv:1802.08773.
10. Bojchevski, A., Shchur, O., Zügner, D., Günnemann, S. (2018). *NetGAN: Generating Graphs via Random Walks.* In Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, 610–619. arXiv:1803.00816.
11. Leskovec, J., Chakrabarti, D., Kleinberg, J., Faloutsos, C., Ghahramani, Z. (2010). *Kronecker Graphs: An Approach to Modeling Networks.* Journal of Machine Learning Research, 11(33), 985–1042.
12. Barabási, A.-L., Albert, R. (1999). *Emergence of Scaling in Random Networks.* Science, 286(5439), 509–512.
13. Watts, D. J., Strogatz, S. H. (1998). *Collective dynamics of 'small-world' networks.* Nature, 393, 440–442.
14. Leskovec, J., Kleinberg, J., Faloutsos, C. (2005). *Graphs over Time: Densification Laws, Shrinking Diameters and Possible Explanations.* In Proceedings of the 11th ACM SIGKDD International Conference on Knowledge Discovery in Data Mining (KDD), 177–187.
15. Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* In Proceedings of EMNLP-IJCNLP, 3982–3992. arXiv:1908.10084.
16. Muennighoff, N., Tazi, N., Magne, L., Reimers, N. (2023). *MTEB: Massive Text Embedding Benchmark.* In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2014–2037. arXiv:2210.07316.
17. Xiao, S., Liu, Z., Zhang, P., Muennighoff, N. (2024). *C-Pack: Packed Resources for General Chinese Embeddings.* In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 641–649. arXiv:2309.07597.
18. Blondel, V. D., Guillaume, J.-L., Lambiotte, R., Lefebvre, E. (2008). *Fast unfolding of communities in large networks.* Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008. arXiv:0803.0476.
19. Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
20. Gower, J. C. (1975). *Generalized Procrustes Analysis.* Psychometrika, 40(1), 33–51.
21. McPherson, M., Smith-Lovin, L., Cook, J. M. (2001). *Birds of a Feather: Homophily in Social Networks.* Annual Review of Sociology, 27, 415–444.
22. Granovetter, M. S. (1973). *The Strength of Weak Ties.* American Journal of Sociology, 78(6), 1360–1380.
23. sentence-transformers. *all-MiniLM-L6-v2* (model card). Hugging Face. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
