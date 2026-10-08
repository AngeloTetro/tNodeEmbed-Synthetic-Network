# Relazione del progetto — dalla rete reale alla rete sintetica

*Documento scritto per capire cosa abbiamo fatto, passo per passo, con linguaggio semplice. I numeri delle tabelle vengono dai file in `results/` (i conteggi dei dataset anche dai file originali); le figure da `results/figures/` e i percorsi sono relativi alla radice del progetto.*

---

## 1. L'obiettivo, in una pagina

Il progetto di esame riguarda **una rete sociale che evolve nel tempo**. I dati sono due:

- il **dataset reale**, che contiene utenti, i loro post e i **collegamenti** fra loro, distribuiti su **9 snapshot** (dal 2022 a luglio 2025);
- il **dataset sintetico**, che contiene **10.000 utenti** con i loro post e alcuni attributi del profilo (schieramento politico, interessi, professione…), ma **senza collegamenti e senza tempo**.

L'obiettivo finale, confermato dal dottorando, è **costruire la rete degli utenti sintetici**: cioè decidere quali utenti sintetici sono collegati tra loro. Il metodo da usare è **tNodeEmbed**, un modello pubblicato nel 2019 che impara rappresentazioni dei nodi *tenendo conto del tempo*.

La difficoltà principale, che avevamo trovato subito, è questa: tNodeEmbed di base usa **node2vec** come primo passo, e node2vec **non sa rappresentare un utente che non ha ancora collegamenti**. Nel nostro dataset ogni collegamento nuovo coinvolge almeno un utente nuovo, quindi applicando il protocollo di valutazione originale **nessun arco risulta ammissibile nel test** (zero archi, in tutte le transizioni) — in pratica non si poteva valutare nulla. Il dottorando ha indicato la soluzione: **togliere node2vec** e usare come punto di partenza **le feature dei nodi**, che nel nostro caso sono **il testo dei post**. Il paper stesso lo permette (sezione 3.1: si può usare qualunque metodo di embedding statico come inizializzazione).

Le sue conferme, testuali, sono: *"Addestri sul reale e fai inferenza sul sintetico"*, *"Sintetico-sintetico va bene"*, *"le feature testuali, che sono le uniche in comune tra i due dataset"*, e la libertà di **ripartire il sintetico** in più snapshot (evitando di modificare il reale).

## 2. I due dataset

| | **reale** | **sintetico** |
|---|---|---|
| utenti | 25.714 nodi | 10.000 |
| post | 173.475 cumulativi (154.580 nei file "per periodo") | 100.456 post utilizzabili (su 100.474 righe) |
| collegamenti | 824.045 archi **diretti** | **nessuno** |
| tempo | 9 snapshot, dal 2022 a luglio 2025 | **nessuno**: 4 batch in ordine casuale |
| cosa sapere su ogni utente | solo il testo dei suoi post | 11 attributi di profilo + post |
| suddivisione dichiarata | 6 snapshot di training, 1 di validation, 2 di test | decisa da noi: 9 snapshot (vedi §5) |

Due cose dette dal dottorando che usiamo come vincoli: **degli utenti reali non esistono attributi** oltre ai post (e il dataset reale **non va modificato**); l'**arco non ha un significato specifico** — è un'interazione fra due utenti, che possiamo leggere come un "follow" solo per comodità. Un campo del sintetico, `exact_level_found`, viene da un progetto sulla tossicità dei commenti e **non c'entra con il nostro lavoro**.

## 3. Che cos'è tNodeEmbed, in parole semplici

Immaginiamo di voler descrivere ogni utente con una freccia (un vettore di numeri) che ne riassume la posizione nella rete. Se la rete cambia nel tempo, la freccia deve cambiare anche lei. tNodeEmbed funziona in **tre passi**:

1. **Inizializzazione**: per ogni istante di tempo (nel nostro caso, ogni snapshot) si dà a ogni utente una rappresentazione di partenza. Nel paper si usa node2vec; **noi usiamo il contenuto dei post**.
2. **Allineamento (Procrustes)**: le rappresentazioni di due snapshot consecutivi vivono in due "sistemi di coordinate" leggermente diversi; l'allineamento le ruota una sull'altra, così l'evoluzione di un utente non viene confusa con un cambio di coordinate.
3. **LSTM**: una rete ricorrente legge la **sequenza** delle rappresentazioni di un utente (snapshot 1 → 9) e la riassume in un'unica freccia finale, che tiene conto della sua storia.

A valle c'è un **classificatore (la "testa")** che risponde alla domanda operativa: *data una coppia di utenti, quanto è probabile che siano collegati?* È quello che ci serve per generare la rete del sintetico.

## 4. La pipeline che abbiamo costruito

| passo | cosa fa | script |
|---|---|---|
| 1 | **Rappresentare il contenuto** di ogni utente per ogni snapshot (encoder scelto al §5) | `src/content_encoders.py` |
| 2 | **Addestrare tNodeEmbed sul reale**: sequenze → allineamento → LSTM → testa di link prediction, con lo split 6/1/2 | `src/tnodeembed_real.py`, `src/transfer_to_synthetic.py` |
| 3 | **Ripartire il sintetico in 9 time step** (due varianti) | `src/build_synthetic_snapshots.py` |
| 4 | **Fare inferenza sul sintetico**: applicare il modello alle coppie di utenti sintetici e scrivere la rete snapshot per snapshot | `src/transfer_to_synthetic.py` |
| 5 | **Analizzare i grafi** (struttura, evoluzione, contenuto) su reale e sintetico con lo stesso codice | `src/analyze_structure.py`, `analyze_temporal.py`, `analyze_content.py`, `analyze_profiles.py` |
| 6 | **Confrontare** i tre dataset con figure uniche | `src/compare_folders.py` |

Lettura e scrittura dei file sono centralizzate in `src/graph_io.py`, così ogni analisi legge allo stesso modo il reale e il sintetico (i file hanno lo stesso formato: `social_network.edg` con due colonne, `posts_current_snapshot.tsv`, `posts_incremental.tsv`).

## 5. Perché abbiamo fatto certe scelte

**a) Il tempo nel sintetico lo imponiamo noi.** Il sintetico non ha date. I suoi post hanno però un `id` che è una **sequenza densa (0…100.455)**: è l'unico ordine "di fabbrica" che esiste. Ordiniamo quindi i post di ogni utente per `id` e li distribuiamo su **9 snapshot**, così ogni utente ha contenuto che si accumula nel tempo. Abbiamo preparato **due varianti**:
- `generated_snapshots/`: **tutti i 10.000 utenti hanno contenuto dal primo snapshot** (nella rete, però, compaiono solo quelli che hanno almeno un arco: al primo snapshot sono 7.134);
- `generated_snapshots_growth/`: gli utenti **entrano progressivamente**, seguendo la curva di crescita del dataset reale (41,8% nel primo snapshot, poi 14,2%, 11%, … fino al 100%).

**b) L'encoder si sceglie con i numeri.** Il contenuto è l'unica informazione che abbiamo, quindi la sua rappresentazione conta. Abbiamo confrontato 6 encoder (tf-idf+SVD, MiniLM, BGE-small/base, GTE-small/base) addestrando **tutta la pipeline** sul reale e guardando validation e test. La colonna della decisione è quella con i **negativi appaiati** (spiegati al punto c), e vince **MiniLM** con la media di 0,534 (gte-base 0,532, bge-base 0,530: il campo è stretto).

**c) I negativi della valutazione sono appaiati.** Per valutare un modello di link prediction servono anche coppie di utenti *non* collegati ("negativi"). Se si scelgono a caso, il modello può "imbrogliare" riconoscendo l'utente nuovo dalla sequenza; per evitarlo, nel confronto appaiamo ogni coppia vera con una coppia falsa che ha **lo stesso primo utente** e una destinazione con **storia di arrivo e grado simili**. Nel CSV riportiamo anche la versione con negativi casuali, per trasparenza.

**d) Come nascono gli archi: due famiglie, ed è il confronto la parte scientifica del lavoro.** Il modello dà un **punteggio** a ogni coppia di utenti; per trasformare i punteggi in una rete servono delle decisioni, e la domanda giusta è *da dove vengono*. Ne abbiamo confrontate tre famiglie:

- **Famiglia "a mano" (v1–v5)**, tenuta solo come confronto: il numero di archi lo decidiamo noi (`k ~ Poisson(1,2)`, oppure la distribuzione dei gradi del reale), eventualmente con l'accrescimento e la reciprocità aggiunta a mano.
- **Famiglia "calibrata sul reale" (v6)**: la soglia non è più una nostra scelta ma un **quantile dei punteggi del modello misurato sulle transizioni reali** — `v6a` fissa il grado uscente medio finale al valore reale (32,05), `v6b` fissa la crescita di ogni snapshot a quella reale — con i vincoli strutturali misurati sul reale (un utente può collegarsi solo a chi entra in quel momento, bersagli scelti in proporzione al grado) e la reciprocità portata al valore reale (0,42) con gli archi inversi.
- **Famiglia "per inferenza" (v7)**, il metodo più corretto: si **fitta un modello di formazione degli archi** sulle transizioni reali — le variabili sono il punteggio di contenuto del modello tNodeEmbed, il grado dei due utenti al passo precedente, se ciascuno dei due è nuovo, e il passo temporale; l'etichetta è *"questa coppia è diventata un arco?"* — e poi si **applica** al sintetico. Qui **non c'è nessun vincolo imposto**: nessuna quota di archi decisa da noi, nessun obbligo di reciprocità, e gli archi fra utenti già presenti sono ammessi **se il modello li ritiene probabili** (nel reale sono rari, ma non impossibili: è la risposta alla domanda "due utenti già presenti possono collegarsi più avanti?" — sì, ed è il modello a deciderlo, non una regola scritta da noi).

La differenza fra le ultime due è tutta lì: in v6 le statistiche del reale vengono **imposte** alla rete, in v7 vengono **imparate** e poi applicate. Teniamo entrambe come reti ufficiali del progetto, perché rispondono a due domande diverse: *cosa predice il modello* (v7) e *cosa serve per somigliare al reale* (v6).

**e) Il fallback del dottorando non è servito.** Aveva proposto, se necessario, di agganciare gli utenti sintetici alla rete reale come nodi isolati. La rimozione di node2vec è bastata: non l'abbiamo implementato.

## 6. I risultati

### 6.1 La scelta dell'encoder (sul reale)

AUC su validation / test1 / test2 (`results/encoder_comparison.csv`):

| encoder | negativi **appaiati** | negativi casuali | media val+test (appaiati) |
|---|---|---|---|
| tf-idf+SVD | 0,489 / 0,521 / 0,505 | 0,986 / 0,931 / 0,941 | 0,505 |
| **MiniLM** | 0,571 / 0,521 / 0,510 | 0,995 / 0,954 / 0,961 | **0,534** |
| BGE-small | 0,541 / 0,517 / 0,519 | 0,996 / 0,958 / 0,967 | 0,526 |
| GTE-small | 0,529 / 0,514 / 0,507 | 0,996 / 0,956 / 0,963 | 0,516 |
| BGE-base | 0,540 / 0,517 / 0,535 | 0,996 / 0,960 / 0,968 | 0,530 |
| GTE-base | 0,543 / 0,523 / 0,531 | 0,996 / 0,959 / 0,969 | 0,532 |

In tutte le righe il numero di archi che soddisfano la **restrizione del protocollo originale** ("il test contiene solo utenti già visti") è **0**: è la conferma che il problema denunciato esiste e che il protocollo è stato adattato.

### 6.2 Il transfer e la rete generata

Modello addestrato sul reale (MiniLM, negativi appaiati) e applicato al sintetico (`results/transfer_report.csv`):

| | AUC reale (val/test1/test2) | nodi | archi | densità | reciprocità | parole sintetiche nel vocabolario reale |
|---|---|---|---|---|---|---|
| tutti presenti | 0,573 / 0,526 / 0,523 | 10.000 | 107.511 | 0,00108 | 0,001 | 55,5% (27.317 su 49.262) |
| arrivo progressivo | 0,578 / 0,526 / 0,529 | 10.000 | 83.017 | 0,00083 | 0,000 | 55,5% |

La **sovrapposizione di vocabolario** è un controllo utile: poco più di metà delle parole usate nei post sintetici compare anche nei post reali, quindi il modello addestrato sul reale ha comunque una base testuale su cui lavorare. (Nota: qui "nodi" sono gli utenti **presenti** nella partizione, cioè quelli che hanno post; nella tabella §6.3 la stessa colonna conta invece i nodi che **compaiono nella rete**, cioè quelli con almeno un arco — da qui le differenze di nodi e densità per la variante "arrivo progressivo": 10.000 utenti presenti, 9.772 con almeno un arco.)

### 6.3 Struttura: reale contro generato

Valori dell'ultimo snapshot (`results/structure_*.csv`, `structure_heavy_*.csv`):

| grandezza | reale | generato (tutti presenti) | generato (arrivo progressivo) |
|---|---|---|---|
| nodi | 25.714 | 10.000 | 9.772 |
| archi | 824.045 | 107.511 | 83.017 |
| densità | 0,00125 | 0,00108 | 0,00087 |
| grado uscente medio | 32,0 | 10,8 | 8,5 |
| grado uscente massimo | 1.290 | 28 | 25 |
| reciprocità | **0,418** | **0,0011** | **0,0004** |
| componenti fortemente connesse | 7.343 | 9.965 | 9.735 |
| k-core massimo | 125 | 19 | 18 |
| comunità (Louvain) | 9 | 9 | 9 |
| modularità | 0,275 | 0,167 | 0,204 |
| transitivity | 0,034 | 0,506 | 0,300 |
| clustering medio | 0,080 | 0,747 | 0,577 |
| diametro (approssimato) | 6 | 3 | 4 |

![Crescita di nodi e archi nei tre dataset](results/figures/compare_growth.png)

![Densità, reciprocità e modularità nel tempo](results/figures/compare_structure.png)

![Coda della distribuzione del grado uscente](results/figures/compare_degree_ccdf.png)

### 6.4 Evoluzione

`results/temporal_*.csv`:

| | reale | generato (tutti) | generato (growth) |
|---|---|---|---|
| archi rimossi in ogni transizione | **0** | 0 | 0 |
| Jaccard degli archi (persistenza), primo → ultimo | 0,52 → 0,96 | 0,50 → 0,89 | 0,42 → 0,86 |
| stabilità delle comunità (NMI), primo → ultimo | 0,52 → 0,60 | 0,41 → 0,48 | 0,47 → **0,74** |
| nuovi nodi per snapshot | 3.652 → 1.073 | 2.006 → 1 | 1.863 → 466 |

(La serie "nuovi nodi" parte dal **secondo** snapshot: il primo non ha una transizione precedente con cui confrontarsi.)

![Evoluzione del reale: nuovi archi, persistenza, stabilità delle comunità, densificazione](results/figures/temporal_real_snapshots_summary.png)

### 6.5 Il contenuto spiega i collegamenti?

Misuriamo quanto il contenuto di due utenti collegati è più simile di quello di due utenti presi a caso (`ratio`: 1 vuol dire "nessuna differenza", più di 1 vuol dire "i collegati si somigliano di più").

| | reale | generato (tutti) | generato (growth) |
|---|---|---|---|
| ratio al primo snapshot | 1,49 | 0,87 | 0,72 |
| ratio all'ultimo snapshot | **2,13** | 0,996 | 1,002 |

Nel **reale** i collegati si somigliano sempre di più col tempo (fino a 2,13 volte le coppie casuali). Nella **rete generata** questo effetto non c'è: 1,0 significa che i collegamenti non sono più "simili nel contenuto" di quanto lo sarebbero due utenti estratti a caso.

![Omofilia del contenuto: reale contro reti generate](results/figures/compare_homophily.png)

Lo stesso vale per gli attributi del profilo sintetico (lift = quante volte più spesso dell'atteso un arco collega due utenti con lo stesso valore). Valori dell'ultimo snapshot per la variante "tutti presenti" / "arrivo progressivo": schieramento politico 1,17 / 1,00, professione 0,98 / 0,87, interessi 0,61 / 1,05 — quindi **nessun attributo viene riprodotto** dalla rete generata in modo apprezzabile (un lift di 1 significa "come il caso").

### 6.6 I profili si possono prevedere?

Sul sintetico gli attributi del profilo sono etichette "vere", quindi possiamo chiedere: un classificatore addestrato sull'**80% degli utenti** (split casuale stratificato, seed 0) sa indovinare lo schieramento politico, la professione, la fascia d'età del restante **20%**? La metrica è l'AUC macro "uno contro tutti" (`results/profiles_*.csv`):

| etichetta | solo contenuto | contenuto + rete | embedding tNodeEmbed | embedding + rete |
|---|---|---|---|---|
| political_leaning (tutti) | **0,827** | 0,827 | 0,689 | 0,689 |
| profession (tutti) | **0,867** | 0,866 | 0,623 | 0,618 |
| age_interval (tutti) | 0,690 | **0,702** | 0,587 | 0,585 |
| political_leaning (growth) | **0,827** | 0,828 | 0,683 | 0,694 |
| profession (growth) | **0,867** | 0,871 | 0,654 | 0,654 |

Il contenuto "grezzo" classifica meglio degli embedding temporali passati per la LSTM, e aggiungere la rete non cambia quasi nulla.

![Classificazione dei profili](results/figures/compare_profiles.png)

### 6.7 Chi si collega a chi (utenti nuovi e utenti già presenti)

Per ogni transizione classifichiamo gli archi nuovi secondo i due estremi: un utente è "nuovo" se compare **nella rete** per la prima volta in quello snapshot, altrimenti è "già presente" (entrato in uno snapshot precedente). Dati in `results/accretion_*.csv`.

| | reale | generato (tutti presenti) | generato (arrivo progressivo) |
|---|---|---|---|
| archi nuovi considerati | 707.604 | 95.302 | 77.936 |
| nuovo → già presente | 49% | 5% | 1,4% |
| già presente → nuovo | 43% | 0% | 73% |
| nuovo → nuovo | 8% | 0% | 14% |
| **già presente → già presente** | **0%** | **95%** | 12% |
| grado medio dell'estremo già presente | 170 → 437 | 2 → 6 | 3 → 15 |
| grado medio di tutti i nodi | 22 → 64 | 3 → 19 | 3 → 15 |

Lettura: nel **reale** i nuovi utenti **si collegano agli utenti degli snapshot precedenti** — il 92% degli archi nuovi è "nuovo &lt;-&gt; già presente" — e **non esiste nessun arco fra due utenti già presenti** (0 su 707.604); inoltre i nuovi utenti scelgono utenti **molto più connessi della media** (437 contro 64). Nella variante **arrivo progressivo** il collegamento nuovo &lt;-&gt; già presente c'è (74%), compare una quota di nuovo &lt;-&gt; nuovo (14%), ma resta un 12% di archi fra utenti già presenti (che nel reale non esiste) e **manca la preferenza per i nodi molto connessi** (il grado dell'estremo è in linea con la media). Nella variante **tutti presenti** il concetto di "nuovo" quasi non esiste — tutti hanno contenuto dal primo snapshot — quindi il **95% degli archi è fra utenti già presenti**: è la differenza strutturale principale fra le due partizioni.

### 6.8 Le due famiglie ufficiali a confronto con il reale

Valori dell'ultimo snapshot (`results/variant_summary.csv`):

| rete | nodi | archi | densità | reciprocità | modularità | transitivity | omofilia |
|---|---|---|---|---|---|---|---|
| **reale** | 25.714 | 824.045 | 0,00125 | **0,418** | 0,275 | 0,034 | **2,13** |
| v6a — calibrata sul grado (tutti) | 10.000 | 64.422 | 0,00064 | **0,420** | 0,661 | 0,000 | 0,99 |
| v6a — calibrata sul grado (crescita) | 8.345 | 78.268 | 0,00112 | **0,420** | 0,503 | 0,006 | 1,00 |
| v6b — calibrata sulla crescita (crescita) | 8.800 | 399.464 | 0,00516 | **0,420** | 0,364 | **0,045** | 1,00 |
| v7b — inferenza, soglia a blocchi (tutti) | 10.000 | 61.560 | 0,00062 | 0,016 | 0,548 | 0,424 | 1,00 |
| v7c — inferenza, soglia globale (tutti) | 10.000 | 57.558 | 0,00058 | 0,183 | 0,737 | 0,093 | 1,00 |
| v7e — inferenza, globale + fine-tuned (tutti) | 10.000 | 57.559 | 0,00058 | **0,721** | 0,460 | 0,005 | 0,99 |
| v7c — inferenza, soglia globale (crescita) | 9.882 | 35.918 | 0,00037 | 0,000 | 0,870 | 0,000 | 1,01 |
| v7e — inferenza, globale + fine-tuned (crescita) | 10.000 | 35.919 | 0,00036 | 0,210 | 0,761 | 0,003 | 1,03 |

Due cose da guardare con attenzione:

- **La reciprocità** (nel reale 0,42): la famiglia calibrata la centra per costruzione (0,420 esatto), quella per inferenza no — va da 0,00 a 0,72 a seconda di come si decide la soglia e dell'encoder. Non è un dettaglio: la reciprocità nasce dal fatto che nel reale i due versi di un legame si creano in momenti diversi, mentre la nostra rete di 9 passi concede poco tempo perché questo accada.
- **L'omofilia del contenuto** (nel reale **2,13**, cioè i collegati si somigliano più del doppio rispetto a due utenti a caso): **nessuna** rete generata la riproduce (tutte intorno a 1,00). È il risultato più importante del progetto: *nonostante il contenuto sia l'unica informazione usata*, le reti generate non riescono a far somigliare i collegati più di due utenti estratti a caso. La spiegazione sta nella sezione successiva.

### 6.9 Il modello di formazione degli archi (v7) — cosa ha imparato

Le variabili sono standardizzate, quindi i coefficienti sono confrontabili fra loro; la tabella è in `results/v7_coefficients.csv` (validation: AUC 0,988, AP 0,577).

| variabile | coefficiente |
|---|---|
| **punteggio di contenuto** (tNodeEmbed) | **+0,18** |
| log-grado dell'utente di partenza | +1,64 |
| log-grado dell'utente di arrivo | +1,57 |
| **l'utente di partenza è nuovo** | **+2,28** |
| **l'utente di arrivo è nuovo** | **+2,08** |
| passo temporale | +0,29 |

Il modello discrimina quasi perfettamente (AUC 0,988) ma **lo fa con la struttura**: il peso dell'utente nuovo è più di dieci volte quello del contenuto, e anche il grado pesa molto di più. Tradotto: *chi entra adesso* e *quanto è connesso* spiegano quasi tutto; *cosa scrive* quasi nulla. Questo è coerente con il fatto che nella rete generata l'omofilia del contenuto resti a 1,00 e con l'AUC intorno a 0,49–0,57 quando si controlla per arrivo e grado (§6.1).

Le due decisioni possibili sul modello danno esiti molto diversi:
- **decisione bayesiana con la prevalenza vera** (un arco c'è se il modello lo ritiene più probabile del contrario): **nessun arco** in tutte le partizioni — il modello non crede mai che una coppia sintetica abbia più del 50% di probabilità di essere un arco, perché la probabilità a priori è minuscola (6,4 × 10⁻⁵ coppie collegate);
- **soglia al tasso reale** (si tengono le coppie in testa alla graduatoria per il tasso misurato sul reale): le reti delle righe v7 della tabella precedente.

### 6.10 Il fine-tuning dell'encoder (leva 1), e perché non è passato

Abbiamo addestrato due encoder (MiniLM e bge-base) con una loss contrastiva sugli archi reali, negativi appaiati e negativi **temporali** (coppie che diventano archi più avanti ma non in quel passo: il tempo nel contesto), con early stopping sulla validation (`results/finetune_report.csv`). Al confronto di protocollo (`results/encoder_comparison.csv`, negativi appaiati):

| encoder | validation | test1 | test2 | media val+test |
|---|---|---|---|---|
| MiniLM (base) | 0,571 | 0,521 | 0,510 | 0,534 |
| **finetuned-minilm** | **0,549** | 0,509 | 0,500 | **0,519** |
| bge-base (base) | 0,540 | 0,517 | 0,535 | 0,530 |
| **finetuned-bge-base** | 0,529 | 0,524 | 0,518 | **0,523** |

Il fine-tuning **migliora la validation** (è la finestra su cui si ferma) ma **peggiora entrambe le finestre di test**: è overfitting — la supervisione è troppo debole e l'encoder si adatta al periodo che vede. Il criterio di tenuta (media migliore del non fine-tunato) **non è soddisfatto**, quindi la pipeline resta sugli encoder pre-addestrati. Le due caselle fine-tuned di v7 servono solo a misurare l'effetto sulla rete generata.

### 6.11 Il disegno 2 × 2: soglia × encoder

Per separare i due effetti abbiamo generato quattro reti, incrociando la modalità di soglia (a blocchi / globale) e l'encoder (MiniLM / fine-tuned), sempre con la stessa decisione al tasso reale:

| | soglia a blocchi | soglia globale |
|---|---|---|
| **MiniLM** | v7b: 61.560 archi, reciprocità 0,016 | v7c: 57.558 archi, reciprocità 0,183 |
| **fine-tuned** | v7d: 61.588 archi, reciprocità 0,002 | v7e: 57.559 archi, reciprocità 0,721 |

Lettura: **la soglia globale è la correzione che conta** (la reciprocità passa da 0,016 a 0,183: prima era un artefatto del calcolo a blocchi, che rompeva la simmetria fra (u,v) e (v,u)); **l'encoder fine-tuned non cambia il volume** (61.588 contro 61.560: identico) ma rende i punteggi più simmetrici, e la reciprocità sale a 0,72 — con un effetto collaterale da segnalare: compare un utente collegato a tutti gli altri (grado 9.999), cioè un hub degenerato che il reale non ha (grado massimo 1.290). Quindi nessuno dei due aggiustamenti avvicina la rete al reale nella sostanza: cambiano la simmetria, non la capacità di riprodurre la struttura.

### 6.12 La rappresentazione del contenuto: due ipotesi provate

Il dottorando ha suggerito che il problema fosse l'encoder (node2vec) e che, avendo noi le feature dei nodi, bastasse toglierlo — cosa che avevamo già fatto. Restava però un punto del paper che avevamo semplificato: nel paper le feature dei nodi (`Q_t`) sono **addestrabili**, mentre noi le tenevamo **congelate**. Abbiamo provato entrambe le direzioni.

**a) Le feature addestrabili non aiutano** (media val+test, negativi appaiati):

| MiniLM | validation | test1 | test2 | media |
|---|---|---|---|---|
| congelate (com'erano) | 0,571 | 0,521 | 0,510 | 0,534 |
| **addestrabili** | 0,533 | 0,527 | 0,476 | **0,512** |

Peggiora: aggiungere capacità addestrabile sui contenuti porta all'overfitting, come era successo col fine-tuning dell'encoder. (La versione completa del paper — una riga per utente e per snapshot — richiede più memoria di quella disponibile sulla scheda e non è stata provata; ha però più capacità, quindi il rischio sarebbe maggiore.)

**b) La codifica "per post" invece migliora.** Era un difetto vero della nostra rappresentazione: prima univamo i post di un utente in una sola stringa, che l'encoder tagliava al suo limite di token (per gli utenti molto attivi sopravviveva il 28% del testo). Ora ogni post viene codificato da solo e i vettori di un utente vengono mediati:

| encoder | concatenato | **per post** |
|---|---|---|
| MiniLM | 0,534 | **0,549** |
| bge-base | 0,530 | **0,540** |
| tf-idf | 0,505 | **0,524** |

Tre encoder su tre migliorano, e nel transfer la media passa da 0,541 a 0,552. È un miglioramento pulito: non c'è addestramento in più, quindi nessun rischio di overfitting.

**c) Ma la rete generata non cambia.** È il test che conta: se il contenuto iniziasse a contare, l'**omofilia** della rete dovrebbe salire. Non sale: **1,004** con la rappresentazione nuova (0,996 con la vecchia; nel reale **2,13**). E il coefficiente del contenuto nel modello di formazione degli archi resta **≈ +0,20** contro **+2,28** di "l'utente è nuovo" (il valore oscilla fra 0,18 e 0,26 da un addestramento all'altro: è una stima, non una cifra esatta).

**Conclusione.** La rappresentazione andava migliorata, e l'abbiamo fatto; ma non era quello il collo di bottiglia. Il contenuto, da solo, non basta a spiegare chi si collega a chi, perché la rete reale è governata da un meccanismo — chi entra, quando entra, quanto è connesso — che un modello che guarda solo il testo **non può vedere**. Per andare oltre servirebbe un modello generativo che impari *anche* la struttura, non solo il contenuto.

### 6.13 V8 — il modello che impara il meccanismo, e cosa resta fuori

Questa è la versione definitiva del progetto. Invece di decidere noi quante connessioni creare, **si fitta sul reale un modello che dice con che probabilità una coppia di utenti diventa un collegamento**, e poi si **simula** la rete sintetica passo per passo: a ogni passo le variabili si ricalcolano **sullo stato della rete appena generata** (i gradi cambiano, gli utenti entrano) e ogni coppia diventa un arco con la propria probabilità. **Nessuna soglia, nessun conteggio deciso da noi**: volume, gradi e hub emergono dal modello.

Le variabili sono: il punteggio di contenuto di tNodeEmbed, il **grado relativo** dei due utenti (il percentile, per poter trasferire il meccanismo fra popolazioni di dimensioni diverse), se ciascuno è **nuovo**, se esiste già l'**arco inverso**, i **vicini in comune**, se i due hanno **scritto in quel periodo**, da quanti passi non ricevono un arco, e il passo temporale. Il modello è una regressione logistica: i suoi pesi *sono* il risultato scientifico. Un dettaglio non l'abbiamo scelto noi: misurando sul reale, **almeno un capo di ogni arco nuovo ha scritto in quel periodo nel 100% dei casi** — quindi è un fatto dei dati e definisce quali coppie sono candidate; invece che la *sorgente* sia attiva vale solo nel 68–81%, e resta una variabile che il modello pesa da sé.

| rete | archi | densità | grado medio | grado max | reciprocità | modularità | transitivity | omofilia |
|---|---|---|---|---|---|---|---|---|
| **reale** | 824.045 | **0,00125** | **32,05** | **1.290** | **0,418** | 0,275 | 0,034 | **2,13** |
| V8, tutti presenti | 129.321 | **0,00129** | 12,9 | 102 | 0,003 | 0,152 | 0,003 | 1,00 |
| V8, arrivo progressivo | 159.804 | 0,00160 | 16,0 | 125 | 0,004 | 0,140 | 0,004 | 1,00 |
| V8, vecchio scenario (controllo) | 168.015 | 0,00168 | 16,8 | 128 | 0,004 | 0,131 | 0,003 | 1,00 |
| V8, **senza contenuto** | 153.100 | 0,00153 | 15,3 | 126 | 0,004 | 0,142 | 0,004 | 1,00 |

**Cosa ha funzionato.** La **densità** della prima rete è **0,00129 contro 0,00125** del reale: praticamente identica. E il grado medio (12,9–16,8) si è avvicinato molto: la versione precedente, che usava una soglia, si fermava a 3,6–5,8 e a una densità 3-4 volte più bassa. Nessuno di questi valori è stato imposto: sono **emersi** dal campionamento delle probabilità fittate sul reale.

**Cosa resta fuori.** Tre cose, e vanno dette: la **coda dei gradi** (il grado massimo è 102–128 contro 1.290: mancano gli hub estremi), la **reciprocità** (0,004 contro 0,418: il modello non riproduce il fatto che i legami vengono ricambiati) e la **transitivity** (0,003 contro 0,034). A queste si aggiunge l'**omofilia del contenuto**: 1,00 contro 2,13.

**Il contenuto, misurato per ablazione.** Togliendo dal modello il punteggio di contenuto — tutto il resto identico — l'AUC di validation passa da **0,9818 a 0,9816** e la rete cambia del **4%** in volume, con **omofilia identica**. È la risposta definitiva alla domanda che ci siamo portati dietro: *il testo, così rappresentato, non aggiunge capacità di previsione e non cambia la rete.* Non è un'opinione, è un esperimento.

**Una cosa che avevo previsto male, e la dico.** Pensavo che cambiare lo scenario del sintetico (da attività uniforme ad attività "a raffica", presa dal reale) avrebbe reso la variabile "ha scritto in questo periodo" importante e avrebbe fatto crescere la rete "a ondate". Il confronto diretto dice che l'effetto sul **volume** è del ~5% e che **nessuna metrica strutturale cambia**. Lo scenario conta meno di quanto pensassi: lo documento come esito, non come conferma.

**Il coefficiente dei vicini in comune è negativo (−4,4), e non è un errore.** Le coppie di controllo sono estratte a caso, e gli archi veri coinvolgono spesso utenti appena entrati — che per definizione non hanno vicini in comune. Il modello impara quindi che, *contro una coppia a caso*, i triangoli non aiutano: è una proprietà della competizione che abbiamo scelto, e spiega la transitivity bassa della rete generata.

**In una frase.** V8 è la rete più vicina al reale che il progetto abbia prodotto sulla struttura di base (densità e gradi), è **imparata** e non imposta, e il suo limite è chiaro: il meccanismo che ha imparato è "chi entra e quanto è connesso", non "chi si somiglia" — e per questo le due proprietà che dipendono dalle relazioni (reciprocità e triangoli) e quella che dipende dal testo (omofilia) restano lontane.

## 7. Cosa non abbiamo fatto, e i limiti da tenere presenti

0. **V8 (il metodo definitivo) non riproduce tre proprietà del reale**: la coda dei gradi (massimo 102–128 contro 1.290), la **reciprocità** (0,004 contro 0,418) e la **transitivity** (0,003 contro 0,034). Le prime due dipendono da meccanismi relazionali che il modello, così com'è costruito, non cattura; la terza anche dal fatto che i negativi sono coppie a caso (che rende negativo il coefficiente dei vicini in comune).
1. **Il contenuto, da solo, non riproduce i collegamenti.** È il risultato principale e va detto per primo: le reti generate hanno **omofilia del contenuto ≈ 1,00** mentre il reale è **2,13**, e nel modello di formazione degli archi il peso del punteggio di contenuto è **≈ +0,11** contro **+3,3** dell'"utente nuovo" (e l'ablazione di V8 lo conferma: 0,9818 → 0,9816 di AUC). Con un controllo per arrivo e grado, l'AUC del contenuto sta intorno a 0,49–0,57 (0,5 = caso). Quindi la pipeline spiega *quando* nascono i collegamenti (chi entra, quanto è connesso) ma non *fra chi*, nella misura in cui conta il contenuto.
2. **La famiglia per inferenza (v7) non riproduce la struttura del reale**: grado uscente medio 3,6–5,8 contro 32,0; reciprocità da 0,00 a 0,72 (non stabile) contro 0,42; e con l'encoder fine-tuned compare un hub degenerato (un utente collegato a tutti, grado 9.999 contro un massimo reale di 1.290). La famiglia calibrata (v6) centra volume e reciprocità **per costruzione**, non perché li abbia imparati: è la differenza fra *imporre* e *predire*.
3. **Il fine-tuning dell'encoder non è passato**: migliora la validation ma peggiora entrambe le finestre di test (media 0,519 contro 0,534 per MiniLM; 0,523 contro 0,530 per bge-base). La supervisione (0,49–0,57 di AUC) è troppo debole perché l'encoder impari qualcosa che generalizzi. Il fine-tuning di bge-base ha girato con budget ridotto (4.000 coppie, 1 epoca, batch 8) perché la scheda da 6 GB andava in esaurimento di memoria con batch più grandi.
4. **La regola a mano (v1–v5) resta come confronto, non come metodo**: il conteggio era una nostra scelta (`k ~ Poisson(1,2)` o la distribuzione dei gradi del reale), ed è la ragione per cui quelle reti avevano reciprocità ≈ 0,001.
5. **L'inizializzazione `Q_t` del paper è usata in senso lato**: non c'è la matrice di embedding addestrabile (una riga per utente e snapshot) inizializzata dalle feature: le rappresentazioni del contenuto sono direttamente l'input della LSTM, e la versione addestrabile che entra nella scheda (una proiezione condivisa, §6.12) è stata provata e non ha aiutato. Allo stesso modo l'**allineamento di Procrustes è un pre-processing** e non compare come termine nella loss (anche l'implementazione di riferimento allena con la sola cross-entropy binaria).
6. **Il fallback dei nodi isolati** proposto dal dottorando **non è servito** e non è stato implementato.
7. **Dettagli tecnici**: 18 righe dei batch sintetici vengono ignorate in lettura (numero di campi diverso dall'intestazione); il confronto fra encoder copre solo la variante di contenuto "per periodo"; la partizione "tutti presenti" resta poco adatta alla famiglia v7, perché lì il concetto di "utente nuovo" quasi non esiste; i coefficienti del modello di formazione degli archi oscillano un po' fra un addestramento e l'altro (il contenuto fra +0,18 e +0,26), quindi vanno letti come stime.

**Cosa servirebbe per andare oltre**: un modello generativo che impari **anche la struttura** — grado, arrivo, reciprocità, chiusura triadica — insieme al contenuto. Tutte le misure di questo progetto puntano lì: il contenuto da solo non basta, e il modello attuale non può vedere la struttura, perché guarda soltanto il testo. Questa è la direzione, non un rimprovero alla pipeline: era il punto di partenza del dottorando (togliere node2vec e usare le feature), ed è stato fatto.

## 8. Glossario minimo

- **AUC**: misura di quanto un punteggio separa due gruppi (qui "coppie collegate" contro "coppie non collegate"). Va da 0,5 (risposta casuale) a 1 (separazione perfetta).
- **Snapshot / time step**: fotografia della rete (o dei post) a un certo momento; i 9 snapshot sono i 9 istanti della nostra linea temporale.
- **Encoder del contenuto**: il metodo che trasforma un testo in numeri (tf-idf, MiniLM, BGE, GTE…).
- **node2vec**: il metodo di embedding statico usato da tNodeEmbed per inizializzare i nodi; non sa rappresentare utenti mai visti, ed è la parte che abbiamo tolto.
- **Embedding**: il vettore di numeri che rappresenta un utente.
- **Procrustes**: la rotazione che allinea due sistemi di coordinate.
- **LSTM**: rete neurale che legge una sequenza e ne ricorda il riassunto.
- **Link prediction**: prevedere se due utenti sono collegati.
- **Negativi**: coppie di utenti non collegati usate come controprova durante l'addestramento e la valutazione.
- **Negativi appaiati**: negativi scelti con lo stesso primo utente e una destinazione con storia di arrivo e grado simili a quelli veri, così il modello non può "imbrogliare" riconoscendo gli utenti nuovi.
- **Omofilia**: tendenza di utenti simili a essere collegati.
- **Densità**: quanta parte dei collegamenti possibili esiste davvero.
- **Reciprocità**: quanta parte dei collegamenti è ricambiata (A→B e B→A).
- **Transitivity / clustering**: quanto spesso due vicini di un nodo sono vicini fra loro (quanto la rete forma triangoli).
- **k-core**: il grado di "nucleo" di un insieme di nodi molto connessi fra loro.
- **Modularità**: quanto la rete è divisa in gruppi coesi.
- **Louvain**: l'algoritmo che trova quei gruppi (le "comunità").
- **NMI**: quanto due suddivisioni in gruppi si somigliano (1 = identiche).
- **Modello di formazione degli archi** (v7): un modello che impara, sulle transizioni reali, con che probabilità una coppia di utenti diventa un collegamento; poi si applica al sintetico. È l'alternativa alle regole scritte a mano.
- **Fine-tuning contrastivo** (leva 1): addestrare l'encoder perché avvicini i testi degli utenti collegati e allontani quelli non collegati.
- **Negativi temporali**: coppie che diventano collegamenti **più avanti** ma non in questo momento; servono a insegnare all'encoder che la stessa coppia "non è ancora" collegata (il tempo nel contesto).
- **Prevalenza (e correzione della prevalenza)**: quanto è rara una cosa. Le coppie collegate sono rarissime (6,4 × 10⁻⁵), quindi le probabilità del modello vanno riportate a quella scala prima di decidere; è il motivo per cui la decisione "bayesiana" di v7 non produce nessun arco.
- **Soglia globale per passo**: una sola soglia per ogni istante di tempo, calcolata su tutte le coppie candidate; è la versione corretta della soglia "a blocchi" usata all'inizio, che rompeva la simmetria fra (u,v) e (v,u).
- **Omofilia del contenuto (ratio)**: quante volte i due capi di un collegamento si somigliano di più rispetto a due utenti estratti a caso (1 = nessuna differenza). Nel reale è 2,13, nelle reti generate è ~1,00.
- **Poisson(1,2)**: distribuzione da cui estraevamo il numero di nuovi archi per utente a ogni snapshot nella vecchia regola (v1–v5): **variante scartata**, tenuta nel confronto.

## 9. Come riprodurre tutto

```bash
# 1) confronto fra encoder sul reale (tutta la tabella §6.1; per un solo encoder basta --encoders minilm)
uv run python src/tnodeembed_real.py --encoders tfidf,minilm,bge-small,gte-small,bge-base,gte-base --content period --negatives random,matched

# 2) partizione del sintetico in 9 time step (due varianti)
uv run python src/build_synthetic_snapshots.py

# 3) transfer: addestra sul reale e genera la rete sintetica (riferimento, solo contenuto)
uv run python src/transfer_to_synthetic.py generated_snapshots minilm matched
uv run python src/transfer_to_synthetic.py generated_snapshots_growth minilm matched

# 4) le varianti a confronto: regole a mano (v1-v5), calibrate (v6), per inferenza (v7)
uv run python src/apply_edge_rules.py                       # v1-v5
uv run python src/apply_v6.py                               # v6a, v6b (soglia calibrata sul reale)
uv run python src/generate_inference.py --encoder minilm --threshold block  --variant v7b
uv run python src/generate_inference.py --encoder minilm --threshold global --variant v7c
# serve prima il fine-tuning dell'encoder (passo 4b), altrimenti l'encoder non esiste
uv run python src/finetune_encoder.py --base minilm   --pairs 8000 --batch 32 --epochs 2
uv run python src/finetune_encoder.py --base bge-base --pairs 4000 --batch 8  --epochs 1
uv run python src/generate_inference.py --encoder finetuned-minilm --threshold block  --variant v7d
uv run python src/generate_inference.py --encoder finetuned-minilm --threshold global --variant v7e

# 5) analisi: stessa cosa per reale_snapshots e per OGNI cartella generata (comprese le varianti)
uv run python src/analyze_structure.py real_snapshots
uv run python src/analyze_temporal.py  real_snapshots
uv run python src/analyze_content.py   real_snapshots
uv run python src/check_real_accretion.py real_snapshots
uv run python src/analyze_profiles.py  generated_snapshots
uv run python src/analyze_profiles.py  generated_snapshots_growth

# 6) tabelle e figure di confronto fra tutte le reti (reale + varianti)
uv run python src/variant_summary.py
uv run python src/compare_folders.py
```

I numeri finiscono in `results/*.csv` (in particolare `v7_report.csv`, `v7_coefficients.csv` per i coefficienti del modello e `variant_summary.csv` per il confronto unico), le figure in `results/figures/`, le reti generate (post + `social_network.edg`) nelle cartelle `Esame Angelantonio Fedele Murolo/generated_snapshots*/`.

Nota pratica: sulla RTX 3060 da 6 GB il fine-tuning di `bge-base` va lanciato con `--batch 8` (con batch 32 va in esaurimento di memoria); MiniLM regge `--batch 32`.
