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

**d) La regola con cui nascono gli archi è dichiarata.** Il modello dà un punteggio a ogni coppia; il *numero* di archi non lo decide il modello: a ogni time step ogni utente **che scrive in quel periodo** crea `k ~ Poisson(1,2)` nuovi archi verso gli utenti **già presenti nella rete** — quelli entrati in quello snapshot o in uno precedente, perché in una rete cumulativa nessuno esce mai — scegliendo quelli con il punteggio più alto, senza contare i self-loop e gli archi già presenti. Gli archi si accumulano, come nel reale.

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

## 7. Cosa non abbiamo fatto, e i limiti da tenere presenti

1. **Il numero di archi è una nostra scelta**, non una previsione del modello (`k ~ Poisson(1,2)`): è per questo che la **reciprocità della rete generata è ~ 0,001** mentre nel reale è 0,418. Varianti della regola di conteggio non sono state ancora provate. Inoltre la regola **non riproduce la preferenza per gli hub** (nel reale gli utenti nuovi si collegano a nodi con grado medio 437 contro 64) e, nella variante "tutti presenti", ammette archi fra utenti già presenti, che nel reale non esistono (0 su 707.604).
2. **L'inizializzazione `Q_t` del paper è usata in senso lato**: non c'è una matrice di embedding addestrabile inizializzata dalle feature; le rappresentazioni del contenuto sono direttamente l'input della LSTM. Allo stesso modo l'**allineamento di Procrustes è un pre-processing** e non compare come termine nella loss (anche l'implementazione di riferimento allena con la sola cross-entropy binaria).
3. **Sul reale il contenuto, da solo, non separa bene i collegamenti** quando si controlla per la struttura (arrivo e grado): è il motivo per cui le AUC con negativi appaiati stanno intorno a 0,49–0,57 (0,5 = risposta casuale).
4. **Non abbiamo usato il fallback dei nodi isolati** (non è servito) e **non abbiamo addestrato/fine-tuned un encoder** con loss contrastiva sui collegamenti reali: resta l'opzione più promettente, ora che la GPU funziona.
5. **Dettagli tecnici**: 18 righe dei batch sintetici vengono ignorate in lettura (numero di campi diverso dall'intestazione); il confronto fra encoder copre solo la variante di contenuto "per periodo"; il progetto **non è sotto git**, quindi "nessun file originale modificato" è verificato per ispezione del codice.

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
- **Poisson(1,2)**: distribuzione da cui estraiamo il numero di nuovi archi per utente a ogni snapshot: in media 1,2.

## 9. Come riprodurre tutto

```bash
# 1) confronto fra encoder sul reale (tutta la tabella §6.1; per un solo encoder basta --encoders minilm)
uv run python src/tnodeembed_real.py --encoders tfidf,minilm,bge-small,gte-small,bge-base,gte-base --content period --negatives random,matched

# 2) partizione del sintetico in 9 time step (due varianti)
uv run python src/build_synthetic_snapshots.py

# 3) transfer: addestra sul reale e genera la rete sintetica
uv run python src/transfer_to_synthetic.py generated_snapshots minilm matched
uv run python src/transfer_to_synthetic.py generated_snapshots_growth minilm matched

# 4) analisi (stessa cosa per reale_snapshots e per le due cartelle generate)
uv run python src/analyze_structure.py real_snapshots
uv run python src/analyze_temporal.py  real_snapshots
uv run python src/analyze_content.py   real_snapshots
uv run python src/analyze_profiles.py  generated_snapshots
uv run python src/analyze_profiles.py  generated_snapshots_growth

# 5) figure di confronto
uv run python src/compare_folders.py
```

I numeri finiscono in `results/*.csv`, le figure in `results/figures/`, le reti generate (post + `social_network.edg`) nelle cartelle `Esame Angelantonio Fedele Murolo/generated_snapshots*/`.
