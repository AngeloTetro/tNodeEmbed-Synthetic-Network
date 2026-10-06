# tNodeEmbed — trascrizione sintetica del post Medium

**Autore**: Uriel Singer — 14 agosto 2019, 5 minuti di lettura.
**URL**: https://medium.com/@urielsinger/tnodeembed-node-embedding-over-temporal-graphs-b7bcbf59938f

> Trascrizione sintetica e fedele del post (in italiano). Il post è la divulgazione ufficiale dell'autore del metodo e della repo; le citazioni tra virgolette sono testuali.

---

## Embeddings nel dominio dei grafi

Esistono diversi modi per embeddare nel dominio dei grafi: si può embeddare **il grafo intero**, **ogni nodo**, o **ogni arco** (un modo naive è fondere gli embedding dei due nodi). Tutti questi embedding si usano poi per rispondere a domande di machine learning.

## Perché i grafi temporali contano

Esempio dell'autore: Eve diventa amica di Alice e Bob **lo stesso giorno**, mentre è amica di Chris **da un anno**. Chi ha più probabilità di diventare amico di Alice: Bob o Chris? In un grafo statico non c'è alcuna differenza (4 nodi, 3 archi). In un grafo temporale l'informazione su **quando** è avvenuto ogni arco risolve la domanda: Bob è più probabile di Chris.

Definizione: il grafo al tempo `t` è il grafo che contiene **tutti gli archi comparsi fino a `t`** (cumulativo).

## tNodeEmbed: i 3 passi

### 1. Initialization

> "The architecture presented in this paper can use any of the known approaches for node embedding over static graphs. Specifically, we opted to work with node2vec, which achieved state-of-the-art performance on various benchmarks."

node2vec descritto come una riduzione dal dominio grafo al dominio testo: ogni nodo è una "parola", le "frasi" sono generate campionando molte random walk, e le frasi passano a **word2vec**; gli embedding di parole tornano nel dominio grafo come embedding dei nodi.

node2vec **inizializza `Q_t`**: si calcolano gli embedding per tutti i `T` grafi `G_{t1} … G_{tT}`. L'esito è un vettore **`d × T` per nodo** (nel post; il paper scrive `T × d` — stessa matrice, orientamento trasposto), dove ogni colonna/riga è un embedding statico a un time step.

### 2. Temporal Node Embedding Alignment

Per osservare come un nodo si muove nello spazio di embedding fra time step, gli embedding vanno **allineati**: si applica **Orthogonal Procrustes** iterativamente a ogni coppia di time step consecutivi.

Assunzione dichiarata: i due spazi sono abbastanza vicini da essere allineati da una matrice di rotazione, ed è ragionevole perché i due step consecutivi **"differiscono solo negli archi aggiunti"** fra i due.

Nota curiosa dell'autore: FAIR usa lo stesso approccio per la traduzione automatica non supervisionata.

### 3. Node Embedding over Time

Nel passaggio finale il vettore `d × T` di ogni nodo viene **fuso in un vettore `d` tramite una LSTM**, collegata a un layer aggiuntivo che predice il task specifico.

## Risultati

### 1. Temporal Link Prediction

Il task si costruisce scegliendo un **pivot time**: tutti gli archi noti fino al pivot → training; gli archi successivi → test.

Risultato principale (tabella del post): un'osservazione interessante è che **più basso il clustering coefficient (CC), maggiore il divario fra tNodeEmbed e la sua baseline statica node2vec**.

Interpretazione dell'autore: più "denso" il grafo, meno informativa è l'informazione temporale per predire i link futuri — perché gli archi tendono ad attaccarsi ai nodi con grado più alto, che nei grafi densi hanno gradi estremamente alti rispetto agli altri.

### 2. Multi-label Node Classification

Su Cora e DBLP: "tNodeEmbed does pretty well" (micro-F1, macro-F1 e AUC a favore di tNodeEmbed su entrambi i dataset).

## Final words e link

- Chiusura dell'autore: "tNodeEmbed nailed it!" — invito a usare il lavoro e continuare la ricerca.
- **Paper**: https://www.ijcai.org/proceedings/2019/0640.pdf
- **Codice**: https://github.com/urielsinger/tNodeEmbed
- Tag: Machine Learning, Graph, Network, Embedding, Time Series Data.

---

## Perché questo post conta per il progetto

Il post conferma in forma divulgativa ciò che serve al nuovo percorso: i **3 passi** (inizializzazione → allineamento → LSTM) sono l'architettura da riprodurre, e il **primo passo è per costruzione sostituibile** con un encoder basato sulle feature dei nodi (la stessa frase "any of the known approaches" compare sia nel paper sia nel post). In più, l'esempio di Eve/Alice/Bob/Chris è il modo più breve di spiegare al dottorando — e a chiunque — perché la dimensione temporale cambia le predizioni.
