# Guida al progetto, in italiano

Questo testo non è un documento da consegnare: serve a te, per capire cosa c'è dentro
`document.pdf`, perché ogni scelta è stata fatta così, e cosa significano le metriche che ci
trovi dentro. È scritto in linguaggio comune, con gli esempi al posto delle definizioni.

---

## 1. Il progetto in cinque righe

Abbiamo due dataset. Uno, **reale**, ha gli utenti, i loro post e i collegamenti fra loro
(chi segue chi), distribuiti su nove istantanee dal 2022 al 2025. L'altro, **sintetico**, ha
10.000 utenti finti con i loro post e i loro attributi, ma **nessun collegamento**: la rete
non c'è.

Il compito è ricostruire quella rete mancante. Non inventando dei numeri a caso, ma imparando
dal dataset reale **come nasce un collegamento**, e applicando quella regola alla popolazione
sintetica. Il risultato è una rete generata che assomiglia statisticamente a quella reale, e un
modello i cui coefficienti si possono leggere e discutere.

Il punto delicato, e la cosa più interessante del lavoro: il testo dei post serve a **capire chi
è l'utente**, ma non a decidere **con chi si collega**. Questa distinzione attraversa tutto il
documento.

---

## 2. Come è fatto `document.pdf`

Il documento segue lo schema **CRISP-DM**, che è il modo standard di raccontare un progetto di
data mining in sei fasi. Non è una scelta estetica: è lo schema che i tre esempi del professore
usano, e ha il vantaggio di raccontare *il percorso*, non solo il risultato.

### 1. Business understanding
Perché il problema esiste davvero. Una piattaforma che ha solo i testi dei suoi utenti non può
calcolare nulla di sociale — non sa chi influenza chi, non sa quali sono le community, non può
vendere pubblicità mirata — perché tutte queste cose richiedono la rete. Comprare o scaricare la
rete spesso è impossibile. Quindi: la si ricostruisce dal testo.

Nella sottosezione **Data mining goals** ci sono quattro obiettivi numerati G1–G4. Servono a una
cosa sola: poter dire alla fine «questo l'ho centrato, questo no». G3, per esempio, è
*quantificare quanto contribuisce il testo*, ed è quello che poi diventa la parte più
interessante dei risultati.

Subito dopo c'è **Positioning within the literature**: dove sta il nostro lavoro rispetto a
tNodeEmbed, a node2vec e agli altri. Non è decorazione: è la sezione che giustifica perché abbiamo
tolto node2vec (non sa rappresentare un utente che entra) e che collega ogni nostra scelta a una
citazione. Prima non c'era, e la bibliografia finale aveva dodici voci che nessuno citava.

### 2. Data understanding
Cosa c'è nei dati. Qui c'è la tabella con i **due dataset a confronto** e la tabella con le
**undici etichette** del sintetico (nazionalità, genere, etnia, religione, orientamento politico,
fascia d'età, stato di origine, interessi, professione, username, biografia), con quanti valori
distinti ha ciascuna. La riga che conta è l'ultima della prima tabella: **l'unica colonna in
comune fra i due dataset è `content`**, cioè il testo del post. È per questo che tutto il lavoro
passa dal testo.

Due fatti del reale guidano il progetto e vale la pena averli in testa: primo, **nessun arco
collega due utenti già presenti** — ogni collegamento nasce quando entra qualcuno di nuovo;
secondo, la rete è estremamente diseguale (grado medio 32, massimo 1.290).

### 3. Data preparation
Come i dati sono stati resi utilizzabili. Tre cose:
- **pulizia**: si tolgono gli auto-anelli, si normalizzano gli identificativi, le istantanee sono
  cumulative e ordinate, una "transizione" è la differenza fra due istantanee consecutive;
- **rappresentazione**: ogni post viene trasformato in un vettore numerico da un encoder, poi i
  vettori dei post di uno stesso utente vengono mediati. Sono stati confrontati sei encoder
  (tabella 2);
- **partizione temporale**: il sintetico non ha date, quindi gli si impone un tempo. Gli utenti
  entrano seguendo la **curva di arrivo misurata sul reale** (il 41,8% è già presente al primo
  passo, poi 14,2%, 11,0%...), e il numero di post per utente segue la distribuzione reale.

Qui nascono **due versioni** del sintetico, ed è la scelta che poi ritorna in tutto il documento:
la partizione **progressiva** (gli utenti entrano uno alla volta) e quella **con tutti presenti
dal primo passo**. La progressiva è quella che presentiamo, perché è l'unica che assomiglia a
una rete che nasce davvero.

### 4. Modeling
Il cuore tecnico. Quattro pezzi, in ordine:
1. **la rappresentazione temporale** (tNodeEmbed): un modello che legge la sequenza delle nove
   istantanee di ogni utente e ne produce un vettore. Prima di leggerla, le nove istantanee
   vengono allineate fra loro (Procrustes), perché altrimenti i vettori di istantanee diverse
   vivrebbero in sistemi di coordinate diversi e non sarebbero confrontabili;
2. **l'addestramento**: un modello che, dati due utenti, dice quanto è probabile che siano
   collegati. Si addestra solo sulle transizioni di training (le prime cinque), con la
   validazione sulla sesta e le ultime due intatte fino alla valutazione;
3. **il generatore**: la parte nuova. Non decide coppia per coppia, ma **per eventi di arrivo**:
   quando un utente entra, si porta dietro un certo numero di archi (`k`), e per ognuno si sceglie
   il partner con una regressione logistica a sette variabili;
4. **la valutazione del modello**: cosa il generatore riesce e non riesce a riprodurre.

La tabella 3 sono i **coefficienti** di quella logistica, ed è la tabella da guardare se si vuole
capire il meccanismo: `+1.22` per "il partner è nuovo", `+1.02` per il logaritmo del grado, `+0.91`
per il percentile di grado, e **`+0.05` per il punteggio di contenuto**. Il contenuto pesa un
ordine di grandezza meno della struttura.

### 5. Evaluation
I risultati, con una gerarchia dichiarata all'inizio (sottosezione 5.1): le **metriche primarie**
sono le statistiche di rete; le **secondarie** riguardano la natura del legame (omofilia del
contenuto); l'**AUC** è solo diagnostica e solo sul reale; e l'AUC come misura di qualità di una
rete generata è dichiarata **non valida** — su un dataset senza archi veri non c'è niente da
prevedere.

Poi, nell'ordine: la tabella grande con reale/progressiva/tutti presenti; la composizione dei
nuovi archi; la spiegazione del perché la partizione progressiva e quanto costa; i cinque
approcci provati prima di quello definitivo e perché sono stati scartati; il contenuto
(omofilia e predizione degli attributi); i tentativi che non hanno funzionato; i limiti; la
riproducibilità.

L'omofilia ha una tabella e, sotto, un grafico a barre: le sei barre sono gli stessi archi
confrontati con sei gruppi di controllo sempre più simili strutturalmente. La barra rossa è la
misura che si legge di solito (2,13), quella in fondo è il controllo più severo (1,29). Il
grafico serve a far vedere a occhio che il numero scende quando il confronto diventa onesto.

### 6. Conclusions
Cosa si è imparato, in tre paragrafi: il meccanismo è relazionale e sequenziale; il testo
identifica l'utente ma non decide la relazione; restano aperti tre problemi precisi (la coda dei
gradi, la dipendenza fra i partner scelti nello stesso evento, una rappresentazione del contenuto
che agisca sulla coppia e non sull'individuo).

---

## 3. Le scelte, e perché

**Perché tNodeEmbed senza node2vec.** node2vec impara la posizione di un utente guardando i suoi
archi: se un utente non ha ancora archi — e nel sintetico nessuno ne ha — non ha niente da
guardare. Nel nostro reale, inoltre, non esiste un solo arco fra due utenti già presenti, quindi
il momento in cui un utente entra è esattamente il momento in cui riceve i suoi archi. Un modello
che si inizializza sui collegamenti è cieco proprio dove serve vedere. L'inizializzazione quindi
viene dal testo, e la parte temporale (LSTM + Procrustes) fa il resto.

**Perché l'evento di arrivo e non la coppia.** Perché è così che funziona il reale: gli archi
nascono a grappoli, quando entra qualcuno. Generare coppie indipendenti produce un grafo fatto di
coppie isolate: la reciprocità sparisce (0,0031 contro 0,4177) perché due archi indipendenti non
si "ricambiano" mai. Con l'evento di arrivo la reciprocità nasce da sola, e così il grado medio e
i poli.

**Perché la partizione progressiva.** Perché una rete che nasce ha degli arrivi, e perché nel
reale il 58% degli utenti finali entra durante il periodo osservato. La versione con tutti
presenti è tenuta come confronto, non come alternativa: sulle sette metriche confrontabili le due
sono equivalenti (distanza 3,84 contro 3,90), quindi la scelta non è metrica ma di senso.

**Perché il contenuto resta fuori dalla decisione.** Non per partito preso: perché i coefficienti
dicono che pesa +0,05 su +1,22, e perché quando lo si toglie (testo azzerato o attribuito a un
altro utente) la rete generata non cambia in modo misurabile. Il contenuto però **identifica**
l'utente: dagli stessi post si predice l'orientamento politico con AUC 0,827 e la professione con
0,867. Sono due ruoli diversi, e vanno tenuti distinti.

**Perché l'omofilia del contenuto sembra contraddire tutto questo.** Perché è confusa con il grado:
gli utenti molto collegati hanno più testi e testi più simili fra loro, quindi il 2,13 iniziale
scende a 1,29–1,55 quando si confrontano archi e non-archi **con grado simile**. Quel residuo è
piccolo (AUC circa 0,58).

---

## 4. Glossario delle metriche

**Grado (out-degree).** Quanti collegamenti escono da un utente. Grado medio 32 nel reale
significa che un utente tipo segue 32 persone. Grado massimo 1.290 significa che il più seguito
ne segue 1.290: la distribuzione è molto sbilanciata.

**Densità.** Quanti archi ci sono rispetto a tutti quelli possibili. Nel reale 0,00125: una rete
grande e molto sparsa. Non è confrontabile fra reti di dimensioni diverse senza normalizzare.

**CCDF e pendenza di coda.** La CCDF è la curva che dice "quanti utenti hanno grado almeno X".
Su un grafico log-log, se la coda è una retta, la rete è "senza scala": pochi poli enormi e una
massa di utenti piccoli. La pendenza misura quanto è ripida: **−2,35 nel reale, −3,98 nella
progressiva** — più ripida significa poli meno estremi, cioè una rete più piatta.

**Reciprocità.** La frazione di archi ricambiati: se A segue B, quanto spesso B segue A. Reale
0,4177 (il 42%), progressiva 0,3224.

**Transitività.** La probabilità che, se A è collegato a B e B a C, allora A sia collegato a C.
Misura quanto la rete è "chiusa" in triangoli. Reale 0,0337, progressiva 0,0212.

**Clustering medio.** Come la transitività, ma mediata su ogni nodo invece che sui triangoli
possibili: dice quanto sono densi i vicinati dei singoli utenti. Reale 0,0804, progressiva 0,0420.

**Modularità.** Quanto la rete si divide in gruppi con molti legami interni e pochi esterni
(community). Reale 0,2751, progressiva 0,1074: la rete generata ha community meno nette. Attenzione:
dipende dall'algoritmo di partizione usato.

**Diametro.** Il massimo numero di passi necessario per andare da un utente a un altro (calcolato
in modo approssimato). Entrambe le reti: 6.

**Omofilia del contenuto.** Quanto sono più simili fra loro i testi di due utenti collegati,
rispetto a due utenti presi a caso. 1,00 = nessuna tendenza, 2,13 = due volte più simili. Il
problema è che "presi a caso" non è un confronto onesto: chi ha molto testo è più simile a
chiunque altro. Da qui i controlli (1,29–1,55).

**AUC.** Area sotto la curva ROC: la probabilità che il modello dia un punteggio più alto a una
coppia collegata che a una coppia non collegata. 0,5 = caso, 1,0 = perfetto. Attenzione: **non è**
una misura di qualità di una rete generata — serve a scegliere l'encoder e a fittare il modello
sul reale.

**Distanza relativa dal reale.** La formula usata in tutto il documento:
`|valore_generato − valore_reale| / |valore_reale|`, sommata su sette metriche confrontabili
(densità, grado medio, grado massimo, reciprocità, modularità, transitività, clustering). Somma
3,84 per la progressiva, 3,90 per quella con tutti presenti. Più bassa = più vicina; ma il numero
non ha un'unità, serve solo a confrontare fra loro reti misurate allo stesso modo.

**Legge di scala.** Il numero di archi per utente che entra non è costante: dipende da quanti
utenti ci sono già. Sul reale si misura una relazione (in forma di potenza) fra popolazione e
`k`; la si applica al sintetico, che ha 10.000 utenti invece di 25.714. È un'estrapolazione, e il
documento lo dice: la legge è fittata su popolazioni fra 10.755 e 21.082, e al primo passo del
sintetico viene applicata a circa 4.100.

---

## 5. I numeri principali, e come leggerli

| cosa | reale | progressiva | tutti presenti |
|---|---:|---:|---:|
| utenti con almeno un arco | 25.714 | 9.897 | 9.901 |
| archi | 824.045 | 271.646 | 280.298 |
| grado medio | 32,05 | 27,45 | 28,31 |
| grado massimo | 1.290 | 274 | 328 |
| reciprocità | 0,4177 | 0,3224 | 0,3035 |
| transitività | 0,0337 | 0,0212 | 0,0209 |
| clustering | 0,0804 | 0,0420 | 0,0435 |
| modularità | 0,2751 | 0,1074 | 0,1030 |
| diametro | 6 | 6 | 6 |
| distanza dal reale | — | **3,84** | 3,90 |

Come si leggono: la rete generata ha **lo stesso ordine di grandezza** del reale su tutto, e
riproduce esattamente il diametro. Perde sui poli (274 contro 1.290) e sulla coesione
(transitività e clustering alla metà circa). Gli archi sono 271.646 contro gli 824.045 del reale,
ma i nodi sono 9.897 contro 25.714: alla stessa scala il volume sarebbe 317.165, quindi il
generatore produce l'**86%** del volume atteso, e la coda dei gradi l'81% di quanto predice la
legge di scala (97% nella versione con tutti presenti).

Sul contenuto: l'omofilia reale 2,13 scende a 1,29–1,55 con i controlli strutturali; gli
attributi si predicono bene dal testo (0,827 / 0,867 / 0,690) e il grafo non aggiunge nulla
(0,826 / 0,865 / 0,683), mentre gli embedding temporali sono peggio (0,619 / 0,587 / 0,514).

---

## 6. Cosa resta aperto

Tre cose precise, che sono anche le tre cose su cui si può continuare:

1. **la coda dei gradi**, più ripida del reale a parità di dimensione: la rete generata non
   produce poli abbastanza estremi. Probabilmente dipende dal fatto che `k` è una media e non una
   distribuzione;
2. **la dipendenza fra i partner scelti nello stesso evento**: oggi ogni partner è scelto
   indipendentemente dagli altri dello stesso evento, e questo abbassa la transitività. Un modello
   che li scelga in modo correlato chiuderebbe la parte triadica;
3. **una rappresentazione del contenuto che agisca sulla coppia** e non sul singolo utente: oggi
   il contenuto dice chi è l'utente, non se due utenti stanno bene insieme.

---

## 7. Cosa tenere presente se se ne parla

- Le cifre vengono da **una sola esecuzione**: niente intervalli di confidenza. I pesi della rete
  addestrata variano al terzo decimale fra processi (stessi 271.646 archi, o 271.655 in una
  ripetizione: 9 archi su 271 mila).
- La scelta dell'encoder fra MiniLM, GTE-base e BGE-base è **un pareggio**, non una vittoria: il
  margine (0,002) è sotto la variabilità fra processi (circa 0,005).
- Il confronto fra i due sintetici **non elegge un vincitore**: 3,84 contro 3,90, e invertendo la
  normalizzazione si inverte l'ordine. La progressiva è la rete da presentare perché è quella che
  ha senso, non perché vince.
- Il primo passo della progressiva è più vuoto del reale (15,7% di nodi collegati contro 41,8%),
  ma quel 41,8% è uno **stock preesistente**, non un passo di nascita: non è un errore da
  correggere, e forzarlo ripeterebbe l'errore di un tentativo precedente.
- Un dataset reale e uno sintetico, entrambi forniti per il progetto: non sappiamo se il
  meccanismo vale per altre piattaforme.
