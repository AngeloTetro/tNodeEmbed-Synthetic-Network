# tNodeEmbed — trascrizione sintetica del paper

**Paper**: *Node Embedding over Temporal Graphs* — Uriel Singer, Ido Guy, Kira Radinsky (IJCAI 2019, pp. 4605–4612).
**Versione completa**: arXiv 1903.08889. **Codice**: https://github.com/urielsinger/tNodeEmbed

> Trascrizione sintetica e fedele del contenuto del paper, in italiano. Le citazioni tra virgolette sono testuali. Le equazioni seguono la notazione del paper.

---

## 1. Il problema

I metodi classici di feature learning su grafi (DeepWalk, node2vec) apprendono rappresentazioni **statiche**: catturano uno snapshot del grafo e ignorano *quando* gli archi compaiono. Un grafo temporale contiene informazione che un grafo statico non cattura.

Analogia degli autori: se node2vec imita gli embedding di **parole**, tNodeEmbed imita gli embedding di **frasi** — ogni "parola" della frase è la rappresentazione del nodo a un istante temporale.

Obiettivo dichiarato: un algoritmo semi-supervisionato che ottimizza due obiettivi insieme — (1) preservare i vicinato statici in uno spazio `d`-dimensionale, (2) preservare la dinamica della rete — con una loss del task a valle (link prediction o node classification).

---

## 2. Definizioni

- **Arco temporale** `(u, v)_t`: arco diretto da `u` a `v` al tempo `t`.
- **Grafo temporale** `G_t = (V_t, E_t)`: il grafo di **tutti gli archi comparsi fino al tempo `t`** — è quindi **cumulativo**.
- **Evoluzione**: la sequenza di snapshot `G_{t1}, …, G_{tT}` con `t1 < … < tT`.
- **Obiettivo**: per ogni nodo `v`, trovare `f_T(v) = F_T(v, G_1, …, G_T)` che minimizza la loss del task.

---

## 3. Il framework

### Ricorrenza e loss

Equazione (1) — l'embedding evolve come stato di una ricorrenza:

```
f_{t+1}(v) = σ( A · f_t(v) + B · R_t · Q_t · v ),    f_0(v) = 0
```

- `v` è un vettore **one-hot** del nodo;
- `A`, `B`, `R_t`, `Q_t` sono matrici **apprese durante il training**;
- `Q_t` si interpreta come la matrice delle rappresentazioni statiche al tempo `t`.

Equazione (2) — ottimizzazione **congiunta end-to-end**:

```
L = min_{A, B, Q_1…Q_T, R_2…R_T}  L_task
```

ottimizzata con **Adam** su tutti i parametri che definiscono `f_T` (quindi anche gli embedding statici e le rotazioni vengono raffinati per il task).

Loss per task:
- **node classification**: `L_task = − Σ_{v∈V} log Pr(class(v) | f_T(v))` (cross-entropy categoriale);
- **link prediction**: `L_task = − Σ log Pr((v1, v2) ∈ E_t | g(f_T(v1), f_T(v2)))`, con `g` = **concatenazione** delle due rappresentazioni.

### §3.1 — Inizializzazione di `Q_t` con embedding statici ⭐ (la sezione chiave)

Equazione (3) — l'inizializzazione impara a preservare i vicinati campionati:

```
L_init = − Σ_t Σ_{v_t ∈ V_t}  log Pr( N(v_t) | Q_t · v_t )
```

dove `N(v_t)` è un vicinato generato con una strategia di network sampling.

Citazione testuale, il punto su cui si fonda la variante feature-based:

> "**The architecture presented in this paper can use *any* of the known approaches for node embedding over static graphs.** Specifically, we opted to work with node2vec, which achieved state-of-the-art performance on various benchmarks."

Cosa fa nella pratica:
- calcola gli embedding node2vec per **tutti i `T` grafi cumulativi** `G_{t1}, …, G_{tT}`;
- l'esito è un vettore **`T × d` per nodo** (`T` = numero di time step, `d` = dimensione);
- questi valori sono usati come **inizializzazione di `Q_t`**, che viene poi **raffinata end-to-end** sulla loss del task.

**Conseguenza architetturale**: node2vec è una **scelta pratica** (scalabilità, stato dell'arte), non un requisito. L'encoder statico è un componente sostituibile: qualsiasi tecnica che produca una rappresentazione `T × d` per nodo può inizializzare `Q_t` — incluse rappresentazioni derivate da **feature dei nodi**.

### §3.2 — Allineamento temporale (Orthogonal Procrustes)

- **Problema**: i training di word2vec/node2vec non garantiscono coerenza degli assi: due grafi identici a tempi diversi possono produrre embedding ruotati.
- **Assunzione**: fra due step consecutivi la maggior parte dei nodi cambia poco (i grafi "differiscono solo negli archi aggiunti"), quindi basta una **rotazione**.
- **Procedura**: `R_{t+1} = argmin_{R^T R = I} ‖R·Q_{t+1} − Q_t‖`; poi `Q'_t = R·Q_t`. Applicata **iterativamente a catena**: `Q_{t2}` allineato a `Q_{t1}`, poi `Q_{t3}` a `Q_{t2}`, e così via.
- L'allineamento è calcolato **solo sui nodi presenti a entrambi i tempi**; i nodi nuovi vengono trasformati con la stessa `R`.
- Dentro la loss end-to-end, l'allineamento è vincolato dall'equazione (4):

```
Σ_t ‖R_{t+1}·Q_{t+1} − Q_t‖ + λ·‖R_{t+1}^T · R_{t+1} − I‖ ,    R_1 = I
```

il primo termine impone similarità fra step consecutivi, il secondo forza `R` a essere una rotazione (`λ` regolarizzatore, `λ = 1` negli esperimenti).

### §3.3 — LSTM: l'embedding temporale finale

- Ogni nodo ora ha la matrice `X^(v) ∈ R^{T×d}`: la **storia dei suoi embedding** nel tempo.
- Una **LSTM** riduce la sequenza a un unico vettore: l'ultimo stato `h_T ∈ R^d` è l'embedding temporale finale del nodo.
- **Link prediction**: l'arco `e = (v1, v2)` è rappresentato dalla **concatenazione** dei due embedding temporali (`2d`) → layer fully-connected di dimensione `d` → layer finale. Nel paper è descritto con softmax; nel codice la testa è sigmoide con `binary_crossentropy`, la **stessa LSTM è condivisa** fra i due nodi della coppia (pesi condivisi), activation `relu`.
- **Node classification**: LSTM → `Dense(num_classes)` → softmax, loss `categorical_crossentropy`.

---

## 4. Setup sperimentale

### Dataset (tabella 1 del paper)

| Dataset | Peso | Direzione | Nodi | Archi | Diametro | Time step di training | Granularità |
|---|---|---|---:|---:|---:|---:|---|
| arXiv hep-ph | sì | no | 16.959 | 2.322.259 | 9 | 83 | mensile (03/1992–12/1999) |
| Facebook friendships | no | no | 63.731 | 817.035 | 15 | 26 | mensile (09/2004–01/2009) |
| Facebook wall posts | sì | sì | 46.952 | 876.993 | 18 | 46 | mensile (09/2006–01/2009) |
| CollegeMsg | sì | sì | 1.899 | 59.835 | 8 | 69 | giornaliera (04/2004–10/2004) |
| PPI | no | no | 16.458 | 144.033 | 10 | 37 | annuale (1970–2015) |
| Slashdot | sì | sì | 51.083 | 140.778 | 17 | 12 | mensile (01/2004–09/2006) |
| Cora | no | sì | 12.588 | 47.675 | 20 | 39 | annuale (1900–1999) |
| DBLP | sì | no | 416.204 | 1.436.225 | 23 | 9 | annuale (1990–1998) |

Note: multi-archi fra la stessa coppia collassati in un singolo arco **pesato**. PPI: timestamp = data di scoperta dell'interazione (preprocessata come data di prima pubblicazione), da HINTdb.

### Protocollo — Temporal Link Prediction

- Pivot scelto in modo che ~**80%** degli archi (o la porzione più vicina possibile) abbia timestamp ≤ pivot.
- Archi `≤ pivot` → positivi di **training**; archi `> pivot` → positivi di **test**.
- Negativi di training: numero identico ai positivi, coppie **non collegate al pivot**, campionate uniformemente.
- Negativi di test: numero identico ai positivi, coppie **non collegate da alcun arco** (a nessun tempo).
- **Restrizione**: *"We focused our task on predicting ties between existing nodes and therefore restricted edges in the test set to be formed only between existing nodes in the training set."* — e la scelta del pivot **tiene conto** di questa restrizione.
- Metrica: **AUC**.

### Protocollo — Multi-label Node Classification

- Split **casuale** 80/20 sui nodi (non temporale).
- Metriche: micro-F1, macro-F1, AUC.
- Dataset: Cora (10 classi tematiche), DBLP (15 campi di ricerca).

### Baseline

node2vec (statico), TMF, TMFntv (TMF + node2vec sul grafo collassato), TFNM, CTDNE, HTNE, DynamicTriad. Tutte usano gli embedding dell'ultimo time step.

---

## 5. Risultati

### Temporal Link Prediction (tabella 2) — AUC

| Dataset | tNodeEmbed | node2vec | TMF | TMFntv | TFNM | CTDNE | HTNE | DynTri | CC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arXiv hep-ph | 0,951 | 0,951 | 0,948 | 0,948 | 0,908 | 0,905 | 0,851 | 0,783 | 0,291 |
| Facebook friendships | 0,939 | 0,938 | 0,886 | 0,925 | 0,814 | 0,757 | 0,724 | 0,535 | 0,148 |
| Facebook wall posts | 0,917 | 0,902 | 0,718 | 0,900 | 0,720 | 0,827 | 0,784 | 0,643 | 0,078 |
| CollegeMsg | 0,841 | 0,823 | 0,809 | 0,819 | 0,654 | 0,841 | 0,838 | 0,630 | 0,036 |
| PPI | 0,828 | 0,799 | 0,753 | 0,798 | 0,712 | 0,800 | 0,782 | 0,761 | 0,017 |
| Slashdot | 0,913 | 0,777 | 0,896 | 0,793 | 0,661 | 0,894 | 0,886 | 0,765 | 0,010 |

Lettura del paper:
- tNodeEmbed **vince** su Facebook wall posts, PPI e Slashdot, ed è **comparabile** sugli altri tre; è il metodo con le performance **più consistenti** (il più alto su tutti e sei).
- Risultato chiave: **più basso il clustering coefficient (CC), maggiore il divario** da node2vec (Slashdot, CC = 0,010: 0,913 vs 0,777). Motivo ipotizzato: nei grafi più densi gli archi si attaccano ai nodi ad alto grado (modello di preferential attachment), quindi la dinamica aggiunge meno informazione.

### Multi-label Node Classification (tabella 3)

| Dataset | tNodeEmbed micro-F1 | macro-F1 | AUC | node2vec micro-F1 | macro-F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Cora (CC = 0,275) | 0,668 | 0,513 | 0,925 | 0,547 | 0,284 | 0,862 |
| DBLP (CC = 0,002) | 0,822 | 0,504 | 0,977 | 0,752 | 0,235 | 0,943 |

tNodeEmbed è il migliore su tutte le metriche per entrambi i dataset (micro-F1 di DBLP pari a TFNM, ma il macro-F1 di TFNM crolla a 0,060: collasso su una classe).

### Grafi sintetici con diverse distribuzioni di grado (tabella 4)

Grafi random con `n = 1000`, `m = 100.000`, `T = 50`, pivot a `t = 45`; grado non power-law:

| Distribuzione | tNodeEmbed | node2vec | CTDNE | CC |
|---|---:|---:|---:|---:|
| Lineare | 0,67 | 0,53 | 0,65 | 0,22 |
| Logaritmica | 0,74 | 0,56 | 0,73 | 0,26 |
| Sinusoidale | 0,79 | 0,64 | 0,78 | 0,31 |
| Esponenziale | 0,85 | 0,79 | 0,83 | 0,36 |

tNodeEmbed batte node2vec su tutte ed è comparabile a CTDNE; di nuovo, il vantaggio cresce al diminuire del CC.

### Effetto dell'allineamento (tabella 5) — AUC

| Dataset | con allineamento | senza allineamento | CC |
|---|---:|---:|---:|
| arXiv hep-ph | 0,951 | 0,950 | 0,29 |
| Facebook friendships | 0,939 | 0,935 | 0,15 |
| Facebook wall posts | 0,917 | 0,916 | 0,08 |
| CollegeMsg | 0,841 | 0,825 | 0,04 |
| PPI | 0,828 | 0,822 | 0,02 |
| Slashdot | 0,913 | 0,860 | 0,01 |

Rimuovere l'allineamento peggiora **3 dataset su 6** ed è comparabile sugli altri; il guadagno dell'allineamento è **maggiore a CC basso** (Slashdot: 0,913 → 0,860). Interpretazione: nei grafi meno coesi la variabilità fra training diversi è maggiore, quindi l'allineamento toglie più rumore. Sui task di node classification il trend è lo stesso (tabella 6: Cora micro-F1 0,668 vs 0,644; DBLP 0,822 vs 0,785).

---

## 6. Conclusioni del paper

- Il framework generalizza l'embedding di grafo catturandone la dinamica; **può sfruttare qualunque tecnica di embedding statico** e impararne la versione temporale alterandola.
- Per imparare con successo la dinamica serve una **storia minima** del nodo; le previsioni su grafi con CC alto non beneficiano in modo significativo della dinamica.
- Future work: apprendere usando la miglior risoluzione temporale possibile, senza fermarsi a snapshot discreti.

---

## Perché questa trascrizione conta per il progetto

La **§3.1** è il fondamento della variante richiesta dal dottorando: l'inizializzazione di `Q_t` è un componente **sostituibile** ("any of the known approaches"), quindi node2vec può essere rimpiazzato da rappresentazioni derivate dalle **feature dei nodi**. Questo (a) abilita la rappresentazione di nodi mai visti (non trasduttivi), (b) supera il vincolo del protocollo che restringe il test ai nodi del training, (c) elimina il rumore stocastico tipico dei training di node2vec. Nella nostra pipeline le feature **sostituiscono** `Q_t` (non c'è una matrice addestrabile inizializzata dalle feature: le feature sono l'input del modello); restano dal paper la **LSTM** e la **testa di link prediction end-to-end**, mentre l'**allineamento Procrustes** è applicato come passo di pre-processing a forma chiusa (come del resto fa anche il codice di riferimento, che allena con la sola cross-entropy binaria).
