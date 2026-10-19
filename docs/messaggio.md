Ciao Francesco,

scusa il ritardo, tra il rinvio della seduta di laurea e la tesi sono stato preso al 100%. Ti riassumo dove siamo, in breve.

La pipeline è quella finale, il modello per eventi di arrivo. Tutti i numeri di questo messaggio sono di quella; le strade provate prima le cito solo tra le cose che non hanno funzionato.

Contenuto. Tolto node2vec, le rappresentazioni iniziali dei nodi vengono solo dal testo. Ho confrontato sei encoder sul reale e il migliore è all-MiniLM-L6-v2. Ogni post è codificato da solo e poi i vettori sono mediati per utente: così l'AUC del protocollo passa da 0,534 a 0,549. Il modello temporale è tNodeEmbed con queste rappresentazioni, Procrustes a catena, LSTM e testa di coppia [h_u, h_v, h_u*h_v].

[allega: encoder_comparison.png]

Valutazione. Nel reale ogni arco nuovo coinvolge almeno un utente che entra, quindi valuto sull'intera transizione e con due tipi di negativi: casuali e appaiati (stesso mittente, arrivo e grado simili). Con i casuali l'AUC è 0,99 in validation e 0,95-0,96 nei due test; con gli appaiati scende a circa 0,5. Uso gli appaiati, che sono la prova più dura.

Generazione. Nessuna soglia e nessuna regola: fitto sul reale e genero passo per passo campionando. L'unità non è la coppia ma l'evento di arrivo, cioè l'utente che entra. Nel reale un arrivo porta da 35 a 111 archi (massimo 1.202) e il 40% è ricambiato subito: l'hub nasce lì. Ho diviso il sintetico in 9 passi seguendo la curva di arrivo e la forma dell'attività del reale. Ho provato anche "tutti presenti dal primo passo": le due reti sono quasi equivalenti (distanza dal reale 3,79 contro 4,39), quindi uso la prima e tengo la seconda come controllo.

[allega: real_vs_synthetic_table.png]

| Metric (last snapshot) | Real | Synthetic |
|---|---|---|
| Edges | 824,045 | 280,298 |
| Mean out-degree | 32.05 | 28.30 |
| Maximum out-degree | 1,290 | 324 |
| Reciprocity | 0.418 | 0.304 |
| Transitivity | 0.034 | 0.0209 |
| Diameter | 6 | 6 |

Il grado massimo della rete finale è 324: il 95% di quello che il reale prevede alla stessa dimensione (340, stimato sui suoi nove snapshot). A pari dimensione restano due differenze: la coda è più ripida (-3.33 contro -2.35 nel decile superiore) e la coesione è più bassa. Sono i due punti aperti.

[allega: out_degree_distribution.png]

Contenuto e legami. Tre reti generate con testo vero, testi scambiati fra utenti e nessun testo hanno proprietà relazionali indistinguibili: il testo non sposta la rete.

[allega: content_control.png]

L'omofilia 2,13 del reale è in gran parte struttura: contro non-archi appaiati per grado scende a 1,51, e a 1,29 con grado uguale entro il 10%.

[allega: homophily_controls.png]

Il testo però riconosce l'utente: sugli utenti sintetici, dal solo contenuto predico profession (AUC 0,867), political_leaning (0,827) e age_interval (0,690); aggiungere la rete non migliora.

[allega: auc_labels_and_negatives.png]

Cosa non ha funzionato. Ho provato prima altre strade - regole a conteggio deciso a mano, statistiche calibrate sul reale, inferenza con soglia, generazione strutturale - e le ho scartate: o imponevano reciprocità e coesione, o le perdevano, e nessuna riproduce il meccanismo per cui gli archi nuovi arrivano in blocco con l'utente. Poi:

- fine-tuning dell'encoder: peggiora (0,519 contro 0,534 su MiniLM, 0,523 contro 0,530 su BGE-base);

[allega: finetuning_vs_base.png]

- feature dei nodi addestrabili: peggiora (0,512 contro 0,534);
- "post come arco": il testo del singolo post non separa le coppie collegate (AUC circa 0,52, cioè caso);
- negativi misti nel fit: peggiorano la rete;
- chiusura triadica dentro l'evento: misurata in modo fine, non cambia nulla.

Consegna. La rete finale è in 9 file .edg, uno per passo, e anche nei 4 file nella forma dei batch che mi hai dato.

Ci sentiamo nei prossimi giorni, anche in chiamata. Se ti interessa, possiamo estendere la previsione del profilo alle altre label del sintetico.

A presto,
Angelantonio
