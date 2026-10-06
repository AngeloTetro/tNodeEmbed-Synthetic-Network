Dottorando:

Ciao, ho caricato su onedrive la cartella con i dati necessati per il tuo lavoro d'esame:
Esame Angelantonio Fedele Murolo
 
La cartella "real_snapshots" contiene i post del dataset reale suddivisi in snapshot. In ogni snapshot trovi questi fil:
social_network.edg: social network tra gli utenti visti fino a quello snapshot, con tutti i precedenti
posts_current_snapshot.tsv: dataframe contenente i solo i post pubblicati nell'intervallo temporale a cui fa irferimento lo snapshot in considerazione
posts_incremental.tsv: file contenente i post pubblicati in questo snapshot, più tutti quelli pubblicati negli snapshot precedenti. Questo file potrebbe esserti utile nel caso in cui il metodo che usi lo richieda
synthetic_snapshots contiene la difvisione dei post sintetici in nquattro snapshot, in maniera casuale. Puoi decidere di suddividerli in un numero maggiore di snapshot


io:

Buonasera Francesco, perdona l'orario, ti scrivo per darti un aggiornamento sui progressi del progetto di Big Data. Leggi pure tutto con calma quando hai disponibilità. Di seguito ti riassumo i risultati delle prime analisi e alcune questioni metodologiche su cui vorrei confrontarmi con te.
 
Partendo dall'implementazione di TNodeEmbed, ho notato innanzitutto alcune differenze strutturali: a differenza del loro dataset (reti non dirette di proteine), noi abbiamo archi diretti, ma questo fortunatamente non rappresenta un problema. L'ostacolo principale riguarda invece il task di valutazione. TNodeEmbed testa la creazione di nuovi archi solo tra coppie di nodi già presenti nel training, ma nel nostro dataset questo non succede mai. Ogni nuovo arco coinvolge sempre almeno un nodo nuovo (nuovo-nuovo, nuovo-vecchio, o vecchio-nuovo); non c'è mai un "ricablaggio" tra due utenti già esistenti nello storico. Di conseguenza, applicando il protocollo standard, otterremmo esattamente 0 archi di test.
 
Per aggirare il problema, ho riadattato il task a uno scenario di cold start: quando si presenta un nuovo utente nel test/validation, gli forniamo solo metà delle sue connessioni storiche e il task consiste nel predire l'altra metà. Non avendo un embedding nel training, rappresento il nuovo nodo calcolando la media degli embedding dei suoi vicini conosciuti. Lo score fra utente nuovo e candidato è bilineare e i negativi sono coppie mai comparse prima come arco. I risultati di questo approccio mostrano che l'utilizzo dello storico (media della traiettoria) batte leggermente la baseline statica legata all'ultimo embedding. L'introduzione della pendenza, invece, porta il modello in overfitting peggiorando le performance. Inoltre, poiché cambiando il seed si genera molto rumore nello spostamento degli embedding, per stabilizzare l'analisi ho deciso di calcolarli facendo la media su 5 seed differenti.
 
Sulla base di queste analisi (oltre al task ho calcolato metriche come modularità, densità, clustering e variazioni nel tempo), mi chiedo se valga ancora la pena introdurre il confronto con il dataset sintetico. Al momento questo dataset manca degli archi (che potrei comunque generare) e presenta forti squilibri, ad esempio il batch 3 ha solo 2 righe. Nel caso decidessimo di tenerlo, andrebbe allineato al reale impostando un numero analogo di snapshot e una linea temporale.
 
Alla luce di tutto questo, vorrei un tuo parere su tre punti fondamentali. Innanzitutto sull'adattamento del task a "cold start" per poter implementare la logica di TNodeEmbed. In secondo luogo, vorrei valutare con te un'eventuale suddivisione diversa degli snapshot già presenti e la modifica dello split train/val/test (TNodeEmbed usa un pivot time, che concettualmente è simile a quello che facciamo noi a livello di snapshot, ma potremmo doverlo ricalibrare). Infine, vorrei decidere se mantenere o accantonare definitivamente il dataset sintetico viste le problematiche.
 
Non potendo allegare direttamente le immagini a causa della mail esterna, ho caricato una panoramica generale sul nostro dataset reale, le AUC calcolate e le differenze con TNodeEmbed in questa cartella Drive: https://drive.google.com/drive/folders/1Kx2L2c97MXLQc5n7JhiUak5vGKph7r05?usp=sharing
Se hai bisogno di ulteriori chiarimenti, script o grafici specifici fammi sapere (al momento ho la repo un po' incasinata ma recupero tutto facilmente), oppure possiamo sentirci direttamente in chiamata.
 
Resto in attesa di un tuo riscontro e ti auguro buona serata.

Dottorando: 

mi ricordo che di base il modello non funzionava con nodi mai visti perchè utilizza node2vec come encoder per ottenere le feature iniziali dei nodi (sezione 3.1 del paper) visto che nei loro esperimenti i nodi non ne hanno oppure le hanno ma sono molto semplici. Nel nostro caso, le feature dei nodi le abbiamo e quindi quell'encoder non serve. Se non ricordo male ne abbiamo parlato quando ci sentimmo. Credo che il problema che hai descritto sia dovuto a questo.
 
Riguardo la divisione in snapshot del dataset sintetico puoi fare come meglio ritieni, non è obbligatorio lasciare quella che sta. Potresti anche cambiare gli snapshot del dataset reale anche se in quel caso eviterei qualora non fosse strettamente necessario.
 
L'obiettivo è quello di costruire una rete tra gli utenti sintetici, e dovresti riuscirci rimuovendo la parte di n2v. Se anche togliendola questo fosse impossibile, una soluzione potrebbe essere quella di aggiungere gli utenti sintetici alla rete reale come nodi isolati, così il modello li riconoscerà, e potrà dunque predire connessioni sia tra di loro che tra loro e gli utenti reali

io:


Ok quindi una volta trainato il modello sul dataset reale senza n2v, date le feature in input, faccio inferenza sul sintetico per i collegamenti tra gli utenti sintetici, corretto? (mi limito a sintetico-sintetico per ora e vedo se possibile, altrimenti testo reale-sintetico). Per quanto riguarda le feature di cui parli però, ti riferisci a quelle dei post nel dataset reale? Perchè confrontando i due dataset, nel reale abbiamo i post degli utenti con le info relative al post stesso, mentre nel sintetico abbiamo le info degli utenti con anche il contenuto di un loro post, e sono ben diverse a livello di label e split, l'unica feature equiparabile è "content" di entrambi i dataset.
 
Il sintetico inoltre non ha quell'incremento della rete presente nel dataset reale (nel sintetico quasi tutti gli utenti sono già presenti dal primo batch), quindi li partiziono cercando di ricreare un'evoluzione della rete analoga a quello reale
 

dottorando:

si, entrambi i punti sono esatti.
Addestri sul reale e fai inferenza sul sintetico. Sinteitco-sintetico va bene.
E riguardo le feature sì, mi riferisco a quelle testuali che sono le uniche in comune tra i due dataset