# messaggio.md — aggiornamento per il dottorando

*Testo da inviare a Francesco. Il corpo del messaggio comincia da "Ciao Francesco", dopo il separatore: quello che segue è scritto per essere incollato così com'è (niente titoli o grassetti da interpretare).*

---

Ciao Francesco,

ti scrivo per aggiornarti sul progetto di esame, che ormai ha un quadro completo. Ti racconto come sono arrivato ai risultati e cosa ho prodotto; in fondo ti chiedo un paio di cose. Leggi pure con calma.

Il protocollo di valutazione, in breve

Nel dataset reale ogni arco nuovo coinvolge almeno un utente che entra, quindi la restrizione standard del protocollo di valutazione lascia zero archi di test. Valuto perciò sull'intera transizione, con negativi appaiati (stesso primo utente, destinazione con storia di arrivo e grado simili) e, per trasparenza, anche con negativi casuali: con questi ultimi l'AUC sta intorno a 0,99, con gli appaiati intorno a 0,5, ed è la seconda la colonna su cui prendo le decisioni.

Come ho ripartito il sintetico in snapshot

Il dataset sintetico non ha date, quindi l'ho ripartito in 9 passi, quanti sono gli snapshot del reale. Il criterio è misurato sul reale: la curva di arrivo (quanti utenti entrano a ogni passo) e la forma dell'attività (quando e quanto scrive ciascuno, con le raffiche collocate a caso dentro la vita dell'utente). Non uso la curva aggregata dei post per snapshot del reale, perché gli snapshot hanno durata molto diversa — da un anno a pochi giorni — e quei conteggi misurano attività per lunghezza della finestra. Resta mia una sola assunzione dichiarata: che gli id dei post riflettano l'ordine di generazione, e quindi l'ordine di arrivo degli utenti.

Ho provato due ripartizioni: una in cui gli utenti entrano progressivamente, con la curva di arrivo che decide quando ciascuno compare (come nel reale), e una in cui tutti gli utenti sono già presenti dal primo passo. I confronti finali, e i numeri che seguono, sono sulla seconda; l'ingresso progressivo resta come controllo di sensibilità, e l'ho girato anche con la versione finale del modello per confrontarle davvero. Le due reti sono quasi equivalenti: la distanza relativa dal reale su otto metriche è 3,79 contro 4,39, che diventa 3,26 contro 3,29 togliendo l'unica voce in cui si separano, la quota di archi fra utenti arrivati nello stesso passo (12,4% contro 17,0%, mentre nel reale è 8,1%). La progressiva va meglio sulla reciprocità (0,3224 contro 0,3035); quella con tutti presenti va meglio su volume, grado medio, grado massimo e coesione.

Come ho generato la rete

Il meccanismo di formazione degli archi non lo decido io: lo fitto sulle transizioni del reale, con una regressione logistica su variabili standardizzate, e poi genero passo per passo campionando, ricalcolando le variabili sullo stato della rete appena generata. Nessun conteggio e nessuna soglia imposti: volume, gradi e hub emergono dal campionamento. L'unica informazione che entra come feature è il contenuto dei post; le label di profilo del sintetico restano fuori dall'input e servono soltanto in valutazione.

Il contenuto e il modello temporale

Per il contenuto ho confrontato sei encoder sul reale a parità di protocollo (tf-idf+SVD, all-MiniLM-L6-v2, BGE-small, BGE-base, GTE-small, GTE-base) e vince all-MiniLM-L6-v2. Poi un miglioramento misurato: codifico ogni post singolarmente e faccio la media dei vettori per utente, perché concatenando i post di un utente in un'unica stringa l'encoder tagliava al proprio limite di token e per gli utenti più attivi sopravviveva il 28% del testo — proprio gli hub. Con la codifica per post l'AUC di protocollo passa da 0,534 a 0,549.

Il modello temporale è tNodeEmbed inizializzato con le rappresentazioni del contenuto: una rappresentazione per snapshot, un allineamento di Procrustes ortogonale a catena fra snapshot consecutivi, una LSTM sulla sequenza dei 9 passi e una testa di coppia su [h_u, h_v, h_u * h_v]. L'addestramento usa le transizioni di training del reale, con early stopping sulla validation e test mai toccato.

Le versioni precedenti, e quella finale

Prima di arrivare a questa versione ho provato altre famiglie di generazione — regole con il conteggio deciso a mano, statistiche calibrate sul reale, inferenza con una soglia sul punteggio del modello — e le tengo come confronto, ma qui ti racconto solo la versione finale, che è quella che funziona.

La versione finale cambia l'unità del modello: non è più la coppia di utenti, ma l'evento di arrivo, cioè l'utente che entra. Nasce da una misura sul reale: un utente nuovo non porta con sé un arco, ne porta un gruppo — in media da 35 a 111 secondo il periodo, con punte di 1.202, cioè lo stesso ordine del grado massimo reale — e circa il 40% di quegli archi è ricambiato dentro lo stesso arrivo. È l'arrivo a creare l'hub, non la somma di decisioni indipendenti. Il modello stima sul reale quante connessioni porta un arrivo, dove vanno i bersagli e quanto sono ricambiate, e poi genera arrivo per arrivo.

I numeri

Valori dell'ultimo snapshot:

- reale: 824.045 archi, grado uscente medio 32,05, grado massimo 1.290, reciprocità 0,418, modularità 0,275, transitività 0,034, diametro 6;
- versione finale: 280.298 archi, grado medio 28,30, grado massimo 324, reciprocità 0,304, transitività 0,0209, clustering 0,043, diametro 6.

Quindi reciprocità, scala dei gradi, coda e triangoli stanno nell'ordine di grandezza giusto. La rete resta un po' più "piatta" del reale, e gli scarti sono tutti nella stessa direzione con una causa unica: la preferenza per i nodi più connessi è sottostimata nel passaggio da 25.714 utenti a 10.000 (l'estremo già presente negli archi nuovi ha grado medio 169, contro 437 nel reale). È il trasferimento di scala, non la forma del modello, e credo sia la direzione in cui il lavoro potrebbe continuare.

Due precisazioni sui numeri, perché è facile leggerli male. La prima: il volume non è un confronto diretto, perché gli archi sono il grado medio moltiplicato per il numero di utenti, e la popolazione sintetica è di 10.000 contro 25.714 — alla scala reale i 280.298 archi su 9.904 utenti equivalgono a circa 317.000, cioè l'88% del valore atteso. La seconda: ho misurato anche la pendenza della coda dei gradi, non solo il massimo, e lì la distanza c'è: l'esponente esce 4,33 contro 3,35 del reale, quindi la coda della versione finale è un po' più leggera del vero, mentre le varianti a soglia la tagliano troppo poco (2,25 e 2,09).

E una cosa che preferisco dirti invece di lasciarla implicita: messa accanto a tutte le versioni che avevo provato prima, la finale non è la più vicina al reale su ogni singola metrica — ce ne sono che fanno meglio sulla densità, sui triangoli o sulla reciprocità. Ma quelle o impongono le statistiche del reale per costruzione, o sbagliano tutto il resto: quello che la finale ha e le altre no è prendere insieme scala dei gradi, coesione e verso della coda senza imporre statistiche alla rete, ed è la ragione per cui la tengo.

Il controllo sul contenuto: tre reti

Per capire quanto conti davvero il testo, ho generato tre reti che differiscono soltanto per il canale testuale sul lato sintetico: testo autentico, testo permutato fra utenti (gli stessi profili riassegnati ad altre persone, così le statistiche del testo restano identiche) e nessun testo. Le tre reti sono indistinguibili nelle proprietà relazionali: reciprocità 0,3035 / 0,3021 / 0,3037, e il volume oscilla senza un ordine preciso. Anche nei pesi fittati il contenuto pesa poco: +0,05 contro +1,21 della variabile "la controparte è nuova" e +1,02 del grado.

L'omofilia del reale, scomposta

Il numero che sembrava contraddire questa lettura è l'omofilia del contenuto della rete reale, 2,13: l'ho scomposta e risulta in gran parte struttura, non scelta di somiglianza. Contro coppie estratte a caso è 2,133, ma contro non-archi appaiati per grado scende a 1,511, e a 1,290 se il grado è uguale entro il 10%. Il motivo è che nel reale vengono collegati gli utenti più connessi e più attivi, e i profili lessicali di chi scrive molto si somigliano da soli: gli estremi degli archi hanno grado medio 403,6 contro 61,7 degli utenti con testo. Resta un residuo piccolo (AUC intorno a 0,58), che vive nei profili TF-IDF cumulativi, cioè in un modo di misurare il contenuto diverso da quello che il generatore usa.

Cosa il testo sa fare

Dall'altro lato, però, il testo conosce l'utente. Prendendo le label del dataset sintetico così come sono — political_leaning, profession, age_interval — e addestrando un classificatore sul solo contenuto dei post, l'AUC è 0,827 su political_leaning, 0,867 su profession e 0,690 su age_interval; aggiungendo la rete (content+graph) non cambia nulla, e gli embedding temporali (tnodeembed) vanno molto peggio, 0,622 / 0,543 / 0,564. La lettura che ne traggo è che chi è un utente sta nel testo, con chi si collega sta nella struttura — e credo sia il risultato più solido del lavoro, perché viene da un controllo e non da una correlazione.

Cose verificate che non funzionano

- Il contenuto non porta il legame: è il controllo delle tre reti qui sopra, e il peso fittato del contenuto resta un ordine di grandezza sotto quello della struttura.
- Il fine-tuning dell'encoder peggiora anche la validation (0,549 contro 0,571 del base) oltre alle due finestre di test: la supervisione è troppo debole e l'encoder si adatta al periodo che vede.
- Rendere addestrabili le feature dei nodi peggiora (0,512 contro 0,534).
- L'idea del "post come arco" — il testo di un singolo post come relazione fra due utenti — è chiusa: nel file degli archi gli id sono utenti e non post (verificato su tutte e nove le cartelle: due colonne, e nessun campo coincide con un id di post), e provata come regola il testo del singolo post non separa le coppie collegate da quelle appaiate (AUC intorno a 0,516, cioè caso). La parte strutturale dell'idea, la co-interazione sullo stesso post, non è verificabile con questi dati: servirebbe sapere quale post è agganciato a ciascun arco.
- Ho provato anche un campionamento dei negativi "misto" nel fit — un terzo di coppie invertite rispetto agli archi veri, un terzo difficili sul grado, un terzo uniformi, che è l'idea più concreta arrivata dall'esterno per sistemare la reciprocità — e peggiora tutto quello che conta: transitività 0,0115 contro 0,0209, archi fra co-arrivati 18,4% contro 12,4%, grado dell'estremo già presente 104 contro 169. I pesi cambiano molto, ma la reciprocità non si sposta, perché nella versione finale non la impara il classificatore: la genera la tabella delle coppie ricambiate. La configurazione documentata resta quella.

Cosa ho prodotto, e i prossimi passi

Ho pronta la relazione con i grafici e un articolo in formato pubblicazione, in italiano e in inglese con i PDF: struttura completa, confronto reale contro la versione finale, il controllo sul contenuto, la scomposizione dell'omofilia e i risultati negativi. I riferimenti li ho verificati uno per uno. Se ti serve te lo giro, anche per farlo leggere al professore.

Le reti te le mando in 9 file .edg, uno per passo, che è la forma da usare per leggere l'evoluzione nel tempo; e anche in 4 file .edg nella forma dei batch che mi hai dato, dove ogni file è la rete ristretta agli utenti di quel batch. Questi ultimi servono se devi agganciare il generato ai tuoi file, non per leggere l'evoluzione: essendo gli stessi utenti quasi in tutti i batch, i quattro file sono in gran parte la stessa rete.

Una nota tecnica, per quando lo rifarai girare: il generatore è ripetibile al bit — stessi 8.293 eventi di arrivo, stessa quota di coppie ricambiate, stessa reciprocità — mentre l'addestramento della rete non lo è, perché i pesi variano al terzo decimale fra un processo e l'altro e il volume passa da 280.298 a 280.299 archi. Per questo i numeri li cito come provenienti da un'unica esecuzione.

Volendo ci possiamo confrontare nei prossimi giorni, anche in chiamata se preferisci, che forse è più veloce. Se ti serve un file, uno script o un grafico specifico, dimmi quale e te lo preparo.

E se ti viene in mente qualche altra idea su cui vale la pena lavorare, sono qui: per esempio si potrebbe estendere la previsione del profilo alle altre label interessanti del dataset sintetico, per vedere quali altre informazioni il testo cattura.

Grazie, e a presto.
Angelantonio
