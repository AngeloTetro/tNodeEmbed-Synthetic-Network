Ciao Francesco,

ti mando un aggiornamento sul progetto, ora che il quadro è completo: cosa ho fatto, i numeri principali e i punti aperti, in poche righe.

Sul protocollo di valutazione: nel reale ogni arco nuovo coinvolge almeno un utente che entra, quindi la restrizione standard lascia zero archi di test. Valuto perciò sull'intera transizione, con negativi appaiati (stesso mittente, arrivo e grado simili) e, per trasparenza, anche con negativi casuali: con questi l'AUC sta intorno a 0,99, con gli appaiati intorno a 0,5, ed è la seconda la colonna su cui decido.

Il sintetico non ha date, quindi l'ho ripartito in 9 passi con un criterio misurato sul reale: la curva di arrivo (quanti utenti entrano a ogni passo) e la forma dell'attività (quando e quanto scrive ciascuno, con le raffiche collocate a caso dentro la vita dell'utente). L'unica assunzione mia è che gli id dei post riflettano l'ordine di generazione. Ho provato due versioni — ingresso progressivo e tutti presenti dal primo passo — e alla fine le due reti sono quasi equivalenti (distanza dal reale 3,79 contro 4,39 su otto metriche): uso la seconda e tengo la prima come controllo.

La rete non la disegno io: fitto sul reale il meccanismo di formazione degli archi e poi genero passo per passo campionando, ricalcolando le variabili sullo stato della rete appena prodotta. Nessun conteggio e nessuna soglia imposti. L'unica feature è il contenuto dei post: gli encoder li ho confrontati sul reale (tf-idf, MiniLM, BGE-small/base, GTE-small/base) e vince all-MiniLM-L6-v2, codificando ogni post singolarmente e mediando per utente (l'AUC di protocollo passa da 0,534 a 0,549). Il modello temporale è tNodeEmbed inizializzato con le rappresentazioni del contenuto, con allineamento di Procrustes a catena, LSTM e testa di coppia [h_u, h_v, h_u*h_v].

La versione finale cambia l'unità: non più la coppia di utenti, ma l'evento di arrivo, cioè l'utente che entra. Sul reale un arrivo porta in media da 35 a 111 archi (punte di 1.202) e circa il 40% è ricambiato dentro lo stesso arrivo: l'hub nasce lì, non dalla somma di decisioni indipendenti. Prima avevo provato regole a conteggio deciso a mano, statistiche calibrate sul reale e inferenza con soglia: nessuna reggeva il confronto.

I numeri dell'ultimo snapshot. Reale: 824.045 archi, grado uscente medio 32,05, massimo 1.290, reciprocità 0,418, transitività 0,034. La mia rete: 280.298 archi, grado medio 28,30, massimo 324, reciprocità 0,304, transitività 0,0209, diametro 6 come il reale. Il massimo sembra basso, ma la legge di scala stimata sul reale (esponente 1,445 sui suoi nove snapshot) prevede a 10.000 utenti un massimo intorno a 340, e la mia rete arriva a 324: siamo al 95% del previsto. Restano diverse la forma della coda (a pari dimensione è più ripida) e la coesione, ed è lì che continuerei.

Sul contenuto ho tre risultati. Le tre reti che differiscono solo per il canale testuale (testo autentico, testi permutati fra utenti, nessun testo) sono indistinguibili nelle proprietà relazionali: il testo non sposta la rete. L'omofilia 2,13 del reale è in gran parte struttura: contro non-archi appaiati per grado scende a 1,51 e a 1,29 con il grado uguale entro il 10%. Il testo però conosce l'utente: dal solo contenuto predico political_leaning (AUC 0,827), profession (0,867) e age_interval (0,690), e aggiungere la rete non cambia nulla.

Le cose che non hanno funzionato, per completezza: il contenuto non porta i legami (è il controllo qui sopra); il fine-tuning dell'encoder e le feature dei nodi addestrabili peggiorano (0,519 e 0,512 contro 0,534); l'idea del "post come arco" è chiusa, perché il testo del singolo post non separa le coppie collegate (AUC intorno a 0,516, cioè caso); i negativi "misti" nel fit peggiorano la rete; e la chiusura triadica dentro l'evento, misurata in modo più fine, non cambia nulla.

Ti lascio le reti in 9 file .edg, uno per passo, e anche nei 4 file nella forma dei batch che mi hai dato. Se ti servono, ti giro volentieri la relazione, l'articolo (in italiano e in inglese, con i PDF) e la nota tecnica.

Ci possiamo sentire nei prossimi giorni, anche in chiamata se preferisci. E se ti interessa, si potrebbe estendere la previsione del profilo alle altre label del dataset sintetico, per vedere cos'altro il testo cattura.

Grazie, a presto.
Angelantonio
