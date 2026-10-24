# FINAL — sintesi di consegna

## Cosa è il progetto

Si impara da una rete sociale reale (25.714 utenti, 824.045 archi, 9 snapshot) **come nascono i
legami**, e si applica il meccanismo a un dataset sintetico di 10.000 utenti di cui si conosce solo
il testo dei post, senza archi e senza date. Il risultato sono due reti sintetiche con la stessa
pipeline **V9**, su due partizioni temporali diverse.

| rete | partizione | ruolo | archi | grado medio | max | reciprocità | transitività | clustering | modularità | diametro |
|---|---|---|---|---|---|---|---|---|---|---|
| reale | — | riferimento | 824.045 | 32,05 | 1.290 | 0,4177 | 0,0337 | 0,0804 | 0,2751 | 6 |
| **ufficiale** | ingresso progressivo | consegnata | 271.646 | 27,45 | 274 | 0,3224 | 0,0212 | 0,0420 | 0,1074 | 6 |
| controllo | tutti presenti | confronto | 280.298 | 28,30 | 324 | 0,3035 | 0,0209 | 0,0433 | 0,1058 | 6 |

## Perché la progressiva

Nel reale il **41,8%** degli utenti finali è già collegato al primo snapshot e il restante **58%**
entra durante il periodo osservato. Una rete che nasce deve avere arrivi, quindi la partizione con
ingresso progressivo è la più realistica; la versione "tutti presenti" resta come controllo.

Il costo è dichiarato e misurato:

- grado medio 27,45 contro 28,30, massimo 274 contro 324;
- archi fra utenti arrivati nello stesso passo **17,0%** contro 12,4% (reale 8,1%);
- grado massimo all'**81%** di quanto la legge di scala del reale prevede a 10.000 utenti (il
  controllo è al 97%);
- primo passo con il **15,7%** dei nodi collegati contro il 41,8% del reale, perché k si riscala sulla
  popolazione presente e al primo passo è solo quella entrata.

Le due reti sono comunque equivalenti sulla distanza relativa dal reale (**3,84** contro **3,89** su
sette metriche, formula in `src/variant_scorecard.py`): la scelta è concettuale, non metrica, e
invertendo la normalizzazione l'ordine si inverte.

## I tre compiti

**1. Link prediction sul reale (tNodeEmbed).** Rappresentazioni dal testo con MiniLM (media dei post,
dopo aver misurato che il testo concatenato veniva troncato), Procrustes a catena, LSTM, testa di
coppia `[h_u, h_v, h_u⊙h_v]`. Confronto di sei encoder con negativi appaiati: vince MiniLM (**0,534**
di media val+test), la codifica per post porta a **0,549**. Con i negativi casuali l'AUC è ~0,99 in
validation e 0,95-0,96 nei test; con gli appaiati **0,51-0,57**, cioè vicino al caso: il contenuto da
solo non predice il legame.

**2. Generazione (V9, eventi di arrivo).** Nel reale nessun arco nuovo collega due utenti già presenti
(quota "vecchio→vecchio" 0,0000% in tutte le transizioni): si genera quindi per **evento di arrivo**,
con k archi per entrante presi dalla distribuzione reale (media 35-111 per transizione, massimo
1.202) e partner scelti da una logistica a sette variabili. Volumi, gradi e hub **emergono**, non si
impongono.

**3. Predizione degli attributi (solo sintetico).** Dal testo dell'utente (TF-IDF + SVD, 128
dimensioni) si predice `political_leaning` (AUC **0,827**), `profession` (**0,867**), `age_interval`
(**0,690**), con split 80/20 per utente. La rete non aggiunge nulla (0,827 → 0,828) e gli embedding di
tNodeEmbed vanno peggio (0,619 / 0,587 / 0,514). Il reale non ha attributi, quindi lì il task non
esiste.

## Il risultato centrale

**Il contenuto non decide chi si collega a chi.** Le tre reti generate con testo vero, testi scambiati
fra utenti e termine di contenuto azzerato sono indistinguibili: archi **271.646 / 261.680 / 271.640**,
reciprocità 0,3224 / 0,3229 / 0,3227, omofilia del contenuto ~1,00 in tutte. Nel modello fittato il
contenuto pesa **+0,05** contro **+1,22** della "controparte nuova" e +1,02 del grado.

L'omofilia 2,13 del reale non contraddice questa lettura: scomposta, scende a **1,51** contro non-archi
appaiati per grado e a **1,29** con grado uguale entro il 10% — è in gran parte struttura.

## Cosa resta aperto

- **Coda dei gradi più ripida** del reale: −3,98 sull'ultimo decile (controllo −3,33, reale −2,35).
  A pari dimensione il reale al primo snapshot sta a −1,69, quindi gli hub generati sono meno estremi.
- **Coesione più bassa**: clustering 0,042 contro 0,080, transitività 0,021 contro 0,034.
- **Primo passo della progressiva troppo vuoto**, per la ragione detta sopra: è il prezzo della
  partizione, non un errore di misura, ma resta un'estrapolazione della legge di k sotto i 10.000
  utenti (fittata fra 10.755 e 21.082).
- **Il task profili** è un diagnostico riuscito, non un risultato sul reale: lì gli attributi non
  esistono.

## Cosa viene consegnato

`Esame Angelantonio Fedele Murolo/` con dieci cartelle: la rete reale, il dataset sintetico, le due
partizioni e le sei reti V9 (tre modalità di contenuto × ufficiale e controllo). Ogni rete ha i 9 file
`.edg` (uno per passo) e i 4 file nella forma dei batch nativi. I numeri stanno in `results/*.csv`, le
figure in `results/figures/` e `docs/riferimenti/`.

Riproduzione completa in `README.md`; dettaglio tecnico e limiti in `docs/pipeline.md`; tentativi
precedenti e loro metriche in `archive/storico.md`.

## Tre avvertenze sui numeri

1. L'addestramento **non è riproducibile al bit**: i pesi variano al terzo decimale fra processi e il
   volume di ~10 archi su 271.646. La ripetizione della configurazione ufficiale dà 271.655 archi.
2. La **distanza relativa** è `|valore − reale| / |reale|`, sommata su sette metriche; archi e nodi
   sono esclusi perché dipendono dalla scala, l'omofilia perché tutte le reti generate stanno a 1,00.
   Cambiando normalizzazione l'ordine fra le due partizioni si inverte: non è un criterio di scelta.
3. I **coefficienti** della rete ufficiale vengono da una ripetizione indipendente della stessa
   configurazione (`results/coefficients_generated_snapshots_growth_v2_v9.csv`), perché la run che ha
   generato la rete scriveva su un file unico poi sovrascritto; il difetto è corretto in
   `src/generate_arrival.py`.
