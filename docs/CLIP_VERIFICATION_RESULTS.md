# Confronto CLIP completato: B/32, B/16 e L/14

Il confronto richiesto è completato: tre checkpoint, 40 regole per checkpoint,
125 immagini Total-Text TRAIN, 1.173 box fissi e tutte le 983 care GT nel
denominatore. Sono risultati **approximate reproduction**, su dati già usati
per sviluppo. Non sono TEST indipendente, riconoscimento di nuove parole o
riproduzione della Table 5. L'esperimento DeepSolo sugli slot è separato e non
contribuisce a questi risultati.

## Risultati principali

Ogni valore è il numero di letture esatte, dopo Unicode casefold e con
punteggiatura conservata, con matching uno a uno dei box a IoU >=0.5.
Tutte le alternative sono stringhe PARSeq già disponibili; tutti i metodi
conservano i medesimi box. PARSeq originale ottiene 368/983; selezionare
l'alternativa con confidence PARSeq massima ottiene 383/983 per tutti i modelli.

| Regola | B/32 | B/16 | L/14 |
|---|---:|---:|---:|
| Crop + prompt della parola scritta |362|361|370|
| Crop + ensemble di tre prompt |361|365|374|
| Crop + parola e descrizione della scena |355|356|366|
| Crop + residuo del contesto |369|364|371|
| Fusion dei ranghi PARSeq/CLIP |378|382|386|
| Accordo conservativo PARSeq/crop/crop ampio |370|370|376|
| Confronto visivo con le candidate renderizzate |371|374|383|
| Differenza prima/dopo copertura del target + accordo PARSeq |372|370|374|
| Accordo PARSeq/target visibile/candidate renderizzate |369|370|371|

La fusione usa la media a pari peso dei midrank della confidence PARSeq e dello
score CLIP crop-ensemble. Era dichiarata prima di vedere gli esiti; non è una
nuova soglia ottimizzata su questi risultati. L/14 è il miglior checkpoint
osservato per questa regola: +18 rispetto alla baseline originale, +3 rispetto
alla selezione per confidence. L'osservazione resta esplorativa tra 40 regole.

| Regola L/14 | Corrette /983 | Recuperate | Perse | Cambiate |
|---|---:|---:|---:|---:|
| Confidence PARSeq |383|26|11|351|
| Fusion dei ranghi PARSeq/CLIP |386|19|1|241|
| Accordo conservativo PARSeq/crop/crop ampio |376|8|0|33|
| Candidate renderizzate, canvas con proporzioni conservate |383|19|4|323|
| Candidate renderizzate, crop deformato in quadrato |375|13|6|331|
| Candidate renderizzate, media dei due crop |381|17|4|330|
| Controllo visivo contro stringhe invertite |371|12|9|264|

Recuperi e perdite sono relativi alla baseline originale, con confronto degli
ID GT corretti, non al braccio confidence. Un cambiamento può non incidere
sull'exact matching. Nessuna perdita osservata sul campione non garantisce
assenza di perdite su immagini nuove.

## Le descrizioni aiutano il prompt?

L'aggiunta diretta della descrizione al prompt peggiora la selezione sul crop
in tutti e tre i modelli: B/32 362→355, B/16 361→356, L/14 370→366. Il
residuo contestuale ha esiti misti e non supera il migliore controllo PARSeq.
Con L/14, la descrizione vera dà 366, quella scambiata con un'altra immagine
365: questo esperimento non dimostra un vantaggio utile del contesto reale.

Le descrizioni sono le cache verificate di Florence-2-large-ft, non BLIP-2.
Contengono la parola GT in 160 dei 482 anchor con associazione GT univoca;
possono quindi avere propri suggerimenti OCR. Non entrano nella generazione
di candidate e non derivano dal GT. I prompt rispettano il limite di 77 token
senza troncare la candidata. Questi risultati non escludono contesti più brevi,
locali o separati dal testo, che non sono stati provati qui.

## «La parola è corretta?» e discriminazione delle lettere

Su 482 anchor con una sola care GT sovrapposta, la confidence PARSeq separa
corrette/errate meglio dei singoli score CLIP. Le 688 associazioni mancanti e
le tre ambigue sono escluse soltanto da questa diagnostica; tutte le 983 GT
restano nella metrica principale. I 482 casi hanno 365 parole corrette e 117
errate. AUROC non è una probabilità calibrata di correttezza.

| Indicatore di correttezza | B/32 AUROC | B/16 AUROC | L/14 AUROC |
|---|---:|---:|---:|
| Confidence PARSeq |0.823|0.823|0.823|
| CLIP crop + parola scritta |0.615|0.613|0.593|
| CLIP crop + parola e descrizione |0.562|0.584|0.539|
| CLIP visivo, candidata renderizzata |0.710|0.715|0.725|

La rappresentazione della candidata come immagine è quindi più promettente
del semplice prompt per questa diagnostica. Tra 67 coppie corretto/errato a
distanza Levenshtein uno, L/14 con candidate renderizzate sceglie la corretta
nell'82.1%, come il controllo PARSeq; il prompt scritto raggiunge 68.7% e il
prompt con descrizione 62.7%. Le coppie sono correlate e non costituiscono
evidenza indipendente di significatività. Font, lunghezza, semantica e stile
possono influire anche sul confronto visivo.

Una direzione motivata dai dati è conservare PARSeq come fonte delle candidate
e usare L/14 per conferma visiva: la fusione offre più recuperi netti, il gate
conservativo limita le modifiche. Entrambi richiedono una regola congelata e
una verifica indipendente prima di una scelta definitiva. Il confronto con
stringhe invertite e lo stretching non migliorano il confronto visivo semplice
con L/14 in questa popolazione.

## Provenienza, hardware e verifiche

Il protocollo è quello di
[v1](../configs/approximate/clip_word_verification_train_v1.json) e
[v2](../configs/approximate/clip_innovative_verification_train_v2.json).
Le revisioni sono B/32 `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`,
B/16 `57c216476eefef5ab752ec549e440a49ae4ae5f3`,
L/14 `32bd64288804d66eefd0ccbe215aa642df71cc41`. Tutti i pesi hanno
checksum upstream verificato. La ricevuta B/16 risolve il precedente limite
del checksum per questo run: SHA256
`ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0`.
Il config storico e i risultati CPU rimangono intatti.

L'host ha usato una RTX 3090, modelli separati e sequenziali, float32, batch16,
input [3,224,224], seed20261007, TF32 disattivato e backend deterministici.
Questi sono `HARDWARE_DEVIATION`; non qualificano il target RTX 3070.

| Modello | Peak allocated bytes | Peak reserved bytes | Scoring v1 s | Scoring v2 s |
|---|---:|---:|---:|---:|
| B/32 |687653888|767557632|30.77|141.31|
| B/16 |796634624|926941184|37.13|225.26|
| L/14 |2023449600|2181038080|71.32|610.66|

I tempi sono somme dello scoring registrato per immagine; escludono download,
caricamento iniziale e valutazione. Non sono un benchmark di latenza end-to-end.
Ogni variante è terminata con successo e senza OOM riportato. La sessione Codex
non espone CUDA: la verifica qui eseguita usa gli output GPU dell'host.

Una verifica indipendente ha ricalcolato dai 750 shard tutte le metriche dei
sei report e verificato input, pool, selezioni, pesi, hash del codice e
provenienza. Tutti i 28 controlli v1 sono conservati nelle 40 regole v2 per ogni
checkpoint. I 53 test CLIP/controlli passano; 762 artefatti originali risultano
immutati nell'audit. Nessun output originale è stato modificato. Si conservano i
fallimenti storici del preflight CUDA e lo smoke CPU; non sono risultati GPU.

Artefatti fuori Git:

- Report originale: `/tmp/clip-verification-host-suite/6920d0b0c030/comparison.json`.
- Tabella completa: `/tmp/clip-verification-host-suite/6920d0b0c030/COMPARISON.md`.
- Audit e riepilogo: `/tmp/clip-verification-analysis-6920d0b0c030/verified_results.json`.
- Ripetizione dell'audit: `.venvs/context-str-main-py311/bin/python /tmp/clip-verification-analysis-6920d0b0c030/audit_completed_suite.py`.

SHA256 del report originale:
`2cdbf5b858ad181eff5322a512d41264e613af109a2ef5287e34fcbcf6eb828a`.
I rischi e la classificazione dei metodi restano in
`CLIP-WORD-VERIFICATION-V1` e `CLIP-INNOVATIVE-VERIFICATION-V2` di
`PAPER_AMBIGUITIES.md`. Calibrazione, generalizzazione TEST, altri captioner
e proiezioni pretrained dello spelling restano `UNRESOLVED`.
