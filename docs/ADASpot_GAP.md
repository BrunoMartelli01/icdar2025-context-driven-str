# AdaSpot: Text + Kernel + Gap, prima prova controllata

**Approximate reproduction / estensione sperimentale.** Questa è la prima fase
scelta dall'utente: separazione delle parole, con PARSeq invariato come controllo
OCR. Mask-Align sulle feature, due decoder di riconoscimento e gate per parola
sono fasi successive. Il percorso STRICT e i suoi componenti restano separati.

Sono stati esaminati il paper locale `papers/2503.15639v2.pdf`, sezioni 3.2–3.3
e Algorithm 1, e la proposta AdaSpot fornita dall'utente, sezioni 4.1–4.2 e 6.
SHA256 della proposta: `7e2a5a663f5ada85c607cb5200a3e17578617e9e387ab682c10e032f25bfb059`.
La proposta documenta una nuova ipotesi; non identifica retroattivamente pesi,
decoder o protocolli mancanti del paper originale. Scelte ed evidenze sono in
[ADASpot-GAP-001](PAPER_AMBIGUITIES.md#adaspot-gap-001--textkernelgap-controlled-extension)
e nella [configurazione approximate](../configs/approximate/adaspot_gap_v1.json).

## Protocollo e riuso

- Total-Text TRAIN1255: training1130 e holdout125, 983 parole care, seed202503,
  manifest e split verificati con SHA256. Il holdout è già stato usato per
  numerosi confronti; i risultati sono esplorativi, non validazione indipendente.
  Nessuna immagine TEST addestra o seleziona questa prova.
- Warm start esatto da Text+Kernel, best epoch61 del confronto multitask,
  checkpoint SHA256 `e65691dda944478e9d8b4f022d45d0d3b21edc7c995274554b4c5ca31336766f`.
  Vengono copiati tutti i tensori comuni, inclusi buffer BN. Chiavi, forme o dtype
  incompatibili fanno fallire l'avvio. Il caricamento finale usa `strict=True`.
- Due training separati: `gap` e `control`, ciascuno50 epoche aggiuntive,
  Adam nuovo, decoder/teste comuni LR1e-5 e weight decay0.01. La nuova testa Gap
  usa LR1e-4. Encoder e relativa BatchNorm rimangono frozen; il decoder si adatta.
- RGB512 con letterbox, mappe256, FP32/batch8, TF32 disabilitato, algoritmi
  deterministici e `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Entrambe le branche usano
  medesimo ordine e augmentazioni indicizzate per sample/epoca.
- La nuova testa1x1, 64→1, aggiunge65 parametri. Pesi zero e bias−2 preservano
  text/kernel e disattivano le barriere a epoch0. Non si riaddestrano le vecchie
  suite né si sovrascrivono target, checkpoint o run storici.

## Target, loss e decoder

Il target Gap è costruito **dopo** la stessa augmentazione e il ridimensionamento
dei word owner: pixel validi di sfondo a distanza euclidea≤3 pixel nativi da
almeno due owner distinti. Il disco di vicinato gestisce ogni orientamento;
frammenti di uno stesso owner non diventano parole diverse. Nessun pixel M/K
viene riscritto. Sono esclusi anche eventuali pixel M conservati dall'interpolazione
senza owner nearest. Una parola già persa nel preprocessing è contata, non inventata.
Il raggio3 e questa definizione sono `ASSUMPTION`, non parametri del paper.
Un ampio spazio fra parole può non produrre target; false barriere possono
tagliare lettere o nuclei. Non si assume un miglioramento.

Loss: somma di masked BCE+soft Dice per M/K/G, pesi1, smooth1. La BCE M conserva
la precedente ponderazione3 dei separator. Ogni loss è mediata per immagine
sui pixel validi, poi sulle immagini con supervisione. Don't Care ha gradiente
zero. Batch interamente invalidi non aggiornano Adam né applicano weight decay.
La branca control ha esattamente le stesse loss M/K senza loss G.

Entrambe le branche usano BFS simultanea8-neighbor con tie deterministici,
M≥0.65 e K≥0.5. La nuova branca esclude G>0.5 **prima** di componenti dei nuclei
e crescita: una barriera può separare anche nuclei fusi. G=0.5 rimane attraversabile.
Ogni componente di foreground senza nucleo è un fallback esplicito. Il decoder
riceve solo probabilità e canvas RGB, mai owner, validità GT o trascrizioni.
Gli ID d'istanza restano distinti, senza contorni sull'unione binaria.

Padding0, nessun filtro d'area e nessun limite10 sono scelte diagnostiche
`ASSUMPTION`, separate da Algorithm1. Il controllo qui usa BFS per il confronto
richiesto dalla proposta; la precedente selezione del checkpoint usava watershed.
Questa differenza rimane dichiarata. La stessa nuova mappa viene decodificata
con Gap attivo e disattivo, senza altro forward, per isolare l'effetto della barriera.

## Metriche, checkpoint e risultati

La metrica geometrica usa bbox nativi esclusivi, IoU≥0.50 e matching massimo
uno-a-uno. Precisione/recall/F1 sono micro su tutte le983 parole e proposte care.
Fusioni e split usano qualsiasi pixel nativo condiviso; denominatore983.
Una predizione è ignorata soltanto se interamente fuori dalla validità GT,
**dopo** inferenza; un box parzialmente ignorato viene mantenuto intero.
Le parole senza supporto predetto o perso nel target rimangono nel denominatore.
Pixel IoU/F1 sono macro per immagine sui pixel validi del foreground finale
M senza le barriere; entrambi vuoti valgono1. Le metriche Gap-off sulle stesse
mappe sono riportate separatamente.

Checkpoint: F1 parole, poi meno parole fuse, recall, pixel IoU e prima epoca.
Epoch0 è candidato: un training che peggiora può selezionare il warm start.
`best.pt`/`last.pt`, optimizer, tutti gli RNG e log epoche sono atomici/resumabili.
Config/sorgenti/input/device/backend cambiati richiedono un output nuovo.
Nessuna valutazione finale è ammessa prima del budget completo di entrambe le branche.

Dopo il training, PARSeq viene caricato in un processo successivo, con checkpoint,
archive/revision, preprocessing32x128, AR e refine1 già qualificati. Sono conservati
crop RGB hash, testi raw, probabilità e provenance. Matching originale RGB
IoU≥0.50 e testo esatto dopo Unicode casefold, punteggiatura mantenuta; la variante
senza simboli è secondaria. P=TP/proposte, R=TP/983, H=2TP/(proposte+983).
Si salvano guadagni/perdite delle identità GT, anche per Gap disattivato.
Questi valori sono proxy locali bbox, non official polygon Hmean o Table3/5.
Il candidato PARSeq non è identificato come l'esatto artifact PARSeq-S del paper.

## Avvio GPU e controllo

Usare gli ambienti, dataset e pesi già presenti; il launcher non installa e
non scarica dipendenze o sostituti. Percorsi obbligatori:

```bash
bash scripts/run_adaspot_gap_host.sh \
  /path/to/qualified/totaltext-v4 \
  /path/to/totaltext-word-multitask-v1/text_kernel \
  /path/to/new/external/adaspot-gap-run \
  /path/to/qualified/parseq
```

`ADASPOT_PYTHON` e `ADASPOT_DEVICE` sono override espliciti opzionali; il device
di training deve essere CUDA. Il comando esegue preflight1255, smoke reale
batch8,50 epoche Gap, export, smoke/control50 epoche/export, quindi PARSeq e
`comparison.md`/`ocr_comparison.json`. Le due reti sono caricate in sequenza.
Il lock impedisce un secondo launcher sullo stesso output; non interrompe
altri processi GPU dell'utente. I log stampano device, step, loss e progresso.

Un errore interrompe la suite e salva la causa/traceback; OOM registra modello,
shape, batch e picchi allocated/reserved. Non cambia automaticamente batch,
precisione o dispositivo. I picchi sono per processo, non memoria globale GPU.
Hardware reale e riduzioni/offload devono rimanere documentati come
`HARDWARE_DEVIATION`. La RTX3090 disponibile su questo host non qualifica8GB
di RTX3070. L'efficacia e la generalizzazione restano `UNRESOLVED`.

## Qualifica dell'implementazione

36 test nuovi e64 regressioni pertinenti passano: pairing, rotazioni, owner
frammentati, ignore/gradienti, batch invalidi, barriera/soglia, nuclei fusi,
decoder deterministico, forme della rete reale, trasferimento rigoroso, resume
RNG/Adam, errori, denominatori e confine file/OCR con cache/tampering.
Il primo tentativo pytest ha fallito la raccolta perché tre file avevano lo
stesso basename; i due nomi ambigui sono stati distinti, senza saltare test.
Smoke GPU, preflight completo e avvio fisico sono registrati separatamente nei
risultati esterni e nella ricevuta leggera di qualifica; i test unitari con fixture
non sono risultati di benchmark o training della nuova rete.

Avvio fisico verificato il15 ottobre2026 su RTX3090: smoke FP32/batch8/RGB512
riuscito, testa Gap aggiornata e nessun OOM. Picchi dello smoke:3.70GiB allocated,
3.98GiB reserved. Il training completo è attivo su `cuda:0`; sono già registrate
le prime due epoche della branca Gap, tutte1130 immagini/142 step per epoca,
poi sono previsti controllo50 epoche e PARSeq. Ai primi due checkpoint i picchi
sono3.95GiB allocated e5.44GiB reserved; non sono una qualifica RTX3070.
Il beneficio attribuibile al Gap richiede ancora il controllo abbinato e l'OCR.

[Ricevuta di qualifica](../provenance/adaspot_gap_v1_qualification.json) e
[log del run](/extra1/bmartelli/context-str-runs/adaspot-gap-v1-20261015T125256Z/training.log)
documentano codice, config, ambiente, device e avvio. Il launcher PID1407798
continua in background indipendentemente dal terminale. I risultati sono
approximate; avvio/smoke/epoche iniziali non significano training completato.

## Verifica dei precedenti, 15 ottobre 2026

**Il principio del Gap è già implementato nel repository.** Il generatore
`segmentation/totaltext_word_regions.py` include già uno sfondo raggiunto da
almeno due owner dilatati, combinato con i contatti tra owner. È già indipendente
dall'orientamento. La vecchia testa `PracticalTotalTextSeparatorHeadAGUNet`
predice M+separator, aggiungendo65 parametri, e sottrae il separator prima delle
componenti connesse. Non è una nuova idea introdotta da questa suite.

Sono stati controllati i due run storici completi di100 epoche, gli ordini
TRAIN identici per tutte le epoche, i checksum dei best checkpoint e i risultati
PARSeq già salvati sullo stesso holdout125/983. Risultati nel loro protocollo:

| Variante storica | Epoca best | Word F1 nativo | Parole fuse | Parole split | OCR esatto /983 | OCR Hmean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M control |68|52,07%|199|77|384|39,75%|
| M + separator |94|48,87%|162|99|362|36,47%|

Questa prova riduce le fusioni, ma peggiora il risultato complessivo. Non si
rilancia il vecchio training. Il nuovo esperimento misura l'aggiunta di G alla
baseline **M+K**, con warm start, BCE+Dice, target ricostruito dopo le trasformazioni
al raggio nativo dichiarato e BFS dei kernel. È un'ablation della combinazione;
non è una dimostrazione di novità del Gap. I risultati storici non sono controlli
causali del nuovo run: loss, threshold, estrazione, inizializzazione e trattamento
delle proposte ignorate differiscono. I nuovi controlli rimangono abbinati.

Precedenti primari esterni consultati:

- [WordFence, ICIP2017](https://arxiv.org/abs/1705.05483): predice bordi attorno
  alle singole parole per separarle. È un precedente del principio, senza
  dimostrare equivalenza dell'intera architettura M/K/G proposta.
- [Border Learning, ICCV2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Self-Organized_Text_Detection_ICCV_2017_paper.pdf): classi testo, bordo e sfondo;
  [repository dell'autore](https://gitlab.com/rex-yue-wu/ISI-PPT-Text-Detector)
  con FCN e supporto ai bordi. Il README è reperibile; non è stato installato
  o riqualificato il vecchio stack Keras/Theano/Python2. Il PDF CVF ha risposto
  HTTP403 all'apertura diretta; il contenuto indicizzato è stato consultato.
- [Repulsive Text Border, CVPR Workshops2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w34/Liu_An_Accurate_Segmentation-Based_Scene_Text_Detector_With_Context_Attention_and_CVPRW_2020_paper.pdf): link repulsivi vicino ai bordi di istanze
  distinte. Affine all'obiettivo, con rappresentazione e loss differenti.
  Anche questo PDF risponde403 all'apertura diretta; è disponibile il testo indicizzato.
- [CM-Net, TIP2022](https://crabwq.github.io/pdf/2022%20CM-Net%20Concentric%20Mask%20based%20Arbitrary-Shaped.pdf): ha una testa chiamata Gap Mask, ma G è la fascia
  **interna** tra contorno della parola e nucleo; è ausiliaria nel training.
  Non equivale al corridoio esterno inter-parola usato qui. PDF originale consultato.

La ricerca non dimostra l'assenza di precedenti della combinazione esatta M/K/G.
Nessuna rivendicazione di prima introduzione o superiorità deriva da questo audit.
La ricevuta leggera `provenance/adaspot_gap_prior_audit_20261015.json` registra
identità e conteggi degli artifact; gli originali rimangono immutati.
