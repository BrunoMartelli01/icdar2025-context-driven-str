# Tre test: dalla maschera V3 a crop leggibili

2026-10-21 — **approximate reproduction**, ricerca esplorativa.
Avviati su PAPER dopo autorizzazione dell'utente. Il TEST e l'holdout TRAIN
sono già stati esplorati; questi esperimenti non sono una validazione indipendente
né una riproduzione STRICT/Table5. Decisione: `MASK-CROP-TEST-001`.

Run esterno:
`/extra1/bmartelli/context-str-runs/mask-crop-three-tests-20261021-v1/`.

## Protocollo congelato

U-Net GuardedV3 epoca447, selezionata sulla validation TRAIN bboxWordF1,
SHA256 `5b44de7b88b5b1d636e8f53c586565f6491aabaae3ccea361e96e32d540c1faa`.
U-Net e PARSeq restano congelati. Input RGB256, maschera nativa128, geometria
stretch, soglia0.5, nessun filtro di area, padding1 nativo, ordinamento storico
per bbox allargata. Riconoscimento PARSeq qualificato, AR+refine1, stesso
checkpoint e trasformazione PIL bicubica32x128 per ogni confronto.

Metrica primaria: matching massimo uno-a-uno bbox **non allargata** IoU>=0.5
e testo Unicode casefold esatto; punteggiatura mantenuta. Tutte le predizioni
e tutte le annotazioni care contano, anche immagini senza proposte; nessuna
neutralizzazione delle predizioni su ignore. Micro precisione/recall/F1.
Per il rifinitore il box di localizzazione originale resta fisso: cambia il
ritaglio RGB letto da PARSeq. I guadagni non possono derivare dal cambiare il
box usato per valutare lo stesso testo.

## Test1: budget dei crop — completato

300 immagini TEST,2204 parole. OCR effettivo sui crop aggiuntivi, non stima
geometrica. Ogni crop identico mantiene la stessa lettura PARSeq congelata.

| Limite | Crop | Corrette | Precisione | Recall | F1 | Recuperate/perse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2086 | 795 | 38,11% | 36,07% | 37,06% | 0/0 |
| 20 | 2552 | 876 | 34,33% | 39,75% | 36,84% | 81/0 |
| Tutte | 2688 | 904 | 33,63% | 41,02% | 36,96% | 109/0 |

Le602 proposte aggiuntive richiedono568 nuove letture uniche; alcune erano già
disponibili nella cache congelata di altre varianti di padding. La rimozione
del cap recupera109 parole ma introduce493 ulteriori proposte che non aumentano
i match esatti: F1 sostanzialmente invariato e precisione inferiore. Questo
motiva una selezione migliore, non la promozione automatica del budget illimitato.
Il cap10 del paper rimane il riferimento.

Audit indipendente PASS:1800 cardinalità di matching ricontrollate con SciPy,
denominatori completi, replay795 della baseline e hash delle fonti verificati.
Risultati: `budget/results.json`, `budget/per_image.json`, `budget/audit.json`.

## Test2: separazioni/unioni e selettore

Ogni crop originale resta un'alternativa. Si generano gruppi di2–4 parti da
semi ad alta soglia, erosione con ricostruzione geodetica e seam RGB/probabilità;
unioni di coppie vicine ricompongono possibili frammenti. Si riusano i generatori
esistenti conservando la corretta mappa stretch nativa128→RGB. Pool massimo64
crop per immagine; generazione priva di GT.

Un piccolo classificatore di correttezza usa geometria, supporto della maschera,
testo/confidence PARSeq e tipo di proposta. Si addestra sulle1130 immagini TRAIN,
con negativi incompleti, multiword e mal localizzati. Costo per crop e margine
per sostituire un gruppo sono scelti sulle125 immagini TRAIN di holdout
(983 care word), includendo la politica identità. Nessuna scelta dipende dal
TEST. Il confronto salva sia le parole potenzialmente presenti nel pool sia
quelle effettivamente emesse: il primo è un diagnostico, non un selettore.

## Test3: rifinitore supervisionato dal riconoscimento

Una piccola CNN riceve un contesto RGB+probabilità e il crop iniziale, e predice
traslazione/scala limitate. Inizializzazione identità: gli hash dei crop RGB e
il conteggio OCR devono riprodurre esattamente la baseline prima del training.
Loss: cross-entropy causale PARSeq con EOS, copertura GT, contaminazione dei
vicini e regressione geometrica. Il gradiente della sola loss PARSeq rispetto
alle coordinate è verificato esplicitamente sul primo batch reale.

Massimo4096 proposte TRAIN selezionate con seed,8 epoche, batch16; esclusioni
di charset/lunghezza/IoU riguardano soltanto gli esempi del training, mai il
denominatore della valutazione. Si sceglie la migliore epoca **appresa** usando
l'OCR reale sul completo holdout125. Il confronto registra anche l'identità
e la politica accettata sul holdout, per non nascondere eventuali regressioni.
Gli hash dei tensori PARSeq prima/dopo devono coincidere e i suoi parametri
non ricevono gradienti né aggiornamenti.

Il campionatore bilineare differenziabile è un proxy di training; la valutazione
usa sempre box interi sul RGB originale e il preprocessore PARSeq qualificato.
La validità di questo trasferimento è una domanda sperimentale, non un'equivalenza.

## Risultati completi: test2/3

Entrambi terminati,8 epoche del rifinitore completate. Audit indipendente PASS:
4800 cardinalità di matching SciPy, popolazioni/denominatori, hash di pesi e
codice, selezioni congelate prima del TEST e box di detection fissi verificati.

| Variante | Crop | Corrette | Precisione | Recall | F1 | Recuperate/perse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2086 | 795 | 38,11% | 36,07% | 37,06% | 0/0 |
| **Test2: tagli/unioni selezionati** | **2190** | **870** | **39,73%** | **39,47%** | **39,60%** | **92/17** |
| Test3: rifinitore appreso, epoca7 | 2086 | 768 | 36,82% | 34,85% | 35,80% | 32/59 |
| Test3: politica accettata sul holdout | 2086 | 795 | 38,11% | 36,07% | 37,06% | 0/0 |

Il selettore sceglie189 sostituzioni:78 threshold split,13 erosion split,
40 seam split,58 unioni. Costo0.15/margine0 sono scelti sull'holdout TRAIN,
dove si passa320→365 corrette (F1 35,01→39,00%). Sul TEST il beneficio netto
è75 parole e2,54 punti F1, con precisione e recall entrambe migliori.
Il pool completo contiene1032 match esatti potenziali in5537 proposte:
è un diagnostico con GT, non l'accuratezza di un selettore disponibile.

Il rifinitore non supera la baseline sull'holdout: la migliore epoca appresa
è7,309 corrette contro320; per questo la politica accettata mantiene i crop
originali, prima di osservare il TEST. Si riporta comunque il suo risultato
appreso negativo768/2204. La loss TRAIN scende ma32 recuperi sul TEST non
compensano59 perdite. Il trasferimento tra proxy bilineare/teacher forcing e
riconoscimento reale AR+refine1 resta una possibile causa da isolare, non una
causalità dimostrata. In questa forma il rifinitore non va promosso.

L'audit trova inoltre **due file immagine identici nel TRAIN e nel TEST locali**:
TRAIN212↔TEST578 e TRAIN81↔TEST656, verificati tramite SHA256. Il risultato
primario conserva tutte300 immagini; nessun dato o protocollo originale è
modificato. La sensibilità secondaria sulle298 immagini/2187 parole rimanenti
mantiene il beneficio del test2:782→858 corrette, F1 36,73→39,35%; il rifinitore
resta negativo755. Non è una nuova valutazione ufficiale né una ricerca
esaustiva di quasi duplicati. Fonti/casi e numeri in
`duplicate_image_sensitivity.json` nel run esterno.

Confronto completo: `comparison.md`, dati `comparison.json`, controllo
`independent_audit.json` nella radice del run. Nessuna promozione automatica
nella pipeline di produzione o nella configurazione STRICT.

## Esecuzione e controllo

Gli output di test2/3 sono in `learned/`; stato aggiornato `learned/status.json`,
log `learned.log`, selezioni congelate prima del TEST in
`learned/choices_frozen_before_test.json`, risultati finali `learned/results.json`.
Il job completo non richiede un ulteriore comando per avviare il TEST.
`launch.json` è la ricevuta di avvio, non lo stato aggiornato del processo.

Codice eseguito da copie isolate: `source_budget_v2/` e `source_learned/`, con
manifest SHA256 completi. Il primo tentativo di avvio budget è fallito prima
dell'inferenza per una dipendenza `pipeline` omessa dalla copia; log e ricevuta
falliti sono conservati. La copia completa ha passato gli import, e il run
budget successivo è completo e verificato. Lo smoke di test2/3 è passato anche
dalla copia isolata prima dell'avvio completo; non è un risultato di efficacia.

GPU effettiva NVIDIA RTX3090/24GiB, FP32, niente AMP/TF32, modelli caricati per
fasi; batch1 U-Net,32 OCR,16 rifinitore. Ogni fase registra picchi allocated e
reserved e segnala OOM/errori. Altri job restano in esecuzione: non confrontare
questi tempi con benchmark su GPU esclusiva. Il backward CUDA di `grid_sample`
è esplicitamente non deterministico durante il training; seed fissati e
inferenza deterministica ripristinata. Nessuna installazione o download.

Configurazione con tutti i percorsi, hash e parametri:
[`mask_crop_three_tests_v1.json`](../configs/approximate/mask_crop_three_tests_v1.json).
Per ripetere scegliere output nuovi e passare esplicitamente modalità/config:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venvs/context-str-main-py311/bin/python -u -m scripts.run_mask_crop_budget \
  --mode approximate --config configs/approximate/mask_crop_three_tests_v1.json \
  --output /percorso/nuovo/budget

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venvs/context-str-main-py311/bin/python -u -m scripts.run_mask_crop_learned_tests \
  --mode approximate --config configs/approximate/mask_crop_three_tests_v1.json \
  --output /percorso/nuovo/learned
```

46 test focalizzati PASS, smoke GPU dei tre percorsi PASS, gradiente reale del
riconoscitore verso il crop PASS. Il controllo scientifico precedente è in
`artifacts/diagnostics/mask-to-crop-research-20261020/README.md`.
