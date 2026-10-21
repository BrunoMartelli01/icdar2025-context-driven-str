# TRAIN: errori di localizzazione, completezza dei crop e recupero OCR

Aggiornamento: 19 settembre 2026.

Questo documento è il punto di ripartenza per il lavoro sul grouping. Raccoglie
risultati, conclusioni e limiti dell'analisi degli errori di practical-remap-v1.
Classificazione: **APPROXIMATE_DIAGNOSTIC / approximate reproduction**.
Non è una riproduzione della Table 5 e non contiene una nuova valutazione TEST.

## 1. Stato congelato e vincoli

- ICDAR13 TRAIN: 229 immagini, 849 parole GT care.
- Checkpoint author-protocol epoch-0098:
  SHA256 `b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2`.
- Prediction binaria nativa128, sigmoid>=0.5; cache e backprojection nearest
  storica verificate. Nessun cambio di risoluzione o threshold in questi audit.
- Baseline practical-remap-v1 congelata, padding6 e top10 invariati.
- Algorithm1 paper-faithful, v1, v2, segmentation, scoring e routing non modificati.
- PARSeq qualificato; testo V1 riutilizzato dalla cache.
- Analisi geometrica CPU. Solo il successivo OCR sui crop diagnostici è stato
  eseguito dall'utente su PAPER; Codex non ha eseguito CUDA.
- Nessun training, grid, nuova segmentation, uso del TEST o selezione di parametri.

## 2. Baseline TRAIN verificata

| Metrica | Risultato |
|---|---:|
| Parole GT |849|
| Predizioni disponibili per associazione diagnostica |745|
| Crop V1 complessivi |1203|
| PARSeq exact casefold |502/849 = 59,13%|
| PARSeq RapidFuzz casefold>=0,8 |578/849 = 68,08%|
| GT con proposal IoU>=0,5 |568/849|
| Errori exact, inclusi output mancanti |347|

Associazione storica: crop con massimo IoU verso ogni GT, disponibile soltanto
con intersezione positiva. Nessuna predizione mancante esclusa dal denominatore.
Questa associazione usa GT SOLO per diagnostica, non per routing.

## 3. Primo audit: quale crop riceve PARSeq?

| Situazione osservata tra i347 errori | Parole GT |
|---|---:|
| Nessun crop retained sovrapposto |104|
| Crop IoU<0,5, non multiword |102|
| Crop multiword |75|
| Crop IoU>=0,5, non multiword, OCR errato |66|

Le quattro righe sono mutuamente esclusive. Non sono però quattro cause provate.
I75 sono parole GT fallite, non il numero di crop multiword dell'intero dataset.

**Risultato chiave:** tra i66 errori con IoU>=0,5,49 escludono almeno un intero
glifo annotato;62 tagliano almeno parte dei glifi. Solo4 contengono tutti i pixel
dei glifi GT. Quindi bboxIoU>=0,5 NON garantisce una parola intera per PARSeq.
Il nome storico `GOOD_BOX_OCR_ERROR` non significa errore intrinseco del recognizer.

Inoltre, nessuno dei347 errori ha una trascrizione exact in un altro crop V1 già
prodotto e geometricamente sovrapposto alla stessa parola. Un semplice reranking
degli stessi output V1 non basta a recuperarli.

## 4. Secondo audit: glifi mancanti e testo estraneo

Si contano pixel dei glifi GT originali, non area delle bbox. I pixel di altre
parole care sono separati dai Don't Care. La presenza di un singolo pixel estraneo
è registrata, ma non prova da sola una fusione significativa.

| Stato del crop tra i347 errori | Parole |
|---|---:|
| Assente |104|
| Incompleto, senza pixel di altre parole care |159|
| Incompleto, con pixel di altre parole care |28|
| Completo, con pixel di altre parole care |52|
| Completo, senza pixel di altre parole care |4|

Gli ultimi4 includono Don't Care: arlboro/Marlboro, ESP/ESPI, TO/MOTO, KIA/OKIA.
Non correggere le annotazioni o le metriche per renderli corretti: documentare
la differenza fra testo visibile e testo care da valutare.

Tra i291 errori con crop incompleto o assente:

- 152 hanno almeno qualche pixel predetto su ogni glifo annotato;
- 139 hanno almeno un glifo senza supporto predetto; 76 non ne hanno su alcun glifo;
- 173 lasciano fuori pixel propri sui quali la segmentation ha predetto foreground.

Sono viste diverse e parzialmente sovrapposte, non conteggi da sommare.
Un minimo di supporto predetto non prova che il glifo sia ben ricostruito.
In83 errori la mask nativa ha precision e recall>=0,8 nella regione della parola,
ma il crop ha IoU<0,5: esistono problemi a valle di una prediction localmente buona.

La completezza perfetta non è necessaria per ogni riconoscimento:98 delle502
parole corrette presentano comunque un taglio di qualche pixel dei glifi.

## 5. Oracle geometrico: quanto recuperano i componenti già predetti?

Controfattuale: union bbox dei raw predicted components associati alla GT, più
padding6 congelato. Nessuna nuova prediction. Questa unione usa GT e NON è una
regola utilizzabile in inference.

- 173 errori incompleti/assenti migliorano la quantità di foreground proprio incluso.
- 89 diventano completamente inclusi.
- 80 diventano completi senza pixel di altre parole care né Don't Care.
- L'unione non è universalmente vantaggiosa: 8 errori peggiorano nella copertura;
  9 aumentano i pixel di altre parole care.

Composizione degli80 casi selezionati prima dell'OCR:

| Categoria geometrica precedente | Casi |
|---|---:|
| GROUPING_UNDERMERGE |49|
| SUCCESS per IoU>=0,5, ma crop tagliato |25|
| TOP10_TRUNCATION |5|
| OTHER |1|

I5 casi top10 non vanno attribuiti automaticamente alla regola di collegamento.
Questi80 sono selezionati usando GT e fallimenti noti: campione diagnostico
condizionato, non una validazione indipendente.

## 6. Prova OCR GPU sugli80 casi

Report locale verificato contro lo SHA comunicato dall'utente; conteggi ricalcolati
dalle80 righe per-word. PARSeq eseguito solo su crop RECOVERED e GT_BOX, V1 in cache.

| Input PARSeq | Exact casefold | Fuzzy>=0,8 |
|---|---:|---:|
| V1 originale |0/80|25/80|
| RECOVERED, union diagnostica |78/80 = 97,5%|80/80|
| GT_BOX |78/80 = 97,5%|79/80|

Le due parole ancora errate sono le stesse nei due controlli:

| Sample / indice GT zero-based | GT | RECOVERED | GT_BOX |
|---|---|---|---|
|117 / 7|`(412)`|`(412")`|`(412")`|
|313 / 0|`GENTLEMEN`|`CENTLEMEN`|`CENTLENIEN`|

Per78 parole la correzione del crop è sufficiente a ottenere exact senza cambiare
rete o recognizer. Sulle due residue anche GT_BOX fallisce: ulteriore grouping
non è una soluzione dimostrata per questi casi.

**Controfattuale contabile**, mantenendo invariati tutti gli altri output:
502+78 =580/849 =68,32%, rispetto a59,13% (+9,19 punti).
NON è una nuova accuracy end-to-end, NON garantisce il top10 globale e NON è una
prestazione raggiunta da un localizer senza GT. Il fuzzy analogo sarebbe633/849,
sempre solo sostituzione diagnostica, non pipeline valutata.

## 7. Conclusioni e decisioni

1. Dare priorità al grouping/crop per il sottoinsieme recuperabile prima di
   riaddestrare U-Net: la prova OCR dimostra un margine reale a mask invariata.
2. Non concludere che la segmentation sia sufficiente ovunque. Esistono glifi
   senza supporto e testo piccolo perso; servono diagnosi separate per questi casi.
3. Non ottimizzare solo IoU. Misurare anche copertura dei glifi, testo estraneo,
   riconoscimento e perdita dei successi V1.
4. Non aumentare indiscriminatamente gap o padding e non unire tutti i frammenti:
   i precedenti overmerge e i controfattuali peggiorati restano un rischio reale.
5. Non usare GT, testo corretto o esito OCR per decidere i gruppi in inference.
6. Nessuna nuova regola è stata implementata o promossa con questo audit.

## 8. Prossimo passo proposto, non ancora eseguito

- Esaminare gli edge/componenti mancanti nei 78 recuperi e distinguere undermerge,
  taglio non rilevato dall'IoU e top10.
- Confrontarli con collegamenti geometricamente simili che fonderebbero parole diverse.
- Verificare se una regola semplice basata solo su prediction/geometria distingue
  i due casi; non inventare soglie per inseguire i78 esempi.
- Confrontare una futura variante con V1 su tutto il TRAIN, contabilizzando sia
  recuperi sia perdita delle502 parole corrette e nuovi overmerge.
- Congelare il protocollo prima di eventuale TEST; i ripetuti studi sul TRAIN
  rendono necessaria una validazione indipendente per affermare generalizzazione.

## 9. Artifact, hash e comandi

Tutti i percorsi seguenti sono relativi alla root del repository. Nessun artifact
precedente è stato sovrascritto per creare questo riepilogo.

| Artifact | SHA256 |
|---|---|
|`artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json`|`82acfe8d860aee1a0f7bf70b9308a83aefb4abc17561105c74324d8b990fc56a`|
|`artifacts/diagnostics/train-v1-ocr-failure-audit-v1/glyph-clipping.json`|`3592cbe10dfabbee852b0f33cb35e660d32bc38b69b457fc747de2dfafc4218b`|
|`artifacts/diagnostics/train-crop-completeness-v1/results.json`|`f49d244e1e6c439bb9ec72e7fdb004ef2029bfe5ee405877f7d5f7e32a269cf9`|
|`artifacts/diagnostics/train-recovered-crops-parseq-v1/results.json`|`6dce36b5ac37f6a097a90dfcdc11bda74f6651377aa677f198a8cefe5dabf17c`|

SHA dei quattro JSON verificati localmente il19 settembre2026. CSV e PNG degli
audit precedenti hanno hash registrati nei rispettivi report/README.
Cache OCR persistente: `artifacts/diagnostics/train-recovered-crops-parseq-v1/ocr-cache/`.

Launcher PAPER, con progress e ripresa dalle letture già salvate:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
bash diagnostics/run_recovered_train_crops_host.sh /extra1/bmartelli
```

Non occorre rilanciarlo per consultare questi risultati. Non riesegue segmentation.
Test registrati:66 test mirati per audit/completezza;7 test per launcher OCR/helper;
py_compile,bash syntax e git diff--check superati. Nessun test CUDA eseguito da Codex.

## 10. Provenienza scientifica e limiti ancora aperti

- `PAPER_SPECIFIED`: componenti/contorni,bbox,padding,filtro area,top10 (sezione3.3).
- `ASSUMPTION`: V1 pratico, associazione diagnostica best-IoU, contatori glyph-color,
  oracle con GT, esattezza casefold e selezione condizionata degli80.
- `HARDWARE_DEVIATION`: PARSeq standalone batch1 sul dispositivo PAPER.
- `UNRESOLVED`: definizione comparabile di accuracy Table5; questi report non
  risolvono recognition unit, matching o gestione multi-output del paper.
- GT_BOX per-word ora disponibile per questi80, non per tutte le849 parole.
  Il vecchio836/849 GT_BOX aggregato non permette di identificare tutti gli errori.
- I README iniziali descrivono lo stato prima della prova GPU; questa sezione6
  aggiorna il precedente limite «OCR dei crop recuperati non disponibile».

Riferimenti di decisione: TRAIN-V1-OCR-FAILURE-001 e TRAIN-CROP-COMPLETENESS-001
in [PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).
