# Routing contestuale della qualità dell'immagine, senza addestramento

Esperimento APPROXIMATE, `approximate reproduction`, esplorativo. Non riproduce
Table5 o il suo protocollo ufficiale. Paper locale §3.5/Algorithm2 già ispezionato:
qui estendiamo la verifica contestuale per stimare **F1 della singola immagine
YOLO640/PARSeq ≥0,8**. Le parole originali restano identiche; l'immagine non
accettata passa interamente a DeepSolo.

## Protocollo congelato

Configurazioni isolate: `configs/approximate/yolo_context_quality_v1.json` e
`configs/approximate/yolo_context_quality_local_paths_v1.json`.
Classificazioni e limiti in `YOLO-CONTEXT-QUALITY-001`, `PAPER_AMBIGUITIES.md`.

- Stessi125TRAIN/300TEST,640 pixel YOLO, PARSeq,425 caption BLIP2 image-only,
 425 branch complete DeepSolo ViTAEv2-S+TextOCR e controlli già auditati.
- TRAIN80: CDF senza etichette e scelta famiglia; TRAIN45: soglie finali.
  Ricevuta congelata prima dei nuovi segnali TEST. TRAIN/TEST sono stati già
  esplorati storicamente: nessuna affermazione di validazione mai osservata.
- Nessun optimizer, gradient update, classificatore appreso, correzione, filtro
  di parole, peso adattato, dizionario GT, installazione o download.
- Modelli FP32 separati sulla GPU effettivamente verificata di PAPER. Registrare
  snapshot/hash, stato prima/dopo, token/input, memoria allocata/riservata e
  fallimenti. Le CDF e le soglie sono calibrazione empirica, non training di rete.

## Prove

1. Ancore letterali: token quotati o attribuiti, per esempio “word PETROSAINS on
   it”. Recupero esatto multiset; edit matching soltanto diagnostico. Caption
   senza evidenza: segnale indisponibile o moltiplicatore neutro. Testo attribuito
   marcato ambiguo; caption fallibili e non esaustive.
2. MPNet: compatibilità della trascrizione con la propria caption meno media di
   tre caption estranee TRAIN80. Nessuna troncatura oltre384 token.
3. CLIP: crop RGB originale, parola e mutazioni deterministiche senza GT;
   margine contro la mutazione più simile. Prompt semplice, stessa caption,
   caption estranea TRAIN80.20° percentile dei margini per immagine. Un prompt
   oltre77 token causa astensione esplicita, mai troncatura.
4. FLAN-T5-base congelato: probabilità teacher-forced Yes/No per supporto
   letterale/coerenza, controllo caption estranea. Target con EOS, pad esclusi.
   Giudice solo testuale: non verifica i pixel, la completezza o la F1.

Controlli: coverage_baseline, all_checks_softcrop, C3 contestuale storico.
Famiglia primaria predefinita base_clip_context; tutte17 famiglie riportate.
Combinazioni fisse `sqrt(baseline * CDF_TRAIN80(segnale))`; contraddizione letterale
`baseline * recupero_caption`. Nessun peso adattato.

Fonti: [CLIPScore](https://aclanthology.org/2021.emnlp-main.595/) valuta
compatibilità immagine/descrizione; [VQAScore](https://arxiv.org/abs/2404.01291)
usa probabilità di risposte visive; [TIFA](https://arxiv.org/abs/2303.11897) scompone
allineamento in domande. FLAN solo testuale qui **non implementa VQAScore**;
una verifica visiva/residua sarebbe un esperimento successivo distinto.

## Metriche

Metrica sorgente: multiset casefold per immagine, punteggiatura conservata,
senza bboxIoU. Don't Care solo nell'evaluatore; tutte983/2204 parole care nei
denominatori. Output vuoti non possono bypassare. Separare quota di immagini
accettate con F1≥0,8, relativo richiamo/intervallo binomiale e F1 micro/TN0 finale.
Soglie empiriche80/90/95% di immagini buone richiedono almeno5 accettazioni su
TRAIN45; nessuna soglia valida significa fallback completo. Budget10/20/30/40/50%
usano ranking senza GT con tie per ID; indicare budget non riempiti.

Solo120/300 immagini TEST leggere raggiungono F1≥0,8: a150 accettazioni la purezza
massima teorica è80%. Il contesto non certifica parole mai rilevate; la copertura
preesistente resta un segnale indipendente fallibile. Migliorare la purezza non
implica migliorare F1 micro contro DeepSolo. Non rivendicare velocità end-to-end
senza includere caption, mappe e verificatori.

## Esecuzione e stato

Su PAPER, dalla repository, senza SSH verso sé stessi:

```bash
.venvs/context-str-yolo-py311/bin/python -m scripts.run_yolo_context_quality \
  --mode approximate \
  --config configs/approximate/yolo_context_quality_v1.json \
  --paths configs/approximate/yolo_context_quality_local_paths_v1.json
```

Aggiungere `--preflight` per verifica iniziale, `--smoke` per6 immagini.
Output esterni `yolo-context-quality-v1/{smoke,full}/HASH`, con piano, codice,
feature, ricevute, soglie e risultati. Restart riusa solo stadi verificati.

Primi20 test focalizzati e smoke con6 immagini/modelli reali passati. Smoke
`1720a648e62e`, risultati SHA7360f1ba8f716a05b48749d4dbd437b13ab807e74d9f96b497616d8c630452cf.
Confronto completo e audit ancora pendenti: nessun risultato positivo anticipato.
