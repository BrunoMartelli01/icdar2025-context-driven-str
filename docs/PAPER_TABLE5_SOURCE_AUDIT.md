# Table5: verifica delle fonti e significato di accuracy

Verifica del16 ottobre2026 su `papers/2503.15639v2.pdf`, sezione4.1,
Table5 e riferimenti42/46/48/66/71. Analisi delle fonti, senza nuova
inferenza, modifiche ai valutatori o rivendicazioni di riproduzione.

**Risultato:** i quattro numeri Total-Text delle baseline si ritrovano come
F1/Hmean end-to-end nelle fonti primarie, con un errore di identità/versione
per Mask TextSpotter. Questo è un forte indizio che la Table5 intenda F1;
non dimostra come siano state calcolate le due righe “Ours”. La loro metrica
operativa resta `UNRESOLVED`. Anche parte dei numeri ICDAR non coincide
con le righe dei modelli nelle pubblicazioni citate.

## Corrispondenze Total-Text

| Riga Table5 | Valore | Fonte primaria e collocazione | Interpretazione verificata |
|---|---:|---|---|
| Mask TextSpotter[42] |71.2|[Mask TextSpotter v3, Table3, PDF p.13](https://arxiv.org/pdf/2007.09482v1#page=13)|F-measure end-to-end, None. Il riferimento42 del nostro paper è invece Mask TextSpotter2018.|
| MANGO[48] |72.9|[MANGO, Table2, PDF p.6](https://ojs.aaai.org/index.php/AAAI/article/view/16348/16155#page=6)|End-to-end, None, input lato lungo1600; 83.6 con Full.|
| SPTS[46] |74.2|[SPTS, Table6, PDF p.6](https://arxiv.org/pdf/2112.07917v6#page=6)|Hmean end-to-end point-based, None; 82.4 con Full.|
| TESTR[71] |73.3|[TESTR CVPR2022, Table1, PDF p.7](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Text_Spotting_Transformers_CVPR_2022_paper.pdf#page=7)|F-score end-to-end, TESTR-Polygon, None; 83.9 con Full.|

“None” indica assenza di lessico esterno. “Full” usa un lessico composto
dalle parole del test set: è un'impostazione diversa, da riportare separatamente.
L'assenza di lessico non elimina i filtri care/ignore del benchmark.

## Verifica delle righe ICDAR

Le colonne sotto sono nell'ordine ICDAR2013 / ICDAR2015 / Total-Text.
I valori originali ICDAR indicano la colonna end-to-end Generic, salvo nota.

| Metodo | Nel nostro paper | Nella fonte primaria | Esito |
|---|---|---|---|
| Mask TextSpotter2018 |84.1 /70.3 /71.2|86.5 /62.4 /52.9|Non coincide con il modello citato; il71.2 appartiene a v3.|
| MANGO |88.7 /72.9 /72.9|88.7* /67.3 oppure73.9* /72.9|IC13 e Total-Text corrispondono; IC15 non coincide.|
| SPTS |88.5 /73.5 /74.2|88.5 /65.8 /74.2|IC13 e Total-Text corrispondono; IC15 non coincide.|
| TESTR |86.4 /73.3 /73.3|IC13 non riportato /73.6 /73.3|Solo Total-Text corrisponde. IC15 senza lessico è65.3.|

Per Mask2018: [Tables1,2,4, PDF pp.12,12,14](https://openaccess.thecvf.com/content_ECCV_2018/papers/Pengyuan_Lyu_Mask_TextSpotter_An_ECCV_2018_paper.pdf).
Per MANGO: Table1; l'asterisco usa i lessici specifici di Mask TextSpotter2019.
Il73.9 IC15 è MANGO*; il67.3 è MANGO con il lessico convenzionale. Quindi
non esiste un unico numero ICDAR indipendente da lessico, scala e task.
SPTS: Tables4/5, confermate dalla
[repository ufficiale](https://github.com/shannanyinxiang/SPTS#performance).
TESTR: Table3 del PDF CVPR, oltre alla Table1.

Alcune coincidenze numeriche chiariscono il rischio di confondere righe:
nel paper SPTS,73.5 è nella riga Mask TextSpotter su IC15; nel paper MANGO,
70.3 è il Word Spotting Generic di MANGO su IC15 e84.1 è l'E2E Generic di
Boundary su IC13. Nel paper TESTR,86.4 compare come **detection F-score su
Total-Text** nell'ablazione della risoluzione1000, non come risultato IC13.
Queste coincidenze non provano da dove gli autori abbiano ricavato i valori.
Potrebbero esistere loro riesecuzioni diverse, ma non sono documentate a un
livello che consenta di ricostruire queste righe.

## DeepSolo e le differenze dichiarate

La [Table5 ufficiale DeepSolo CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Ye_DeepSolo_Let_Transformer_Decoder_With_Explicit_Points_Solo_for_Text_CVPR_2023_paper.pdf#page=7)
riporta Total-Text None: R50 senza TextOCR79.7, R50 con TextOCR82.5,
ViTAEv2-S con TextOCR83.6. La Table1 definisce esplicitamente None/Full
come F1-measure; anche il testo usa talvolta il termine accuracy per questi
stessi risultati. La Table8 supplementare riporta IC13 Generic R50=90.1.

Nel nostro paper78.4+4.1=82.5: corrispondenza esatta con il R50 **con
TextOCR**, che differisce dal nostro candidato qualificato senza TextOCR.
Invece79.8+3.9=83.7 non coincide con il83.6 tabellato per ViTAEv2-S.
La Table6 DeepSolo dà IC15 Generic76.9/79.1 per R50 senza/con TextOCR e
79.5 per ViTAEv2-S con TextOCR. Il generico “82%” nel nostro paper non
identifica una di queste impostazioni. Non è lecito scegliere checkpoint o
lessico per far coincidere un confronto.

## Come viene calcolato il punteggio

Con TP = parole abbinate correttamente dopo localizzazione e trascrizione,
FP = predizioni care non corrette e FN = parole care non recuperate:

`P = TP/(TP+FP)`; `R = TP/(TP+FN)`;
`F1 = Hmean = 2PR/(P+R) = 2TP/(2TP+FP+FN)`.

I contatori sono aggregati sul dataset, non una media dell'accuracy delle
immagini. Il [valutatore ufficiale DeepSolo, commit dbadae9](https://github.com/ViTAE-Transformer/DeepSolo/blob/dbadae995035246bad3376c7a44c015c69e9b313/DeepSolo/adet/evaluation/text_eval_script.py#L451)
calcola precision/recall da contatori globali e restituisce hmean. Per il
matching poligonale usa IoU **strettamente maggiore di0.5**; neutralizza una
predizione quando oltre metà della sua area ricade in un singolo ignore.
Le regole testuali e care dipendono dal task: E2E e Word Spotting non sono
intercambiabili. Non coincide con bbox-IoU>=0.5 o con casefold puro.

La formula aggregata non basta a rendere uguali i protocolli:

- MANGO dichiara un adattamento del matching con IoU>0.1 per localizzazioni
  grossolane. Questo non autorizza a equipararlo al matching poligonale0.5.
- SPTS associa punti al centro GT più vicino, controlla la trascrizione e
  conta i duplicati come FP. Il suo
  [eval.py ufficiale, commit3f61ec9](https://github.com/shannanyinxiang/SPTS/blob/3f61ec9d3cfd6d69d84086ee975e588f4ec1a926/evaluation/eval.py#L184)
  restituisce Hmean dai contatori globali; Total-Text attiva i filtri Word
  Spotting, inclusa lunghezza minima3 e restrizioni sui caratteri. La CLI
  seleziona il massimo Hmean sullo sweep di confidence0.80..0.94. Questa è
  una proprietà del codice esaminato, non la prova del protocollo “Ours”.
- TESTR specifica IoU fra poligoni; i punti Bezier vengono prima convertiti.
- Nel nostro paper mancano formula della recognition accuracy, lessici,
  matching, ignore e codice di valutazione per stabilire l'equivalenza.
  Inoltre il conteggio168/229 su IC13 contrasta con233 TEST della sua Table1;
  senza manifest questo non dimostra quale split sia stato realmente usato.

## Implicazioni per i nostri esperimenti

La nostra accuracy con TN=0 è `J = TP/(TP+FP+FN)`, distinta da F1. **Solo
a parità di contatori/protocollo** valgono `F1=2J/(1+J)` e `J=F1/(2-F1)`.
Quindi, se78.4/79.8 fossero F1, corrisponderebbero a J=64.47/66.39%,
non a J=78.4/79.8%. È una conversione condizionale, non una ricostruzione
del risultato degli autori.

Il precedente confronto YOLO/CLIP resta una comparazione locale valida fra
gate sullo stesso backbone/fallback: TEST CLIP F1=73.45%, J=58.04% con150
accettazioni descrittive. Il suo matching multiset senza IoU, casefold con
punteggiatura e ignore nativo256 non è il benchmark poligonale ufficiale.
Non basta convertire J in F1 per rivendicare equivalenza con il paper.
Per un confronto esterno servono evaluator/GT ufficiali, stesso checkpoint,
lessico, normalizzazione, ignore e protocollo di soglia; l'output T1 senza
localizzazione richiede inoltre una regola non specificata dagli autori.

## Evidenza e riproducibilità

PDF autori con versioni arXiv fissate, PDF CVF pubblicati, estrazioni testuali,
immagini delle tabelle controllate visivamente, codice ufficiale SPTS e
manifest SHA256 sono conservati fuori Git:
`/extra1/bmartelli/context-str-artifacts/paper-table5-source-audit-20261016/`.
`source_audit.json` e `comparison.csv` distinguono ogni cella confermata,
diversa o di versione differente; non modificano risultati storici.
MANGO è verificato sia sul preprint degli autori sia sulle tabelle del PDF
AAAI accessibili via browser; un download diretto AAAI è fallito e resta
registrato, senza sostituzioni di valori. Nessun modello, dataset o pacchetto
è stato installato/modificato. Riferimento scientifico:
`TABLE5-SOURCE-METRIC-AUDIT-001` in `docs/PAPER_AMBIGUITIES.md`.
