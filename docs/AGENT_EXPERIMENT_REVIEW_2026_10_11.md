# Audit degli esperimenti degli agenti — 11 ottobre 2026

Classe dei risultati: **approximate reproduction / estensioni sperimentali**.
Questo documento analizza risultati già prodotti; non avvia training, modifica
checkpoint o promuove una variante nella pipeline. Le nuove proposte hanno
beneficio **UNRESOLVED**. Le scelte sperimentali esistenti conservano le
classificazioni e i riferimenti di [PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

## Esito principale

Il segnale meglio sostenuto dai confronti controllati è la separazione tramite
**nuclei di parola/kernel**. Diversi interventi aumentano la copertura o recuperano
alcune parole, ma distruggono crop prima corretti oppure producono molti falsi
positivi. Occorre misurare insieme corrette, perse, recuperate, precisione, recall
e Hmean. Un aumento del solo numero di parole corrette non stabilisce il
miglioramento complessivo.

Gli esperimenti `unet-word-research-v2` e `unet-crop-links-v1` sono ora completati:
le sei branche di training V2 hanno 50 epoche registrate; `crop_word` e
`crop_links` hanno 20 epoche. I relativi risultati OCR riportano `SUCCESS`.
Le indicazioni di training ancora pendente nelle documentazioni di implementazione
sono storiche e non descrivono questi nuovi artifact. Non occorre duplicare quei
training per sapere come sono andati.

## Protocollo e comparabilità

I confronti qui riportati usano il holdout TRAIN di Total-Text, 125 immagini e
983 parole care, split seed 202503; il training usa le altre 1.130 immagini.
Questo holdout è stato riutilizzato per molti esperimenti e selezioni: non è una
verifica indipendente. Non sono risultati sul TEST ufficiale, Table 3 o Table 5.

Per l'OCR, un vero positivo richiede matching massimo uno-a-uno tra bbox RGB
originali esclusivi, IoU >= 0,50 e trascrizione PARSeq esatta dopo `casefold`.
La punteggiatura rimane nella metrica primaria. Dato `TP`, con `N` crop care:

- precisione = `TP / N`;
- recall = `TP / 983`;
- Hmean = `2 * TP / (N + 983)`.

Queste sono metriche end-to-end testo+crop, non accuratezza del recognizer a
crop fissato. Tutte le 983 parole rimangono nel denominatore. Il protocollo
esclude dopo localizzazione soltanto istanze interamente ignorate e mantiene
interi i box parzialmente ignorati; i protocolli originali registrano il sistema
di coordinate in cui si applica l'ignore. Le metriche geometriche native e quelle
sui supporti proiettati in RGB non sono intercambiabili.

I run V1/V2 e crop-links usano RTX 3090 FP32, con segmentazione e PARSeq caricati
separatamente; V2 batch 8 e crop training batch 4. Sono deviazioni hardware
registrate, non una qualifica della RTX 3070. Gli audit centers/refinements
riutilizzano mappe esportate e letture PARSeq CPU qualificate. La verifica CPU dei
refinements riproduce i testi storici GPU per tutti i 1.037 crop care del controllo;
non generalizzare tale identità a mappe, dispositivi o batch diversi.

## Research V2: risultati completi

Fonte corrente: [comparison.json](/extra1/bmartelli/context-str-runs/unet-word-research-v2/comparison.json)
e [comparison.md](/extra1/bmartelli/context-str-runs/unet-word-research-v2/comparison.md).
La [documentazione V2](UNET_WORD_RESEARCH_V2.md) descrive le scelte e la precedente
qualifica CPU; le righe CPU preliminari non vanno sostituite a questo run GPU.

| Variante | Epoca selezionata | Word F1 nativo | Crop care | OCR /983 | Precisione OCR | Recall OCR | Hmean OCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Riferimento frozen | 2 della sorgente | 64,26% | 1.037 | 520 | 50,14% | 52,90% | 51,49% |
| Continuazione control | 0 | 64,26% | 1.037 | 520 | 50,14% | 52,90% | 51,49% |
| Glyph isolated | 50 | 64,32% | 1.038 | 520 | 50,10% | 52,90% | 51,46% |
| Glyph staged | 7 | 64,26% | 1.037 | 520 | 50,14% | 52,90% | 51,49% |
| Threshold frozen | 0 | 64,26% | 1.037 | 520 | 50,14% | 52,90% | 51,49% |
| DB core | 14 | 64,34% | 1.050 | 522 | 49,71% | 53,10% | 51,35% |
| Links isolated | 33 | 5,84% | 20.200 | 490 | 2,43% | 49,85% | 4,63% |

DB core recupera cinque parole e ne perde tre: il saldo +2 non migliora Hmean.
Le teste glyph non migliorano l'OCR. Per continuation e threshold il checkpoint
selezionato è quello precedente al training aggiuntivo: le ulteriori 50 epoche
non producono un modello preferibile secondo il criterio dichiarato.

Il confronto postprocessing usa invece le **stesse mappe embedding epoch 50**
del V1, che non sono le mappe del controllo o del best embedding epoch 0:

| Estrazione delle stesse mappe | Word F1 | Crop care | OCR /983 | Precisione | Recall | Hmean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard | 50,90% | 1.563 | 498 | 31,86% | 50,66% | 39,12% |
| Watershed | 61,24% | 1.104 | 500 | 45,29% | 50,86% | 47,92% |
| Soft embedding cost | 61,52% | 1.104 | 494 | 44,75% | 50,25% | 47,34% |

Watershed elimina molta frammentazione rispetto all'aggregazione hard; il costo
embedding soft non dà un ulteriore guadagno OCR. Tutte e tre conservano lo stesso
foreground. Non attribuire questo confronto a un miglioramento della segmentazione.

### Perché i pixel links falliscono qui

L'audit degli export di tutte le 125 immagini, escludendo gli `ignored_instance_ids`
registrati, conta l'area effettiva del supporto di ogni proposta:

| Variante | Crop care | Area mediana nativa | Area = 1 pixel | Area <= 4 pixel |
| --- | ---: | ---: | ---: | ---: |
| Riferimento | 1.037 | 182 | 5 | 25 |
| Links isolated | 20.200 | 1 | 17.022 | 18.410 |
| DB core | 1.050 | 179 | 5 | 26 |

L'84,27% delle proposte links è un singolo pixel. Le parole coinvolte in split
sono 722/983, contro 83/983 del controllo. Questi sono **risultati osservati**.

La [regola di inferenza](../localize/unet_word_research_v2.py) richiede che entrambe
le direzioni di un link abbiano probabilità >= 0,5; ogni pixel foreground isolato diventa una
propria istanza. La [loss](../segmentation/unet_word_research_v2.py) supervisiona
anchor GT positivi, mentre l'inferenza usa il foreground predetto. La loss train
scende 1,007 -> 0,548, ma l'epoca 50 produce 35.732 crop, peggio dell'epoca 33
selezionata. Questi sono **fatti dell'implementazione e dei log**.

Che il collo di bottiglia dominante sia calibrazione delle affinità, falsi positivi
foreground, soglia reciproca o capacità delle feature frozen rimane **UNRESOLVED**.
Sono ipotesi da isolare. Un filtro d'area rimuoverebbe molti frammenti, senza
dimostrare di ricostruire le lettere mancanti; non va applicato silenziosamente per
nascondere il fallimento. Aumentare semplicemente le epoche non è sostenuto dai log.

## Secondo passaggio addestrato sui crop

Fonti: [comparison.json](/extra1/bmartelli/context-str-runs/unet-crop-links-v1/comparison.json),
[comparison.md](/extra1/bmartelli/context-str-runs/unet-crop-links-v1/comparison.md)
e [protocollo](UNET_CROP_LINKS.md).

| Variante | Word F1 RGB | Crop care | OCR /983 | Precisione | Recall | Hmean | Recuperate / perse vs controllo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Controllo | 63,62% | 1.054 | 521 | 49,43% | 53,00% | 51,15% | 0 / 0 |
| Crop word/kernel | 60,25% | 1.301 | 524 | 40,28% | 53,31% | 45,88% | 75 / 72 |
| Crop links | 6,43% | 18.851 | 476 | 2,53% | 48,42% | 4,80% | 63 / 108 |
| Global links V2 riusato | 5,82% | 20.298 | 490 | 2,41% | 49,85% | 4,61% | 20 / 51 |

Il crop training word/kernel aumenta gli abbinamenti geometrici da 648 a 688,
ma quasi compensa recuperi e perdite OCR e abbassa nettamente la precisione.
Le parole coinvolte in split passano da 124 a 217. Il secondo passaggio produce
candidati utili ma sostituire in blocco i crop base non è una soluzione dimostrata.
Il ramo links ripete il problema di frammentazione; il global-links riusa V2,
non costituisce una seconda prova indipendente.

**Perché 520 e 521 non sono la stessa baseline:** V2 costruisce box dalla geometria
nativa con inversione del letterbox; crop-links proietta i supporti alle coordinate
RGB e valuta lì. Conteggi e ignore producono 1.037 contro 1.054 proposte care.
Non attribuire il +1 a una nuova rete: il checkpoint di controllo è lo stesso.
Ogni variante va confrontata con il controllo esportato dal proprio protocollo.

Il precedente [ricrop senza training](UNET_CROP_REFINEMENT.md) aveva già mostrato,
nel solo smoke su immagine 147, maggiore frammentazione di "MOTORS". Non era una
misura quantitativa di miglioramento sul holdout e non va confuso con il run nuovo.

## Text+Kernel, centri e scelta conservativa dei crop

Il [confronto multitask completo](WORD_MULTITASK_RESULTS_V1.md) è il controllo
causale più chiaro per l'aggiunta dei kernel: word F1 58,96% (Text), 62,13%
(Text+Kernel), 58,41% (Text+Kernel+Center). I kernel riducono le parole coinvolte
in fusioni da 293 a 171, con aumento degli split 48 -> 95. Il modello con centri
ha migliori metriche pixel ma peggiori metriche di parola. La separazione delle
istanze è un obiettivo distinto dal solo foreground.

La successiva [ricerca V1](UNET_WORD_RESEARCH.md) porta il controllo Text+Kernel
a 520 OCR. Il glyph con decoder condiviso dà 517, l'adaptive threshold 505.
Il ranking appreso top10 migliora 429 -> 438 sul proprio insieme frozen; non è
confrontabile come guadagno diretto con il controllo senza cap da 520.

Il [report centers V2](../artifacts/center-guided-word-crops-v2/REPORT.md) confronta
una baseline diversa, da 425 OCR, con le reti di offset. V2 migliora V1 da 335 a
348 parole, ma perde 152 successi della base recuperandone 75. Il
[diagnostico delle perdite](../artifacts/diagnostics/center-guided-v2-word-loss-audit-v1/REPORT.md)
separa le cause osservabili:

- 113/152 perdite non hanno più alcun box IoU >= 0,50; 39 hanno box geometricamente
  ammissibile ma testo sbagliato. Nessuna è attribuita alla competizione del matching.
- Sulle parole perse, la copertura mediana del supporto annotato nel crop scende
  96,8% -> 71,1%; la contaminazione media da altre parole sale 3,9% -> 21,6%.
- L'errore centro scende solo 8,66 -> 8,42 pixel; l'altezza mediana delle parole è
  13 pixel. Rimangono 189 parole con più centri supportati e 128 centri assegnati
  a pixel di più parole. Per 132 parole manca qualsiasi pixel della maschera base.
- Crop incompleti o multiword possono ricevere confidence OCR altissima: FISHING
  diventa FISH, THE/FINEST diventano THEFINEST. Confidence non certifica completezza.

Una selezione conservativa è **già stata misurata**, non è una nuova proposta:

| Politica | Crop care | OCR /983 | Precisione | Recall | Hmean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base centers experiment | 886 | 425 | 47,97% | 43,23% | 45,48% |
| V2 globale + espansione | 1.065 | 348 | 32,68% | 35,40% | 33,98% |
| Vietare fusioni/mixed; modifiche locali solo se confidence migliora | 946 | 451 | 47,67% | 45,88% | 46,76% |

La terza politica recupera 34 e perde 8 parole. È una prova post-hoc sul holdout
riusato; non dimostra generalizzazione. Migliorare affidabilità dei voti,
supervisionare lo sfondo o rendere le scale adattive sono **ASSUMPTION** da testare,
con beneficio **UNRESOLVED**. Non sono spiegazioni causali già validate.

## Risoluzione, recall e regole frozen

La [suite risoluzione](TEXT_RECALL_RESOLUTION.md) è completa nei JSON esterni.
`control512` e `tversky512` selezionano entrambi epoch0; `control768` epoch50;
`tversky768` epoch10, soglia 0,90. La maggiore copertura media per parola non dà
automaticamente maggiore pixel recall o complete-word rate. Il confronto a
maschera Tversky768 fissata legge 443 parole con contorni, 358 con offset V2 e
525 con kernel trasferiti: qui il raggruppamento delle istanze cambia molto l'esito.

La [suite refinements](TEXT_RECALL_REFINEMENTS.md), con
[risultati machine-readable](../artifacts/text-recall-refinements-v1-qualified/summary.json),
isola altre regole sulle mappe frozen:

| Fonte / regola | Crop care | OCR /983 | Precisione | Recall | Hmean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint512 baseline | 1.037 | 520 | 50,14% | 52,90% | 51,49% |
| Joint512 seeded-only / soft fallback | 1.006 | 520 | 51,69% | 52,90% | 52,29% |
| Hybrid256 baseline | 1.143 | 525 | 45,93% | 53,41% | 49,39% |
| Hybrid256 seeded-only | 960 | 514 | 53,54% | 52,29% | 52,91% |
| Hybrid384 baseline | 1.213 | 528 | 43,53% | 53,71% | 48,09% |
| Hybrid384 bounded growth | 1.234 | 530 | 42,95% | 53,92% | 47,81% |

Il filtro Joint512 conserva esattamente le stesse 520 parole corrette togliendo
31 proposte errate. Il migliore Hmean osservato nella suite, Hybrid256 seeded-only,
sacrifica 11 parole rispetto alla propria baseline; usa inoltre due checkpoint.
La griglia 384 aggiunge soltanto tre parole rispetto a Hybrid256 e più falsi
positivi. Non è evidenza che il semplice aumento di risoluzione risolva il problema.

### GT crop: 838/983, non una conferma del precedente 93%

Nel diagnostico riproducibile con lo stesso PARSeq e gli stessi 983 GT bbox,
l'accuratezza a identità fissata è **838/983 = 85,25%**. Il precedente 93% era
un dato riferito dall'utente, privo in questi artifact di protocollo/provenienza
sufficienti per verificarne la comparabilità. Non si conclude che il 93% fosse
sbagliato: non è il valore misurato da questo esperimento.

Delle 463 parole non lette dal Joint512, 334 non hanno bbox predetto IoU >= 0,50;
262 di queste sono lette correttamente dal proprio GT bbox. Altre 129 hanno un
bbox ammissibile ma nessun testo esatto, e 63 sono leggibili dal GT bbox. In tutto
325 fallimenti correnti sono leggibili con quei crop annotati. Questo sostiene
lavoro su copertura, purezza e geometria dei crop. Il GT bbox è un diagnostico,
non un candidato di inferenza o un tetto assoluto: alcuni crop predetti leggono
correttamente parole il cui rettangolo GT non è riconosciuto.

## Evitare duplicazioni e delimitare la nuova prova

Sono già presenti kernel watershed/BFS, heatmap center, offset/voting/consenso,
embedding PAN-like hard e soft, links globali e sui crop, adaptive threshold,
DB core raster, glyph auxiliary, ranking top10, loss Tversky e input768, ricrop
frozen e riaddestrato, distance markers, espansione limitata, fallback filtering,
selezione conservativa delle famiglie, rettifica PCA/Bezier non appresa e
erosione/ricostruzione. Per i dettagli: [crop recovery](TOTALTEXT_CROP_RECOVERY_SUITE.md),
[text recall crops](TEXT_RECALL_CROPS.md), [center audit](../artifacts/diagnostics/center-guided-v2-word-loss-audit-v1/REPORT.md).
Una variante di tali meccanismi va dichiarata come ablation/correzione, non come
metodo completamente nuovo.

La nuova suite in preparazione introduce una **testa privata centerline+raggio
sulla baseline frozen**, classificata **ASSUMPTION**. Non è stata valutata e non
ha benefici misurati. La proposta riguarda una linea longitudinale per parola,
distinta dalla heatmap di un punto centrale e dagli offset verso quel punto.
Una linea da sola non definisce l'altezza del crop: occorre il raggio oppure un
legame esplicito con la maschera word. La
[costruzione GT curva esistente](../segmentation/totaltext_curved_box_v4.py)
usa già una centerline per costruire poligoni, ma non addestra una testa che
predica linee. I nuovi target non devono sovrascrivere i target originali.
Scelte precise e ambiguità appartengono alla documentazione della nuova suite.

## Evidenza e limiti dell'audit

Per V2 e crop-links sono stati letti `run_manifest.json` ed `epochs.jsonl` di
ogni branca; verificati numero/ultima epoca, `SUCCESS` OCR e uguaglianza degli hash
registrati `comparison -> ocr_comparison` e `comparison -> validation` per tutte
le branche. Sono stati ricavati precisione/recall/Hmean dai conteggi OCR e letti
i protocolli registrati. L'audit dei singleton usa `foreground_area` di tutti gli
export V2, dopo l'esclusione degli ID ignorati registrati nella validazione.
Non è stato ripetuto il training o l'OCR; non è una riqualifica completa di ogni
checkpoint, crop, immagine e funzione del repository.

Hash SHA256 rilevati per i file letti:

| File | SHA256 |
| --- | --- |
| `/extra1/bmartelli/context-str-runs/unet-word-research-v1/comparison.json` | `c6704de0e313a9f419f8d3aca7060a69e3f8f669e39b0cc4f397881d81a9f88d` |
| `/extra1/bmartelli/context-str-runs/unet-word-research-v2/comparison.json` | `cc737b83301bf5d6b7b2f31a59ee5745dee77359c016c2393d68fcc021b0b010` |
| `/extra1/bmartelli/context-str-runs/unet-word-research-v2/ocr_comparison.json` | `43bdc6f6374f4211585644b325d7542286bc39702503519eea9b1b46b73abdd6` |
| `/extra1/bmartelli/context-str-runs/unet-crop-links-v1/comparison.json` | `e680d21bb67a8a15a6c9ee9b0d2d13900b282b21406794befc61a7b680f8f8f5` |
| `/extra1/bmartelli/context-str-runs/unet-crop-links-v1/ocr_comparison.json` | `f7e998e19417284d386f792ab469eaf769d78230ebc8bf49d87c35a3ebdb90cd` |
| `/extra1/bmartelli/context-str-runs/totaltext-text-recall-resolution-v1/comparison.json` | `b7165e8fde959b9ddba2721219e0e06ef9fca8c82b9c8cebf53fdb41bd6f5632` |
| `artifacts/text-recall-refinements-v1-qualified/summary.json` | `c076c22b713d9854ac663acb4e5bb90a2319ce6324aeb2f71a8ab445924d0ddc` |
| `artifacts/center-guided-word-crops-v2/summary.json` | `f08981d713688389f91c7acfbc1bc454e3e8df6c58eb87acf481c73bc6629e67` |
| `artifacts/diagnostics/center-guided-v2-word-loss-audit-v1/policy_ablation.json` | `bc80bfd4ae3d612db4668ec577e92e7aa70551182dfb42eb78a2592083b581d4` |

## Priorità dei prossimi test

1. Conservare la baseline Text+Kernel e il suo crop come controllo. Misurare
   separatamente segmentazione, ricostruzione di istanza e selezione del crop;
   per ogni variante riportare recuperi e perdite della stessa identità GT.
2. Qualificare centerline/raggio prima con test di target, pairing, coordinate,
   ignore, parole piccole/curve/vicine, assenza di GT in inferenza e warm start
   rigoroso; poi smoke reale con picchi allocated/reserved e OOM esplicito.
   Confrontare alla baseline frozen sul medesimo protocollo senza promettere gain.
3. Confrontare eventuale selezione conservativa con regole già misurate, stessa
   popolazione/costo/cap e letture OCR qualificate. Non premiare solo il numero
   di recuperi lasciando fuori le perdite o i falsi positivi.
4. Congelare target, loss, soglie, criterio di checkpoint e ricette prima della
   valutazione successiva. Usare una validazione indipendente da questo holdout
   ripetutamente consultato e repliche con seed dichiarati prima di sostenere
   generalizzazione; mantenere il TEST ufficiale distinto dalla ricerca delle
   iperparametrizzazioni. Ogni futuro risultato resta approximate reproduction
   finché non soddisfa le evidenze e il protocollo STRICT.
