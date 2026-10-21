# Centerline e raggio: prova nuova sulla U-Net frozen

**Approximate reproduction / estensione sperimentale.** La prova verifica se
una linea longitudinale e lo spessore locale aiutano a ricostruire parole intere.
Non sostituisce componenti del percorso STRICT e non riproduce TextSnake.
La [configurazione](../configs/approximate/word_centerline_v1.json) registra
scelte, hash e classificazioni; l'ambiguità scientifica è
[WCL-001](PAPER_AMBIGUITIES.md#wcl-001--word-centerline-and-radius-extension).

## Perché questa prova

L'[audit degli agenti](AGENT_EXPERIMENT_REVIEW_2026_10_11.md) mostra che i nuclei
Text+Kernel sono un riferimento più solido dei link per pixel. I nuovi ricrop
guadagnano e perdono molte identità e aumentano i falsi positivi; una copertura
maggiore della maschera non garantisce maggiore precisione OCR. I precedenti
centers sono punti/offset verso il centro di parola, non linee longitudinali.
Quei training sono già completati: questa suite non li rilancia.

Una linea da sola non determina l'altezza del crop. **TextSnake** descrive le
parole tramite dischi lungo l'asse, con raggio e orientamento, e ricostruisce la
regione dai dischi. La proposta qui implementata usa la stessa intuizione
geometrica con una testa privata nella U-Net esistente.
[Fonte primaria, ECCV 2018, sezioni 3.1–3.5](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Shangbang_Long_TextSnake_A_Flexible_ECCV_2018_paper.pdf).

## Protocollo fissato prima del training

- Total-Text TRAIN1255: 1.130 immagini di training, 125 held-out, 983 parole care;
  seed202503 e lista/hash identici ai confronti precedenti. Holdout riutilizzato,
  non verifica indipendente sul TEST ufficiale.
- Copia esatta del Text+Kernel control V1, best epoch2, con checksum verificati.
  Tutti i pesi e buffer della rete sorgente, inclusi BatchNorm, restano frozen.
  Non si riaddestra il controllo. Questo misura cosa può imparare una testa
  dalle feature già disponibili; un insuccesso non esclude il beneficio di una
  futura U-Net completamente riaddestrata sulla nuova supervisione.
- Due convoluzioni private3x3, 128→32→32, seguite da1x1 a due canali. Input:
  feature decoder e primo livello encoder. Output: probabilità della linea e
  raggio locale positivo (`softplus + 0.5` pixel nativi). Inizializzazione della
  linea a logit−2 e del raggio a2 pixel, ultimo layer con pesi zero.
- RGB512×512 con letterbox, mappe256×256, stessi preprocessing e augmentations
  della sorgente qualificata. Adam1e−4, weight decay0.01, batch8,20 epoche,
  precisione FP32 e TF32 disabilitato. Tutti i valori sono nella configurazione.
- Checkpoint scelto per F1 geometrico nativo del decoder `line_radius` a
  IoU≥0.5; tie: meno GT fuse, recall, fgIoU, epoca più precoce. Include epoch0.
  Non si sceglie il checkpoint guardando l'OCR dei tre metodi.

I target sono aggiuntivi: le GT originali restano immutate. Si assottiglia
deterministicamente ciascuna **regione piena della parola**, dopo la stessa
augmentazione e il ridimensionamento. Il raggio è la distanza euclidea dal
bordo, con zero padding. La supervisione della linea è una fascia di semilarghezza
raggio/5, tagliata allo stesso owner. Il raggio della fascia proviene dal pixel
di scheletro più vicino. Si conservano frammenti, pixel piccoli e ramificazioni
con audit esplicito; un componente cancellato dall'assottigliamento conserva il
pixel interno più lontano dal bordo, con tie row-major.

È una **ASSUMPTION**: lo scheletro raster Zhang–Suen non coincide necessariamente
con la centerline da coppie di lati del poligono usata da TextSnake. Non si
predicono sin/cos dell'orientamento, non si accorciano le estremità, non si usa
il suo algoritmo di striding. La loss è BCE+Dice per la linea, più SmoothL1
su log-raggio; pesi uguali per owner entro ogni immagine, poi media per immagine
per il raggio. Non è la loss originale di TextSnake. Questa prima prova non
aggiunge clDice contemporaneamente, per non confondere due cambiamenti.

## Tre crop sulle stesse immagini e lo stesso checkpoint

| Metodo | Regola | Domanda sperimentale |
| --- | --- | --- |
| `baseline` | Identico watershed Text+Kernel attuale | Riferimento interno preservato |
| `line_markers` | Componenti della linea predetta come seed, crescita entro il foreground frozen; fallback esplicito sui componenti senza seed | La linea separa meglio parole fuse? |
| `line_radius` | Scheletro della linea predetta, dischi con raggio predetto, anche fuori dal foreground frozen | La nuova rappresentazione recupera crop incompleti? |

Le linee sono sogliate a0.5 e connesse a8 vicini. I dischi in competizione
assegnano ogni pixel al minimo rapporto distanza/raggio; i tie conservano
l'ordine row-major delle componenti. Si taglia al canvas RGB, non alla GT.
Il raggio è limitato a128 pixel con conteggio delle limitazioni. Nessun filtro
per area, nessun limite di crop, padding0: sono diagnostici approximate,
separati dall'Algorithm1 del paper. I crop finali sono bbox RGB originali;
questa prova **non** introduce anche una rettifica curva.

Il checkpoint pubblico PARSeq qualificato viene caricato in un processo successivo.
È lo stesso candidato approximate dei confronti precedenti; l'identità esatta
del PARSeq-S del paper resta [UNRESOLVED, R-001](PAPER_AMBIGUITIES.md#r-001--exact-parseq-s-artifact-is-unidentified).
La metrica primaria
richiede IoU bbox originale≥0.5 e testo esatto dopo Unicode casefold, con matching
massimo uno-a-uno; la punteggiatura rimane. Tutte le983 parole restano nel
denominatore. Si escludono soltanto predizioni interamente ignorate, dopo aver
creato i crop senza GT. Si salvano precisione, recall e Hmean, più identità
guadagnate/perse rispetto alla baseline. La metrica senza simboli è secondaria.
F1 della linea e MAE del raggio sulla fascia GT aiutano a distinguere problemi
di apprendimento da problemi del decoder.

## Qualifica locale e limiti

La qualifica CPU è nel
[JSON target/oracle](../artifacts/word-centerline-v1-qualified/target_qualification.json)
e nello [smoke reale](../artifacts/word-centerline-v1-qualified/cpu-smoke/smoke.json).
Passano **50 test mirati e132 test di regressione correlati**; sintassi shell
e CLI sono verificate. Il [record leggero di qualifica](../provenance/word_centerline_v1_qualification.json)
registra hash, verifiche svolte e lavoro ancora pendente.
Sulle125 immagini, tutti i983 owner sopravvivono, ciascuno con uno scheletro
connesso; nessuna componente della fascia unisce owner diversi.26 componenti
hanno scheletro di un solo pixel.389 owner hanno pixel con oltre due vicini8:
questo conteggio include geometria discreta dei raccordi, non dimostra389
ramificazioni longitudinali reali.

Con **linea e raggio GT**, la ricostruzione produce983 proposte e968 matching
nativi IoU≥0.5: precisione/recall/F1 **98,47%**, fgIoU macro85,02%, fgF1
macro91,83%. Il decoder produce comunque due istanze con pixel su owner multipli
e due parole con più istanze sovrapposte. I dischi discretizzati non ricostruiscono
perfettamente ogni bbox. Questo è un **oracle dei target**, non performance della
rete, non OCR e non un tetto assoluto di riconoscimento. Una linea ideale risulta
una rappresentazione utile; la predizione rimane da verificare.

Lo smoke CPU usa due immagini512×512, batch2/workers0, forward/backward reali
e tutti i decoder: la testa cambia, i tensori sorgente rimangono identici.
Batch ridotto e CPU sono registrati come deviazioni hardware. In questo ambiente
CUDA e il driver NVIDIA non sono disponibili: non sono stati eseguiti il training
completo, la valutazione PARSeq o misure VRAM della nuova suite.

Durante la preparazione, il primo preflight ha fallito un controllo troppo
restrittivo: il riferimento storico contiene owner ridimensionati non ancora
tagliati a target/validity, mentre il dataset li taglia. Il controllo ora confronta
le stesse mappe tagliate, mantenendo tutti gli ID attesi nel denominatore; un
test specifico verifica questo caso. L'errore iniziale è conservato nella
qualifica preliminare `artifacts/word-centerline-v1-qualification/failure.json`.

## Comando sul computer con GPU

Dalla radice del repository:

```bash
bash scripts/run_word_centerline_host.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/context-str-runs/unet-word-research-v1/control \
  /extra1/bmartelli/context-str-runs/word-centerline-v1 \
  /extra1/bmartelli/context-str-artifacts/parseq
```

Il comando verifica artifact, esegue il target oracle, uno smoke GPU **batch8**,
20 epoche della sola testa, tutti i crop, poi PARSeq. Non installa dipendenze,
non scarica pesi e non rilancia le vecchie branche. `DEVICE=cuda:0` e `PYTHON`
sono selezioni runtime esplicite opzionali. Usa il virtualenv principale già
presente. Un lock della suite e uno single-GPU serializzano questo launcher;
non controllano processi precedenti avviati senza quel lock.

Rilanciare lo stesso comando riprende da `last.pt` con optimizer e RNG, oppure
verifica e riusa i risultati completi. Config, sorgenti, hardware/device,
preprocessing, checkpoint e split modificati richiedono un nuovo output. Sono
conservati `best.pt` e `last.pt`; non vengono prodotti checkpoint per ogni epoca.
Una run incompleta non può accedere alla valutazione finale. Non si cambia batch,
precisione o modello automaticamente dopo OOM; il fallimento viene registrato
con modello, input, batch e picchi allocated/reserved.

Risultato da leggere:
`/extra1/bmartelli/context-str-runs/word-centerline-v1/centerline_ocr_comparison.md`
e il corrispondente JSON. Le mappe e ricette di ogni crop permettono di controllare
perché una parola è stata guadagnata o persa. Il vantaggio del nuovo metodo è
**UNRESOLVED** fino al completamento di quel confronto.

Test ingegneristici:

```bash
.venvs/context-str-main-py311/bin/python -m pytest -q \
  tests/segmentation/test_word_centerline.py \
  tests/localize/test_word_centerline_decode.py \
  tests/scripts/test_word_centerline_runner.py \
  tests/scripts/test_word_centerline_ocr.py
```

## Altre direzioni nuove, da provare separatamente

**TextField** impara un campo orientato dal bordo più vicino verso l'interno,
per raggruppare parole irregolari. È un'alternativa alle linee/raggi e differisce
dagli offset al centro globale già provati; non è implementato in questa suite.
La fonte primaria descrive magnitudine per testo/non-testo e direzione per
separare istanze: [TextField](https://arxiv.org/abs/1812.01393).

**Soft-clDice** può essere una successiva ablation della loss per ridurre
interruzioni della linea. È stato studiato per strutture tubulari quali vasi,
neuroni e strade; non garantisce da solo che due parole vicine rimangano separate.
Occorre misurare anche fusioni e OCR: [clDice, CVPR2021](https://arxiv.org/abs/2003.07311).

Prima di cambiare anche il backbone, confrontare loss/recall della linea e
qualità del raggio: se le feature frozen non riescono a impararli, il passo
successivo ragionevole è una **nuova** ablation con decoder riaddestrato e
controllo a pari budget, conservando questa baseline. La sua utilità è
un'ipotesi, non un risultato disponibile.

Rettifica ruotata, Bezier, distance markers, watershed/kernel, embedding,
pixel-links e threshold adattive sono già presenti nel repository. Non sono
proposti come novità; la [rettifica Bezier esistente](../diagnostics/curve_rectification_train.py)
potrà essere riusata in un confronto successivo sui crop nuovi.
