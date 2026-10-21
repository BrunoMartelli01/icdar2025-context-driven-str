# CLIP e PARSeq sul pool ad alto recall

Esperimento **approximate reproduction**, Total-Text TRAIN development.
Affronta la precisione di un insieme di crop già capace di recuperare molte
parole. Non cambia il percorso STRICT, i generatori, PARSeq, DeepSolo, box o
trascrizioni. Paper locale sezioni3.3–3.5: CLIP admission non è specificato.

## Quali esperimenti sono stati trovati

La repository contiene pool più ricchi di quello del primo confronto CLIP:

| Pool/risultato | Crop o output | Parole corrette disponibili/selezionate |
|---|---:|---:|
| Baseline/soglie/finestre, prima della selezione |18.912|782 oracle /983|
| Gruppi complete-link IoU0.55 |8.683 gruppi|781 group oracle /983|
| Selezione guarded_consolidated dei gruppi |7.462 output|738 selezionate /983|
| Tutti i generatori senza variazioni finali |40.380|811 oracle /983|
| Tutti i generatori incluse variazioni finali |472.358|854 oracle /983|

Fonti: `TOTALTEXT_CROP_WORD_GROUPS.md`, `CROP_GROUP_REFINEMENTS.md`,
`TOTALTEXT_CROP_RECOVERY_SUITE.md`, gli artefatti congelati sotto
`/tmp/totaltext-crop-recovery-suite/d957d64fb251/` e
`/tmp/totaltext-crop-group-refinements-v2/train-full/`.
Un oracle è una misura GT-assisted della disponibilità delle letture, non la
prestazione di un selettore. I pool più ampi includono anche crop trasformati:
per testarli servirebbe ricostruire gli esatti pixel delle rispettive ricette.

Il primo test usa i 7.462 output già selezionati, con recall75.08% e
precisione9.89%. Questi hanno una lettura corretta per738 parole; ogni scelta
successiva filtra soltanto, senza cambiare il crop o introdurre nuove parole.
Il controllo PARSeq confidence>=0.95 ne conserva704 da2.806 output,
precisione25.09%, recall71.62%. È il confronto da superare.

La normalizzazione di questo pool rimuove punteggiatura/simboli Unicode,
oltre al casefold. Il precedente confronto CLIP a box fissi conservava la
punteggiatura: i suoi368/405/983 non sono direttamente intercambiabili con
questi conteggi. Il matching resta uno a uno su bboxIoU>=0.5, non polygon Hmean.

## Input autentici e separazione delle annotazioni

Si riusano gli `inputs.json` sigillati preparati dall'esperimento sui crop,
non i punteggi BERT o le sue scelte. Per ciascun output si controllano ID,
box, lettura raw, hash RGB e confidence rispetto alle cache native PARSeq e
alla selezione originale. Il config fissa gli SHA dei due file di provenienza.
Tutte le immagini e i crop sono verificati sui pixel RGB originali prima
dell'inferenza CLIP. I crop minuscoli ricevono esplicitamente `channels_last`.

Prefiltro confidence>=0.5:5.410 crop, con738 parole corrette nel precedente
controllo TRAIN, ricalcolato in questa sessione. La soglia è motivata da questi
dati già osservati: è una scelta di sviluppo esplicita, non una garanzia di
trasferimento o un parametro del paper. Tutte le esclusioni sono registrate;
le altre2.052 predizioni restano nei controlli non-CLIP e nel riferimento.
Nessun crop escluso entra nei bracci CLIP, neppure con soglia finale zero.

L'inferenza legge solo osservabili, immagini e pesi. Tutti gli score e le
decisioni a ranghi sono salvati prima di aprire `evaluation_labels.json`.
Le descrizioni non entrano in questo test. Le annotazioni servono poi per
valutare e selezionare su TRAIN un punto della curva, con questo uso dichiarato.

## Segnali provati

- **Corrispondenza parola/crop:** cosine tra crop e prompt scritto, più media
  normalizzata di tre template già usati nel confronto precedente.
- **Corrispondenza visiva:** renderizzare la stessa parola PARSeq con Sans,
  Serif e Mono, nero/bianco e bianco/nero; confrontare il crop con la media
  normalizzata dei sei embedding visivi. Canvas quadrati conservano il campo
  visivo. Font, dimensioni, polarità e hash dei rendering sono salvati.
- **Testo rispetto a sfondo:** differenza delle cosine con due ensemble di
  prompt, uno di testo leggibile e uno di sfondo/texture/forme senza testo.
- **Parola rispetto a sfondo:** differenza tra il prompt della parola e
  l'ensemble di sfondo. Può ridurre l'effetto della sola plausibilità visiva.
- **Glyph rispetto a immagini uniformi:** differenza tra la candidata
  renderizzata e la media normalizzata di immagini nere/bianche senza lettere.
  È un controllo sulle feature di sfondo, non segmentazione delle lettere.

Questi ultimi contrasti sono ipotesi: CLIP non viene dichiarato un rilevatore
calibrato di testo. Semantica, font, lunghezza della lettura e distribuzione
dei crop possono influenzare tutti i segnali.

Ogni segnale viene trasformato nel proprio rango empirico, con midrank per
i pareggi, sul TRAIN attivo senza leggere etichette. Medie geometriche a pari
peso combinano rango PARSeq e rango CLIP; si provano anche due combinazioni
con glyph e testo/sfondo insieme. Il riferimento CDF è salvato per applicare
la stessa trasformazione a dati nuovi, senza ricalibrarla su TEST. I ranghi
non sono probabilità.

## Precisione a recall conservato

Si riportano tutti i punti della griglia0,0.05,...,0.95,0.97,0.99 per15 famiglie,
oltre alla baseline:331 controlli complessivi. Per ciascuna famiglia si sceglie
su TRAIN la precisione massima che conserva almeno95% delle **stesse identità
GT** corrette nella baseline: almeno702 delle738. Recuperare altre identità
non compensa la perdita delle parole protette.

Confidence e physical-score sono controlli. Se questi superano i bracci CLIP,
il report li mantiene come migliori, senza imporre un risultato CLIP positivo.
Tutte le983 care GT rimangono nel denominatore, incluse quelle non recuperabili.
Si salvano precision/recall/Hmean proxy, corretti, outputs, recuperi/perdite e
match per immagine. Gli export preservano ID/box/letture/RGB e non contengono
annotazioni; la scelta della soglia resta esplicitamente guidata da TRAIN.

## Esecuzione

Dal terminale GPU dell'host, con L/14 già disponibile:

```bash
bash scripts/run_clip_high_recall_admission_host.sh
```

Il default è L/14 CUDA:0, FP32, batch16, un modello per processo. Nessuna
installazione o download. Log `/tmp/clip-high-recall-admission-host.log`.
Il report viene stampato al termine ed è sotto
`/tmp/clip-high-recall-admission-v1/<model>/<protocol-hash>/`.

Altri checkpoint: `--model vit_b32` o `--model vit_b16`. Il primo smoke CPU è:

```bash
bash scripts/run_clip_high_recall_admission_host.sh --model vit_b32 --device cpu --limit 1
```

Il run completo locale B/32 usa `--model vit_b32 --device cpu`, senza limite.
Per controllare gli input senza inferenza, aggiungere `--check`.
Modello, pesi/config/tokenizer, software, codice, device, prefiltro e campione
definiscono il protocollo. Cache alterate o incompatibili falliscono; SHA del
payload rileva modifiche agli score. Non si sovrascrivono risultati storici.
I caricamenti ripetuti hanno ricevute separate e i run interrotti riprendono
solo shard compatibili. CUDA indisponibile e OOM vengono propagati senza
fallback di device, precisione, batch o modello.

## Verifiche

59 test focalizzati passano. Coprono ranghi/pareggi, protezione delle identità
GT, conservazione dei crop, esclusioni a soglia zero, geometria, cache
corrotte, RGB minuscolo e un errore di memoria simulato registrato e propagato.
Il vero smoke B/32 CPU sulla prima immagine è terminato:26 output,15 crop
scorati, due parole GT. Il migliore punto dello smoke mantiene2/2 da2 output,
contro2/2 da3 del controllo confidence0.95. È solo verifica di esecuzione;
non dimostra un miglioramento della precisione sull'intero pool.

Il test completo B/32 CPU ha terminato su tutte125 immagini e tutti331
controlli, con5.410 crop realmente scorati e605.15s di scoring registrato.

| Punto TRAIN | Corrette | Output | Precisione proxy | Recall proxy | Perse dalla baseline |
|---|---:|---:|---:|---:|---:|
| Baseline high-recall |738|7.462|9.89%|75.08%|0|
| PARSeq confidence>=0.95 |704|2.806|25.09%|71.62%|34|
| PARSeq + CLIP ensemble, score>=0.45 |706|2.784|25.36%|71.82%|32|
| PARSeq + CLIP parola/sfondo, score>=0.45 |714|2.798|25.52%|72.63%|24|
| PARSeq + glyph, score>=0.20 |717|4.435|16.17%|72.94%|21|
| PARSeq + glyph residuale + testo/sfondo, score>=0.35 |712|3.234|22.02%|72.43%|26|

Il punto migliore osservato è `native_word_advantage@0.45`. Lo score CLIP è
`cos(crop, ensemble_prompt_parola) - cos(crop, ensemble_prompt_sfondo)`;
lo score finale è la radice del prodotto dei ranghi empirici di questo
contrasto e della confidence PARSeq. Conserva714/738=96.75% delle identità
corrette della baseline. Il miglioramento rispetto a confidence0.95 è piccolo:
dieci parole corrette in più e otto output in meno, +0.43 punti percentuali
di precisione. Non dimostra generalizzazione, né una precisione elevata.

Un ulteriore controllo **post-hoc** fissa il numero di output a2.798: scegliere
le2.798 confidence PARSeq più alte dà703 corrette; il punto PARSeq/CLIP ne
dà714. È una diagnostica del ranking a pari budget, non una regola di
deployment o un budget di parole ricavato dal GT per immagini nuove.

L'audit controlla125 shard di score e125 selection, hash dei payload/codice,
identità dei crop e tutti331 aggregati rispetto alle valutazioni per immagine.
Report: `/tmp/clip-high-recall-admission-v1/vit_b32/7423fe6400b8/results.json`,
SHA256 `baa1ba9bf9bf5e862e9c6ec05f57e204a72abae64647e3924d0f13a978e8ca80`;
audit: stesso percorso, `independent_audit.json`.
Gli export per ogni punto selezionato sono in `selected_outputs.json`;
la trasformazione per dati futuri è in `calibration.json`.

Anche il vero smoke L/14 CPU è terminato su image14,15 crop scorati,
tutti331 controlli e2/2 GT mantenute. L/14 GPU su questo nuovo pool resta
da eseguire con il comando iniziale; i risultati del vecchio test non ne
sostituiscono la misura. Precisione/placement/batching sono `HARDWARE_DEVIATION`;
il seed è una `ASSUMPTION` riproducibile. Su CPU i peak GPU sono null;
nessuna qualifica della RTX3070 viene dedotta dallo smoke.
Protocollo e classificazioni: `CLIP-HIGH-RECALL-ADMISSION-V1` in
`PAPER_AMBIGUITIES.md` e config in `configs/approximate/`.
