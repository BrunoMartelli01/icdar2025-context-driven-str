# CLIP per verificare le trascrizioni PARSeq

Esperimento **approximate reproduction**, Total-Text TRAIN125 già usato per
sviluppo, non valutazione TEST indipendente e non riproduzione della Table 5.
Protocollo: [configurazione](../configs/approximate/clip_word_verification_train_v1.json).

## Come si inserisce nel progetto

Il lavoro di riferimento usa AG-UNet con encoder ResNet-50, componenti della
maschera, crop dell'immagine originale e PARSeq-S. Confronta le letture con una
descrizione BLIP-2 attraverso MPNet e similarità lessicale, poi ricorre a
DeepSolo quando il punteggio è insufficiente. Le sezioni 3.3–3.5 e gli
Algoritmi 1/2 del PDF locale sono la fonte; i dettagli ancora incerti restano
in `PAPER_AMBIGUITIES.md` e `STRICT_REPRODUCTION_STATUS.md`.

Il repository contiene ora runtime PARSeq e molti esperimenti approssimati,
oltre allo scaffold storico descritto nel README. L'adapter PARSeq carica
rigorosamente un checkpoint ufficiale con provenienza, ma il suo abbinamento
al PARSeq-S del paper rimane irrisolto. Questo esperimento riusa letture
esistenti: non modifica segmentazione, PARSeq, DeepSolo o lo scoring STRICT.

## Implementazioni CLIP esistenti

`scripts/crop_clip_runtime.py` e `recognition/crop_clip_battery.py` confrontano
ogni variante di crop con la sua lettura. Il punteggio cambia insieme a due
variabili: i pixel e la parola. Con padding, la vecchia selezione CLIP
peggiorava molto. La relativa documentazione riporta 374/983 per la regola
CLIP threshold-only con margine 0.02, contro 368/983 della baseline e 403/983
del controllo PARSeq; i box scelti potevano cambiare.

`recognition/group_word_selection_runtime.py` affronta già questo problema:
le parole concorrenti condividono l'immagine di riferimento. Il nuovo modulo
estende il confronto con descrizioni, controlli e misure della discriminazione
tra letture corrette ed errate. **I risultati nuovi hanno box fissi**, quindi
non sono intercambiabili con i numeri della vecchia batteria.

## Idee dalla letteratura

* [CLIP originale, §3.1.4 e appendice E.2](https://arxiv.org/pdf/2103.00020):
  valuta anche OCR e osserva sensibilità al dominio e alle cifre/caratteri
  ripetuti. Per OCR, le virgolette nel prompt aiutano; descrive anche ensemble
  nello spazio degli embedding. Queste osservazioni motivano prompt di parole
  scritte e una media di embedding, senza garantire il risultato su Total-Text.
* [CLIPTER, ICCV 2023](https://arxiv.org/abs/2301.07464v2): combina feature
  della scena intera con feature del riconoscitore tramite attenzione con gate.
  Richiede fine-tuning. L'idea trasferibile è separare contesto globale ed
  evidenza locale; il semplice cosine reranking qui provato non è CLIPTER.
* [CLIP4STR](https://arxiv.org/html/2305.14014v4): usa rami visuale e multimodale
  con decoder di riconoscimento addestrati, seguiti da predizione e raffinamento.
  È una direzione per un verificatore specializzato, non una prova che CLIP
  generico congelato sappia correggere ogni parola.
* [Disentangling visual and written concepts in CLIP, CVPR 2022](https://joaanna.github.io/disentangling_spelling_in_clip/):
  studia la separazione tra concetti visivi e spelling, anche su parole senza
  significato, con proiezioni apprese. Motiva distinguere riconoscimento delle
  lettere e plausibilità semantica; una proiezione addestrata costituisce un
  esperimento diverso da questo verificatore senza training.

## Confronto controllato

Gli input sono i 1.173 anchor originali della batteria TRAIN125. Per ciascuno,
le alternative sono esclusivamente le stringhe PARSeq **raw** ottenute da
varianti di soglia a padding zero. Una parola ripetuta conserva tutti gli ID
di origine e il massimo della confidence PARSeq disponibile. Non si aggiungono
parole dal GT, da un dizionario o dal captioner. La baseline originale resta
sempre nel pool. Il box finale resta quello della soglia 0.5/padding zero,
qualunque parola vinca.

Ogni candidata vede quattro immagini comuni al suo anchor:

1. immagine originale intera;
2. crop originale della baseline;
3. crop con larghezza/altezza raddoppiate, limitato ai bordi dell'immagine;
4. stesso crop ampio con il rettangolo della parola coperto dal colore mediano.

Un canvas quadrato con colore mediano conserva tutto il campo visivo prima del
processor ufficiale. Il quarto input è un controllo sul contesto circostante:
non è una nuova proposta di localizzazione, e la copertura sintetica può
introdurre artefatti.

I prompt confrontati sono parola raw, parola scritta tra virgolette, ensemble
di tre template, parola più descrizione, residuo contestuale, e descrizione
scambiata ciclicamente con quella dell'immagine successiva nella popolazione
completa. Il prompt contestuale è:

```text
a photo of the written word "PAROLA". Scene: DESCRIZIONE
```

La parola viene prima della descrizione. Se il prompt supera i 77 token, si
riduce solo il prefisso comune della descrizione; la parola non viene mai
troncata. Budget, prefisso effettivo e prompt sono salvati per immagine.
Le descrizioni sono quelle già generate da **Florence-2-large-ft**, sulle
stesse immagini originali verificate tramite SHA256. Non sono BLIP-2; possono
anche contenere proprie ipotesi di testo, rilevate esplicitamente nell'audit.

Il residuo è l'ipotesi sperimentale
`normalize(embedding(parola + caption) - embedding(prompt senza parola + caption))`.
Sottrarre soltanto lo score della caption, identico per tutte le alternative,
**non cambierebbe l'ordinamento**. Normalizzare la differenza tra feature può
invece cambiarlo: il beneficio deve essere misurato, non presunto.

Si confrontano selezione CLIP libera, confidence PARSeq, fusione con pari peso
dei midrank PARSeq/CLIP-crop-ensemble e un gate conservativo: modifica solo
quando PARSeq, crop e crop ampio concordano, con guadagno cosine del crop
strettamente maggiore di 0.02. I pareggi mantengono la baseline. Nessun valore
è stato ottimizzato sugli esiti di questa batteria.

## Metriche e limiti

La metrica principale è il numero di parole esatte dopo Unicode casefold,
punteggiatura conservata, con matching deterministico uno a uno dei box a
IoU >= 0.5. Tutte le care GT restano nel denominatore, comprese quelle senza
predizione; stringhe vuote non fanno match. Le annotazioni vengono aperte
soltanto dopo aver salvato tutte le selezioni. Il report conserva ID di GT,
predizioni, recuperi, danni e cambiamenti, oltre a un oracle dei pool a box fissi.

Per rispondere anche a «questa parola è corretta?» si misura AUROC sui box
della baseline che si sovrappongono a esattamente una care GT, con conteggi
separati per associazioni mancanti/ambigue e score assenti. La confidence
PARSeq per questo controllo è quella **originale della baseline**, mentre il
ranking delle alternative usa la migliore confidence disponibile per parola.
Si misura inoltre se la parola corretta supera le alternative sbagliate nel
pool, separando le coppie a distanza Levenshtein uno. Le coppie/anchor sono
correlati; non sono campioni indipendenti per una pretesa di significatività.

Cosine, margine e softmax su un insieme ristretto di candidate non sono una
probabilità calibrata di correttezza. Non viene introdotta una nuova soglia
di routing DeepSolo. L'eventuale scelta del miglior braccio su queste 125
immagini è esplorativa; serve una regola congelata e una verifica indipendente.

## Ripetere il test e confrontare i checkpoint

Nessuna dipendenza viene installata e nessun modello viene scaricato dal runner.
Gli input/path sono dichiarati nel
[profilo locale](../configs/approximate/clip_word_verification_local_paths_v1.json).
Le cartelle dei checkpoint devono avere come nome il commit completo. Il
runner salva i checksum dei file, controlla l'architettura B/32, B/16 o L/14,
e rifiuta missing/unexpected/mismatched keys. Non esegue fallback tra modelli.

```bash
.venvs/context-str-main-py311/bin/python -m pytest -q \
  tests/recognition/test_clip_word_verifier.py \
  tests/recognition/test_totaltext_crop_clip_battery.py \
  tests/recognition/test_group_word_selection.py

.venvs/context-str-main-py311/bin/python -u -m scripts.run_clip_word_verification \
  --model vit_b32 --device cpu
```

Per uno smoke reale aggiungere `--limit 1`. Per GPU usare esplicitamente
`--device cuda:0`: se CUDA non è disponibile, l'esecuzione fallisce e registra
l'errore senza cambiare device. Float32, batch16, quattro thread CPU,
seed/settings deterministici e modelli caricati separatamente sono registrati
come `HARDWARE_DEVIATION`. Peak allocated/reserved sono misurati su CUDA;
su CPU sono `null`, senza attribuire misure VRAM inesistenti. L'OOM viene
registrato e propagato; un test simula il ramo di errore, senza qualificare GPU.

Per completare il confronto su un host con rete, scaricare i due snapshot
ufficiali con il client già presente nell'ambiente, negli stessi path del
profilo locale. Non serve reinstallare pacchetti:

```bash
.venvs/context-str-main-py311/bin/huggingface-cli download openai/clip-vit-base-patch16 \
  --revision 57c216476eefef5ab752ec549e440a49ae4ae5f3 \
  --include config.json preprocessor_config.json tokenizer.json tokenizer_config.json \
            special_tokens_map.json vocab.json merges.txt pytorch_model.bin \
  --local-dir /tmp/clip-word-verification-models/57c216476eefef5ab752ec549e440a49ae4ae5f3

.venvs/context-str-main-py311/bin/huggingface-cli download openai/clip-vit-large-patch14 \
  --revision 32bd64288804d66eefd0ccbe215aa642df71cc41 \
  --include config.json preprocessor_config.json tokenizer.json tokenizer_config.json \
            special_tokens_map.json vocab.json merges.txt model.safetensors \
  --local-dir /tmp/clip-word-verification-models/32bd64288804d66eefd0ccbe215aa642df71cc41

# Usare un output distinto: i risultati CPU precedenti hanno un contratto diverso.
.venvs/context-str-main-py311/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('configs/approximate/clip_word_verification_local_paths_v1.json')
profile = json.loads(p.read_text())
profile['output_root'] = '/tmp/clip-word-verification-train-v1-cuda'
Path('/tmp/clip-word-verification-cuda-paths.json').write_text(json.dumps(profile))
PY

set -e
for model in vit_b32 vit_b16 vit_l14; do
  .venvs/context-str-main-py311/bin/python -u -m scripts.run_clip_word_verification \
    --paths /tmp/clip-word-verification-cuda-paths.json --model "$model" --device cuda:0
done
```

Le revisioni B/16 e L/14 sono verificate nelle pagine di commit ufficiali:
[B/16](https://huggingface.co/openai/clip-vit-base-patch16/commit/57c216476eefef5ab752ec549e440a49ae4ae5f3),
[L/14](https://huggingface.co/openai/clip-vit-large-patch14/commit/32bd64288804d66eefd0ccbe215aa642df71cc41).
Nel config storico B/16 il checksum upstream resta assente. La successiva
suite host ha verificato e conservato la ricevuta upstream di entrambi i
checkpoint aggiuntivi, senza cambiare il config o gli output CPU originali.

L'audit iniziale disponeva solo di B/32: DNS/CUDA/NVML indisponibili nella
sessione locale avevano prodotto fallimenti espliciti dei preflight. Il
successivo run GPU dell'host ha completato **B/32, B/16 e L/14**, inclusi
rendering e stretching, sulle 125 immagini. Vedere
[confronto completo verificato](CLIP_VERIFICATION_RESULTS.md).

## Risultati misurati il 7 ottobre 2026

Il run completo B/32 CPU ha terminato su tutte le 125 immagini, 1173 anchor e 983
care GT. Il ceiling delle parole disponibili **a box fissi** è 405/983. Le
alternative vengono da 4878 crop threshold-only; i template e le regole erano
fissati prima dello scoring. Tutti i 28 bracci sono disponibili nel JSON.

| Regola, box finali identici | Corrette /983 | Recuperate | Perse | Parole cambiate |
| --- | ---: | ---: | ---: | ---: |
| PARSeq baseline |368|0|0|0|
| Confidence PARSeq sulle alternative |383|26|11|351|
| CLIP immagine intera, parola raw |344|15|39|411|
| CLIP crop, parola raw |353|15|30|375|
| CLIP crop, parola scritta tra virgolette |362|14|20|363|
| CLIP crop, ensemble |361|13|20|365|
| CLIP crop, parola+descrizione |355|14|27|368|
| CLIP crop, descrizione scambiata |359|13|22|358|
| CLIP crop, residuo contestuale normalizzato |369|19|18|370|
| CLIP crop ampio, ensemble |360|21|29|393|
| CLIP crop ampio con lettere coperte, ensemble |347|23|44|437|
| Fusione midrank PARSeq+CLIP |378|16|6|256|
| Gate di accordo PARSeq/crop/crop ampio |370|2|0|9|

**Aggiungere direttamente la descrizione non aiuta il crop in questo test**:
355 contro 362 del template senza descrizione. È persino inferiore al controllo
con descrizione scambiata, 359. Sull'immagine intera il prompt contestuale, 355, è
vicino al template scritto, 353, comunque sotto la baseline, 368. Il residuo
contestuale recupera 19 parole ma ne perde 18: il +1 netto è esplorativo e non
evidenza di un guadagno affidabile. L'ampio controllo mascherato347 suggerisce
che la sola plausibilità del contesto circostante è insufficiente; il controllo
di copertura ha però anche un'alterazione visiva sintetica.

Il gate conservativo recupera `RumCo.` → `Rum` (`total_text:464`) e
`DECAIDIES` → `DECADES` (`total_text:989`). Le altre sette modifiche non
aggiungono parole esatte; zero danni osservati non garantisce zero danni su
altri dati. La confidence PARSeq da sola ottiene più recuperi complessivi;
la fusione CLIP, 378, è ancora sotto il controllo, 383. Perciò questo run non
giustifica adottare CLIP come correttore predefinito.

Anche l'identificazione di letture errate è più debole con CLIP. Su 482 anchor
con una sola associazione GT, 365 baseline corrette e 117 errate:

| Evidenza | AUROC corretto/errato | Confronti vinti, tutte160 coppie |67 coppie a un edit |
| --- | ---: | ---: | ---: |
| Confidence PARSeq |0.823|81.9%|82.1%|
| CLIP crop raw |0.580|57.5%|56.7%|
| CLIP crop scritto |0.615|66.9%|59.7%|
| CLIP crop ensemble |0.622|66.9%|59.7%|
| CLIP crop con descrizione |0.562|60.6%|53.7%|
| CLIP crop residuo contestuale |0.646|70.0%|67.2%|
| CLIP crop ampio ensemble |0.645|65.0%|70.1%|

Ci sono inoltre 688 anchor senza sovrapposizione a care GT e 3 con associazione
ambigua, esclusi soltanto dall'AUROC condizionale e conservati nelle metriche
principali. Le descrizioni menzionano letteralmente 160 dei 482 target associati:
non sono pura informazione semantica senza testo. Nel sottogruppo con target
menzionato, il template crop senza descrizione dà 134/160 letture corrette,
mentre il prompt contestuale 129/160; senza menzione dà 225/322 contro 223/322.
Questa suddivisione è post-hoc e non una nuova regola selezionata sul GT.

Il risultato pratico è usare CLIP, se ulteriormente validato, come **evidenza
secondaria con astensione dal cambiamento**, subordinata a supporto visivo e
PARSeq. Un futuro confronto B/16/L/14 deve tenere gli stessi pool e box;
una successiva calibrazione e verifica indipendente può stabilire un gate.
Le versioni con patch più piccole B/16 e L/14 hanno più posizioni visive a
224 pixel: è un motivo per testarle sui caratteri, non una garanzia di guadagno.
CLIPTER/CLIP4STR e le proiezioni di spelling indicano direzioni con adattamento
appreso. Pesi/proiezioni già addestrati potrebbero essere usati congelati senza
training locale, ma vanno qualificati come componenti diversi; un nuovo
fine-tuning locale cambierebbe il vincolo training-free. Il
[codice ufficiale delle proiezioni](https://github.com/joaanna/disentangling_spelling_in_clip)
indica artefatti preaddestrati e una proiezione normalizzata delle feature.
Disponibilità, provenienza e compatibilità con il B/32 locale restano da
verificare; questa batteria non li ha valutati.

Artefatti locali:

* Report completo: `/tmp/clip-word-verification-train-v1-final/vit_b32/train125/results.json`,
  SHA256 `e9a6db80ffee2c33c3d2120d42ce12dcc0b87c43029647917ff4bb045c0c3e2e`.
* Configurazione risolta, provenienza e input: stessa cartella,
  `resolved_configuration.json`; prompt/score/scelte per immagine in `scores/`.
* Stato del confronto dei checkpoint e fallimenti:
  `/tmp/clip-word-verification-train-v1-final/comparison_status.json`.
* Smoke reale CPU: `vit_b32/smoke-1/results.json`. Score e scelte della prima
  immagine sono identici tra smoke e run completo indipendente. Nella cartella
  smoke è anche conservato `failure.json` della prova CUDA fallita; il device
  nel record separa chiaramente questa prova dal risultato CPU riuscito.

Le 43 unit test mirate passano. Il ramo OOM è verificato mediante simulazione.
Lo scoring CPU completo ha impiegato 303.1 secondi cumulati, float32/batch16,
senza misure di VRAM e senza pretesa di tempo end-to-end della pipeline.
I run B/16,L/14 e GPU restano **non eseguiti**, con preflight falliti conservati.

La successiva [batteria innovativa](CLIP_INNOVATIVE_VERIFICATION.md) aggiunge
confronti prima/dopo copertura, glyph sintetici, ordine delle lettere e
ingrandimento del testo, con un unico comando host per tutti e tre i checkpoint.
Le due configurazioni e i loro risultati restano separati.
