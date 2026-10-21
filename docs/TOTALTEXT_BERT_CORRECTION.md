# BERT dopo PARSeq: Total-Text letterbox

Esperimento **APPROXIMATE / approximate reproduction**, diagnostico TRAIN-holdout.
Non è una riproduzione strict, una misura Table 5 o un risultato sul TEST.

## Percorso effettivo del progetto

Il modello in `models/unet_adv.py` segmenta l'immagine. I target v4 sono
costruiti dalle regioni delle singole parole; la trasformazione letterbox
preserva il rapporto d'aspetto, con input 512×512 e maschera nativa 256×256.
Il localizzatore estrae contorni esterni con OpenCV e riporta i rettangoli
attraverso l'inversa della letterbox all'RGB originale. PARSeq riconosce ogni
crop e conserva testo, probabilità dei token e checksum. Il nuovo componente
lavora **dopo** questi output: non cambia maschere, contorni, crop o PARSeq.

Il riferimento scelto dopo l'indicazione dell'utente «ultima variante
letterbox» è il controllo a una testa `word_f1_control`, selezionato per F1
di localizzazione sulla validazione TRAIN, epoch 68. Questa identifica una
scelta precisa tra le diverse varianti presenti; non deriva dalla migliore
accuracy OCR osservata. Il ramo con testa ausiliaria di separazione è un altro
modello e non entra in questo confronto.

Fonte: `context-str-runs/totaltext-letterbox-word-f1-separator-parseq-val-v1/`.
Riutilizzo OCR dichiarato dal valutatore originale:
`context-str-runs/totaltext-v4-image-frequency-ablation-v1/validation-pad0-parseq-v1/ocr/`.
Entrambe le directory devono essere indicate esplicitamente alla preparazione.
Il lettore verifica report, protocollo, maschere, checkpoint, bbox, hash RGB,
testo e record OCR. Cache discordanti o mancanti bloccano il run.

Il protocollo preesistente usa soglia maschera 0,5, padding zero e contorni senza
filtro d'area/cap top10. Sono scelte pratiche già presenti, diverse da Algorithm 1.
La selezione word-F1 e la letterbox sono anch'esse approssimazioni. Le sezioni
3.3–3.5 del PDF locale specificano localizzazione, BLIP-2 e scoring
MPNet/Levenshtein con fallback DeepSolo; non specificano un correttore BERT.
I percorsi strict, MPNet, BLIP-2 e DeepSolo non sono modificati.

## Modello e ricerca

Modello iniziale: **google-bert/bert-base-uncased**, circa 110M parametri,
revisione `86b5e0934494bd15c9632b12f734a8a67f723594`, FP32, congelato.
La revisione è verificata nell'[API ufficiale del modello](https://huggingface.co/api/models/google-bert/bert-base-uncased).
Il run scarica soltanto i pesi safetensors e i file di configurazione/tokenizer
di quella revisione e registra SHA256 di ciascuno. Nessun training locale,
quantizzazione, cambio di precisione automatico o installazione di pacchetti.

Fonti e decisioni:

| Fonte primaria | Idea utilizzata o limite |
| --- | --- |
| [Salazar et al., Masked Language Model Scoring, ACL 2020](https://aclanthology.org/2020.acl-main.240/) | Valutare alternative tramite likelihood pseudo-logaritmica di un MLM congelato. Il paper tratta ASR/NMT; il trasferimento a OCR è un'ipotesi. |
| [Kauf e Ivanova, A Better Way to Do Masked Language Model Scoring, 2023](https://arxiv.org/abs/2305.10588) | Mascherare il sottotoken corrente e tutti i successivi della stessa parola, evitando che BERT ricopi sottotoken futuri del candidato. |
| [Fang et al., ABINet, CVPR 2021](https://arxiv.org/abs/2103.06495) | Separare evidenza visiva e linguistica, con contesto bidirezionale. Questo correttore non implementa o riproduce ABINet. |
| [Jayanthi et al., NeuSpell, EMNLP 2020](https://aclanthology.org/2020.emnlp-demos.21/) | BERT già adattato alla correzione ortografica inglese è una futura alternativa utile; la sua validazione non dimostra efficacia su Total-Text. |

Ho esaminato anche [`murali1996/bert-base-cased-spell-correction`](https://huggingface.co/murali1996/bert-base-cased-spell-correction),
revisione visibile `d2a5bbccc41a0f4ff5e7c16e1c3b8d96ba8883b2`.
Il [config originale](https://huggingface.co/murali1996/bert-base-cased-spell-correction/blob/main/config.json)
ha una classificazione a 100.002 etichette. Lo
[snippet degli autori](https://github.com/neuspell/neuspell/blob/master/scripts/huggingface/huggingface-snippet-for-neuspell.py)
media le rappresentazioni dei sottotoken e applica una testa word-level,
con mapping esterno `vocab.pkl`. Caricarlo come masked LM inizializzerebbe una
testa diversa e non riprodurrebbe il correttore. Per il primo esperimento ho
scelto il BERT-base MLM originale, il cui decoder è direttamente verificabile.
**Non è un BERT fine-tuned per OCR.** La testa e il mapping NeuSpell richiedono
un adattatore e una qualificazione distinti, non una sostituzione silenziosa.

## Esperimento congelato

Config: `configs/approximate/totaltext_v4_bert_correction_v1.json`.
Confronto tra `word_only` e `local_ocr`; se viene fornita una caption associata
a ogni immagine, si aggiunge `local_ocr_caption`. Tutte le condizioni valutano
gli stessi candidati per isolare l'effetto del contesto sullo scoring.

- KEEP è sempre il testo originale PARSeq, con capitalizzazione originale.
- Si correggono soltanto parole ASCII di almeno tre caratteri. Numeri,
  punteggiatura, stringhe vuote, crop multiword e nomi con maiuscole interne
  restano invariati. Unchanged non significa riconosciuto correttamente.
- Probabilità minima nativa PARSeq ≥0,95: KEEP. È una protezione euristica,
  non una probabilità calibrata di correttezza.
- Alternative dal vocabolario BERT a parola intera e dalle parole osservate
  nei vicini/caption: distanza Levenshtein ≤1 per parole corte, ≤2 da sei
  caratteri, distanza relativa ≤0,3; massimo 24 candidati incluso KEEP.
  L'ordine è distanza, presenza nel contesto, ordine lessicografico. Il limite
  può escludere una parola corretta; non usa vocabolari GT o dizionari Total-Text.
- Vicini: sovrapposizione verticale ≥0,5, gap orizzontale ≤8 altezze e massimo
  tre parole per lato. Testi originali congelati, nessuna correzione iterativa
  che propaghi una parola già modificata agli altri crop.
- Caption: primi 80 termini whitespace, dichiarati come budget; scena nel
  primo segmento BERT e scritte locali nel secondo. Input oltre 256 token e
  target sconosciuti/speciali causano astensione esplicita, senza troncamento
  ulteriore o riempimento dell'output.

Score del candidato: media dei log-score dei suoi sottotoken con mascheramento
PLL-word-l2r. Sono salvati anche somma, log-score individuali e numero di token.
Il paper di Kauf/Ivanova usa una somma a livello di frase: la media sul solo
target è una nostra **ASSUMPTION**, non un posterior OCR calibrato.

```
gain = mean_BERT(candidate) - mean_BERT(PARSeq) - 1.0 * edit_distance
correggi soltanto se gain > 1.5; KEEP sulle parità
```

Tutti i parametri sono nella config. Non sono ottimizzati sui risultati BERT,
non è prevista una ricerca sul TEST o promozione automatica.
Lo scoring conserva l'evidenza visiva tramite limiti agli edit e protezione
PARSeq; non possiede likelihood PARSeq delle nuove stringhe. Rischi residui:
nomi propri, lingue diverse dall'inglese, bias verso parole comuni, crop fusi,
contesto geometrico errato e testo inventato dalle caption.

## Esecuzione

Sul computer con accesso alla rete, nell'ambiente esistente:

```bash
bash scripts/run_totaltext_bert_correction_host.sh \
  --source-dir /extra1/bmartelli/context-str-runs/totaltext-letterbox-word-f1-separator-parseq-val-v1 \
  --reuse-ocr /extra1/bmartelli/context-str-runs/totaltext-v4-image-frequency-ablation-v1/validation-pad0-parseq-v1/ocr \
  --hf-cache /extra1/bmartelli/context-str-artifacts/huggingface \
  --output-dir /extra1/bmartelli/context-str-runs/totaltext-letterbox-bert-correction-v1 \
  --device cpu
```

Per la GPU utilizzare esplicitamente `--device cuda:0`; indisponibilità/OOM
interrompono il run. Il programma usa soltanto BERT durante la correzione.
Registra modello, dtype, batch delle mascherature, limite input, device,
software, seed, determinismo, tempo di caricamento e di correzione e picchi
CUDA allocated/reserved. Su CPU i campi VRAM sono `null`, non zero misurato.
Il tempo non comprende U-Net, contorni, PARSeq, caption o DeepSolo.
Il caricamento accetta soltanto le eventuali chiavi inattive pooler/NSP
elencate nella config e registra tutte le chiavi; missing/mismatch nella testa
MLM o nel backbone sono fatali.

Gli stadi sono `prepare`, `fetch-model`, `infer`, `evaluate`. Preparazione e
valutazione possono essere ripetute; la ripresa dell'inferenza verifica
provenienza e checksum di ogni shard. Cambiamenti di codice, configurazione,
device o caption richiedono una nuova directory/versione.
Il lanciatore scarica pesi, **non installa o aggiorna dipendenze**.

`infer` legge `inputs.json`, mai `evaluation_labels.json`. Sample ID e hash
servono solo al collegamento degli artefatti, non entrano nel tokenizer BERT.
Ogni crop conserva bbox, ordine, testo originale, alternative, score e scelta.
`results.json` viene prodotto soltanto dopo tutte le immagini/condizioni;
gli errori sono salvati in `errors/` e propagati. Nessun campione fallito viene
tolto dal denominatore.

Caption opzionali, da passare con `--captions FILE`, schema:

```json
{
  "schema_version": 1,
  "model_id": "model-used-to-describe-Total-Text",
  "revision": "40 hexadecimal characters",
  "model_files_sha256": {"weights-file": "64 hexadecimal characters"},
  "generation": {"all_decoding_parameters": "resolved values"},
  "samples": [
    {"sample_id": "total_text:14", "image_sha256": "64 hexadecimal characters", "text": "Scene description"}
  ]
}
```

Richiesta copertura esatta delle 125 immagini, con checksum identici. Le
caption attualmente trovate nel repository per ICDAR13 non sono riutilizzabili.
Non sono state generate nuove caption Total-Text in questo intervento.

## Metriche e stato verificato

Aggregazione micro sulle **983 care words**, incluse le parole senza un crop.
Don't Care già esclusi dagli shard qualificati originali. Matching bipartito
massimo uno-a-uno tra bbox a IoU≥0,5 e testo exact Unicode casefold,
con punteggiatura conservata. Si riportano anche exact case-sensitive e
RapidFuzz casefold ratio≥0,8. Empty output non corrisponde a una GT nonempty.
Geometria e metriche sono riusate dal modulo di valutazione già presente;
non sono il protocollo ufficiale polygon spotting di Total-Text.

Recuperi/danni sono differenze tra gli insiemi di GT corrette prima/dopo;
la loro differenza deve coincidere con il miglioramento del totale exact.
Si conservano GT, prediction e indici per immagine nella sola evaluation.

Preflight reale su `word_f1_control`: **125 immagini, 949 crop, 983 parole**,
503 match geometrici a IoU≥0,5, **384 exact casefold (39,06%)**, 372 exact
case-sensitive, 475 fuzzy. 555 output hanno forma eleggibile; dopo la
protezione PARSeq restano 137 possibili target, prima della disponibilità dei
candidati BERT. Questo è il controllo originale, non un risultato BERT.

46 test CPU passati, inclusi esecuzione di una piccola istanza BERT casuale
solo come fixture ingegneristica, mascheramento dei sottotoken, isolamento GT,
config namespace, soglie, checksum, cache storica, astensioni, matching,
denominatore e regressioni PARSeq/metriche. Il primo preflight ha esposto la
serializzazione dell'hash upstream; il secondo la cache OCR esterna. Entrambi
sono stati corretti rispettando il contratto originale e testati.

Preflight salvato in `/tmp/totaltext-letterbox-bert-preflight-v3/`.
Al preflight i pesi BERT mancavano; l'utente ha poi completato il lanciatore
sul computer con rete. L'inferenza reale e la verifica sono ora completate
su CPU. CUDA, VRAM GPU, effetto caption e TEST restano non misurati.
Il TRAIN-holdout è già stato esplorato in altri esperimenti, quindi non
costituisce validazione indipendente.

## Risultati reali completati — 6 ottobre 2026

Il report completo originale è
`/extra1/bmartelli/context-str-runs/totaltext-letterbox-bert-correction-v1/results.json`,
SHA256 `a7ae0a603323c4d2d6696b7383dbc6c6dd2646432f8200f4e52e8f674ee788ef`.
Verificati 250 shard (125 per condizione), hash, crop, gate e metriche
ricomputate sul denominatore originale. Nessun errore runtime o astensione
per token limit/UNK; nessun sample escluso. Il backbone e la testa MLM non
presentano chiavi mancanti o mismatch; soltanto le quattro chiavi inattive
pooler/NSP previste sono escluse. Pesi safetensors SHA256:
`68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`.

| Regola | Exact / 983 | Recuperate | Rovinate | Crop modificati | Secondi di correzione CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| PARSeq congelato | 384 | 0 | 0 | 0 | — |
| BERT `word_only`, protezione ≥0,95 | 381 | 0 | 3 | 10 | 4,41 |
| BERT `local_ocr`, protezione ≥0,95 | 383 | 0 | 1 | 8 | 4,88 |
| BERT `word_only`, protezione ≥1,0 | 341 | 1 | 44 | 62 | 13,31 |
| BERT `local_ocr`, protezione ≥1,0 | 363 | 3 | 24 | 42 | 16,38 |

BERT senza contesto rovina `Kong → King`, `Attire → Lattice` e
`MOTEL → MODEL`; il contesto locale preserva le prime due ma trasforma
`MOTEL → HOTEL`. Altri crop modificati non hanno match GT a IoU≥0,5 e non
costituiscono recuperi/danni exact in questa metrica. Questo non li rende
semanticamente corretti e non è prova che tutti i cambiamenti siano innocui.

### Diagnosi e ablation della protezione

L'audit GT-assisted trova un limite teorico di 420 parole esatte scegliendo
fra i candidati già salvati, con uno-a-uno per crop. Le 36 parole aggiuntive
sono un **oracle**, non un selettore utilizzabile. In 29/36 casi la protezione
native-confidence blocca la correzione, inclusi `MERICAN → AMERICAN` e
`WORL → WORLD`; negli altri sette BERT mantiene il testo originale. Probabilità
elevata sui caratteri prodotti non verifica che manchino altri caratteri.

Per isolare questa causa ho eseguito una **ablation post-hoc sul TRAIN**,
config `configs/approximate/totaltext_v4_bert_confidence_ablation_v1.json`.
Cambia soltanto `protect_min_token_probability` da 0,95 a 1,0: il valore
esattamente saturo resta protetto, tutte le altre regole sono identiche.
Non è un protocollo scelto prima del primo run né una validazione indipendente;
nessun tuning, grid o selezione sul TEST. Il run originale è preservato.

Report completo dell'ablation:
`/tmp/totaltext-letterbox-bert-confidence-ablation-v1/results.json`, SHA256
`6919040e863c2b18f0727a0458957ed7e9e0bf4c244ba5545dc3c9f3534de7d5`.
Con contesto recupera `EVER → EVERY`, `Oper → Open` e `OPICAL → TROPICAL`,
ma rovina 24 parole prima corrette. Il netto −21 mostra che allentare la
protezione non risolve il problema di selezione in questa configurazione.
La maggiore plausibilità linguistica non fornisce evidenza sufficiente che
una scritta contenga realmente il candidato.

Entrambi i run usano BERT FP32, CPU a quattro thread, batch16; le latenze
escludono caricamento e tutti i modelli upstream. Picchi VRAM sono `null`,
non una misura GPU. Geometria invariata: 503 match IoU≥0,5 in ogni condizione.
46 test CPU/regressione passati nuovamente. Non viene promossa nessuna regola
di correzione sulla base di questi risultati negativi. Resta da qualificare un
correttore spelling-fine-tuned o un contesto caption realmente riferito a
Total-Text, oltre a una futura validazione indipendente.

Audit machine-readable in
`/tmp/totaltext-letterbox-bert-completed-audit-v1.json` e
`/tmp/totaltext-letterbox-bert-confidence-ablation-audit-v1.json`.
