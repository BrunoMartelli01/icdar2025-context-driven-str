# Correzione contestuale OCR: confronto TRAIN congelato v1

## Scopo e confini

PRACTICAL_APPROXIMATION / modalità APPROXIMATE. Nuovo diagnostico, non sostituisce
Algorithm1, practical-remap-v1/v2, MPNet/scoring, routing o DeepSolo. Non è Table5.
Nessuna nuova segmentation, PARSeq, caption o training. Nessun TEST.

Si tiene fisso V1 come controllo già auditato (502/849 exact,578/849 fuzzy), non
come affermazione che sia il migliore per ogni metrica. Cambiare contemporaneamente
localizer e correttore impedirebbe di isolare l'effetto del modello linguistico.
L'oracle GT-dependent che recupera78/80 non entra nel generatore o nel prompt.

## Modelli e input congelati prima dei risultati

| Modello | Revisione | Precisione |
|---|---|---|
|google/flan-t5-base|7bcac572ce56db69c1ea7c8af255c5d7c9672fc2|FP32|
|Qwen/Qwen2.5-0.5B-Instruct|7ae557604adf67be50417f59c2c2f167def9a775|BF16|
|Qwen/Qwen2.5-1.5B-Instruct|989aa7980e4cf806f80c7fef2b1adb7bc71aa306|BF16|

Fonti ufficiali: [FLAN-T5](https://huggingface.co/google/flan-t5-base),
[Qwen0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[Qwen1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/commit/989aa7980e4cf806f80c7fef2b1adb7bc71aa306).
Revisioni verificate tramite repository/config ufficiali, non mutable main al run.

Per ogni modello si confrontano tre condizioni, una chiamata per immagine:

1. `ocr_only`: tutti i testi dei crop e bbox in ordine geometrico.
2. `ocr_caption`: stessi input più caption InstructBLIP completa già cached.
3. `caption_only`: estrazione del testo dichiarato visibile dalla sola caption;
   controllo senza OCR e senza corrispondenza spaziale.

Totale massimo2061 generazioni (3 modelli x3 condizioni x229 immagini).
Modelli caricati uno alla volta nell'ambiente principale esistente. Nessuna
installazione/upgrade. Download solo snapshot necessari in cache HF persistente,
soli safetensors/tokenizer/config, non copie PyTorch/TF/Flax. Circa5GB di nuovi
pesi se nessuno è già cached; nessun salvataggio duplicato di immagini/crop.

Config: `configs/approximate/contextual_ocr_correction_train_v1.json`.
Greedy,beam1,max768 nuovi token,max2048 token input,batch1,seed0. Nessun tuning
dopo risultati. Input oltre limite non troncati di nascosto: astensione esplicita.
Le differenze FP32/BF16 sono registrate come HARDWARE_DEVIATION.

## Prompt, output e protezioni

- I prompt completi sono nella config e salvati con ogni generazione.
- Testi dei blocchi e caption sono dati, mai istruzioni da eseguire.
- Non passare GT, sample ID, percorsi, risultati dell'oracle o esattezza precedente.
- Non inventare confidenze: l'input congelato usato qui espone bbox e testo,
  non confidenze calibrate. Non aggiungere una stima artificiale.
- Output JSON con esattamente gli ID originali e azioni keep/correct/uncertain.
- Keep/uncertain devono mantenere identico il testo. Nessun ID nuovo, fusione
  di ID, riempimento di blocchi originariamente vuoti o cancellazione con correct.
- Un output malformato viene conservato con errore e astensione esplicita:
  per l'immagine si mantengono tutti gli OCR originali; caption_only produce
  output vuoto. Non sparisce dal denominatore e non si riprova con altri prompt.
- Runtime/OOM/checkpoint mismatch interrompono il run con errore, non sostituiscono
  modello o precisione. L'astensione di schema è parte dichiarata del protocollo,
  non un errore runtime ignorato.
- Il prompt NON garantisce assenza di allucinazioni; JSON corretto non significa
  testo corretto. Per questo si misurano esplicitamente danni e surplus.

## Provenienza, cache e ripresa

Input da `context-str-runs/caption-routing-five-models-train-v1`, report SHA:
`389219133531817df915413a14ef01f727f0e6d710dc1d1eb177ad843e43b46b`.
Verificati protocol seal,hash input,229 caption,modello caption e immagini associate.
Il preparatore produce solo testo/bbox, nessun GT e nessuna lettura di immagini.

Per evaluation soltanto: audit
`artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json`, SHA
`82acfe8d860aee1a0f7bf70b9308a83aefb4abc17561105c74324d8b990fc56a`.

Protocollo e generazioni sono atomici e sigillati. Il restart verifica identità
di prompt,codice,modello/runtime e riusa ogni risultato valido, inclusi gli output
malformati già osservati. Un modello interamente completato non viene ricaricato.
Hash dei pesi/config/tokenizer salvati al primo caricamento. Non sovrascrivere
report con protocollo diverso; richiede un nuovo output directory/versione.

## Metriche e interpretazione

- Exact casefold e fuzzy>=.8 sulle849 GT con associazione storica del crop fissa:
  baseline502/578 riprodotta prima di qualsiasi GPU. Non scegliere un altro crop
  usando il nuovo testo o la GT. Parole senza crop restano nel denominatore.
- Conteggio errori corretti e parole corrette rovinate, per-word prima/dopo.
- Conteggi lessicali separati: multiset casefold dei token whitespace per immagine,
  duplicati limitati alla molteplicità GT. Recall,precision,surplus,missing.
- Caption-only ha solo metriche lessicali: NON confrontare queste come se fossero
  exact per crop. Surplus è proxy diagnostico, non prova di allucinazione, anche
  a causa di Don't Care e della tokenizzazione.
- Numero output invalidi e token-limit; tempi di correzione e VRAM per immagine;
  loading e numero parametri nel metadata modello.
- La latenza misurata NON comprende segmentation/PARSeq/caption/fallback. Nessuna
  conclusione sul risparmio end-to-end prima di misurare l'intero sistema.
- Nessuna promozione automatica: discutere guadagno netto, danni, surplus e costo.
  Il TRAIN è stato esplorato ripetutamente; generalizzazione richiede validazione
  indipendente dopo congelamento, non ottimizzazione sui suoi singoli errori.

## Comando PAPER

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
bash diagnostics/run_contextual_ocr_correction_host.sh /extra1/bmartelli
```

Foreground,progress visibile; usa `.venvs/context-str-main-py311` già esistente.
Solo l'utente esegue GPU. Codex può eseguire prepare,evaluate e unit test CPU.

Output persistenti sotto `artifacts/diagnostics/contextual-ocr-correction-train-v1/`:
`protocol.json`, `inputs.json`, `baseline.json`, `models/`, `generations/`,
`results.json`, `results.md`; errori in `errors/`. Il report finale viene prodotto
solo quando tutte le generazioni richieste sono presenti e verificate.

## Classificazione scientifica

`ASSUMPTION`: correzione generativa, prompt/schema/astensione, metriche diagnostiche
e scelta V1 come controllo. `HARDWARE_DEVIATION`: caricamento sequenziale,batch1,
FLAN FP32/Qwen BF16. `UNRESOLVED`: risultati Table5 comparabili e risparmio reale
di fallback. Il paper sezioni3.3/3.4 descrive localizzazione e contesto, non questa
correzione: nessuna modifica o pretesa di fedeltà del nuovo componente.

## Verifiche completate prima del run GPU

- 36 test CPU passati: isolamento prompt, schema/ID, astensione, corruzione cache,
  metriche recupero/danno/duplicati, regressioni caption-routing e joined-T3.
- Import reali Transformers4.47.1, AutoConfig Qwen2 e GenerationConfig superati;
  nessun modello caricato e nessuna chiamata CUDA eseguita.
- py_compile, bash -n e git diff--check superati.
- Prepare ed evaluation baseline eseguiti due volte: ripresa idempotente verificata,
  229 immagini/1203 crop/229 caption,502 exact/578 fuzzy.
- SHA protocol:`15d4eef2a0227969fd64944936b98f5e64cb898317f6b6432974cdffaf11c80a`.
- SHA inputs:`a1b7320caceabd56f3816fcfbb4a84bd2e2bf4a885c65b97d35a7eb947fc33a4`.
- SHA baseline:`2b700b21e3fe25e7568426869faa6f29ce26e0cad77acc1517a2d91f1a1b3c6c`.

I risultati dei tre correttori sono ancora PENDING_GPU, non zero e non misurati.
