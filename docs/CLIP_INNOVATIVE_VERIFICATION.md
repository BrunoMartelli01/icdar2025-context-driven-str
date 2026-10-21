# Nuovi utilizzi di CLIP e comando per il confronto su GPU

Questa è una batteria **approximate reproduction**, su Total-Text TRAIN125 già
usato per sviluppo. Confronta tre checkpoint congelati su identiche parole
PARSeq e box; non è una valutazione TEST indipendente o una Table5 riprodotta.
La [batteria precedente](CLIP_WORD_VERIFICATION.md) e i relativi risultati
rimangono separati.

## Un comando dal terminale dell'host

```bash
bash /extra1/bmartelli/projects/icdar2025-context-driven-str/scripts/run_clip_verification_suite_host.sh
```

Il comando esegue i test, verifica accesso CUDA e input scientifici, riusa il
ViT-B/32 locale e scarica gli snapshot fissati ViT-B/16 e ViT-L/14 mancanti.
Non installa pacchetti. I processi usano una GPU, float32 e batch16; i modelli
si caricano separatamente. Si registrano GPU effettiva, forme degli input,
peak allocated/reserved e fallimenti/OOM. Non viene cambiata automaticamente
precisione, device, modello o batch dopo un errore.

I download sono limitati ai file del modello/processore/tokenizer richiesti,
senza scaricare copie TensorFlow/Flax dei pesi. Il downloader verifica commit
completo, ETag LFS SHA256 e bytes dei pesi. Anche per B/16, il cui checksum
upstream nel vecchio config era irrisolto, viene conservata la ricevuta
verificata; il vecchio config non viene mutato. In modalità `--offline`,
un checksum upstream assente resta esplicitamente non verificato.

Log: `/tmp/clip-verification-host-suite.log`, personalizzabile con
`CLIP_SUITE_LOG`. I risultati sono in
`/tmp/clip-verification-host-suite/<protocol-hash>/`:

* `COMPARISON.md`: tabella delle regole per ogni modello;
* `comparison.json`: confronto completo e provenienza dei download;
* `classic/`: tutti i confronti v1, inclusi prompt con descrizione;
* `innovative/`: nuovi confronti, con prompt/score/scelte e hash dei rendering;
* `downloads/`, `hardware.json`, `launch.json`: provenienza e configurazione;
* `suite_failure.json` e/o `failure.json`: errori propagati, quando presenti.

Il percorso viene stampato nella riga finale `COMPLETED`. Il codice,
configurazione, software, device e limite di immagini definiscono un nuovo
albero di output; una ripetizione dello stesso protocollo riusa gli intermedi
compatibili. Si preservano i vecchi esperimenti CPU. Per cambiare il percorso
di output usare `--output-root /percorso/scelto`.

Per un controllo locale senza download e con il modello già disponibile:

```bash
bash scripts/run_clip_verification_suite_host.sh --models vit_b32 --device cpu --offline --limit 1
```

Per uno smoke GPU iniziale sui tre checkpoint, aggiungere `--limit 1` al
comando completo; i risultati restano etichettati smoke. Il run senza limite
usa tutte le 125 immagini e tutte le 983 care GT. Lo script non apre TEST.

## Ipotesi implementate

1. **Dipendenza dalla regione scritta.** Per ogni candidata `w`:
   `score(w) = cos(crop_ampio,w) - cos(crop_ampio_con_target_coperto,w)`.
   Un punteggio alto misura quanto la presenza del target sostiene quella
   candidata. Le alternative condividono entrambi gli input. Si copre l'intero
   box, quindi anche lo sfondo al suo interno: non è una prova causale che il
   modello abbia riconosciuto ogni lettera. Si provano raw/quoted/ensemble e
   residuo contestuale, oltre a un gate di accordo con PARSeq.
2. **Parole candidate come immagini.** Ogni stringa PARSeq completa è scritta
   in DejaVu Sans, Serif e Mono, con nero su bianco e bianco su nero. Si confronta
   il crop reale con la media normalizzata delle feature visuali di queste
   immagini. Anche la candidata entra nel ramo visivo, evitando la sua
   tokenizzazione testuale per questo punteggio. Non è garantita la rimozione
   dei bias semantici: l'encoder visivo di CLIP può rappresentare il significato
   del testo. Font/forma/lunghezza possono diventare scorciatoie.
3. **Ordine delle lettere.** Si sottrae la similarità della stessa stringa
   renderizzata al contrario: medesimi caratteri, font e polarità, diverso
   ordine. `STOP`/`POTS` è un esempio di controllo, non un nuovo output OCR.
   Le stringhe invertite non entrano mai nel pool selezionabile. Palindromi e
   caratteri ripetuti identici al loro inverso producono zero evidenza d'ordine.
   Non si ottiene così un puro misuratore di spelling: significato e kerning
   possono comunque cambiare.
4. **Più spazio per il testo nell'input.** Il crop viene anche deformato fino
   a riempire il quadrato della risoluzione ufficiale CLIP, con resize bicubico.
   Con parole molto larghe, il canvas quadrato precedente lascia al testo pochi
   pixel in altezza. Si prova l'ingrandimento con template testuali e rendering
   visivi, separatamente dal canvas che conserva le proporzioni. La deformazione
   è un'assunzione sperimentale, non una modifica del preprocessore STRICT.

La selezione conserva sempre i box e le stringhe raw PARSeq. Il gate dei
glyph richiede accordo tra confidence PARSeq, effetto della regione scritta e
confronto visuale multiview, con margini fissi. Nessun GT o testo del captioner
viene usato per generare nuove candidate. Si salvano testo, font/checksum,
polarità, geometria e hash RGB per ricreare ogni probe sintetico. Il charset
del rendering è ASCII stampabile: input vuoti, invisibili o non supportati
falliscono esplicitamente, senza glyph sostitutivi.

Il confronto completo dell'host è ora terminato per **B/32, B/16 e L/14**:
[risultati verificati e interpretazione](CLIP_VERIFICATION_RESULTS.md).
Il report è `/tmp/clip-verification-host-suite/6920d0b0c030/comparison.json`.

La batteria completa ha **40 regole per checkpoint**, inclusi tutti i 28
controlli precedenti. Quella solo controfattuale ha 33 regole e non dichiara
di aver valutato rendering o stretching. Il protocollo, le assunzioni e le
deviazioni hardware sono in
[config v2](../configs/approximate/clip_innovative_verification_train_v2.json)
e `CLIP-INNOVATIVE-VERIFICATION-V2` in `PAPER_AMBIGUITIES.md`.

## Fonti e ulteriori idee

[FDP, ACM MM2024](https://arxiv.org/abs/2408.00441) identifica sia la scala
percettiva limitata del testo sia l'intreccio tra concetti visuali e semantici
come problemi di CLIP per scene-text retrieval. Include componenti apprese:
le nostre prove di scala/spelling sono ipotesi ispirate da questa diagnosi,
non una riproduzione della sua architettura o dei suoi risultati.

[T-MARS, ICLR2024](https://arxiv.org/abs/2307.03132) maschera testo e ricalcola
similarità CLIP per filtrare dati di pretraining in cui il testo domina la scena.
Qui usiamo il confronto prima/dopo per studiare il contributo del target a
ciascuna candidata: compito e criterio di selezione sono differenti.

[Disentangling visual and written concepts in CLIP](https://joaanna.github.io/disentangling_spelling_in_clip/)
studia anche parole senza significato e proiezioni che separano spelling e
concetti visuali. Proiezioni ufficiali già apprese potrebbero essere usate
congelate, qualificando compatibilità, versione e checkpoint; non sono ancora
scaricate o integrate nella batteria. Un'altra estensione sarebbe confrontare
coppie di parole con un'unica lettera diversa e usare i patch visivi della zona
differente, ma il mapping fra patch CLIP e caratteri è ancora da definire.

## Verifiche già eseguite in questa sessione

**53 test passano**, inclusi integrità dei checkpoint scaricati, ordine/charset
dei rendering, pooling di font, pool immutabili, selezioni v1 preservate,
denominatori vuoti e OOM simulato e propagato. Un primo test aveva rilevato
un errore di parentesi nel nuovo ramo; è stato corretto prima dell'inferenza.
Lo script host completo ha terminato un vero smoke B/32 CPU, con tutte le
40 regole, in `/tmp/clip-verification-host-suite/47395c888c47/`. È un controllo
ingegneristico su una immagine, non prova del beneficio dei glyph.

La sola parte controfattuale è stata valutata su tutti i vecchi score B/32
TRAIN125, senza nuova inferenza e senza alterare i 28 controlli originali:

| Regola | Corrette /983 | Recuperate | Perse | Parole cambiate |
|---|---:|---:|---:|---:|
| Baseline PARSeq |368|0|0|0|
| Confidence PARSeq sulle alternative |383|26|11|351|
| Differenza prima/dopo, prompt scritto |362|14|20|360|
| Differenza prima/dopo, residuo contestuale |366|19|21|378|
| Accordo PARSeq + effetto della regione scritta |372|4|0|23|

Il gate recupera 4 parole senza perdite osservate; gli altri 19 cambiamenti
non aumentano l'exact matching. La scelta libera del massimo score resta
insufficiente e la confidence PARSeq raggiunge un totale maggiore. Sulle
67 coppie a un edit, l'effetto del prompt scritto sceglie la corretta nel
68.7%, contro 59.7% del prompt scritto sul crop nel confronto precedente e
82.1% del controllo PARSeq. Sono osservazioni su dati di sviluppo correlati,
non significatività o garanzia di sicurezza delle correzioni.

Report completo:
`/tmp/clip-innovative-verification-v2/vit_b32/counterfactual/results.json`,
SHA256 `0901ef57307bac7b0b361ad0b4844906cfc5d4a1d3e273609b5448075ab138c6`.
Metriche, matching e denominatori sono quelli della v1: exact casefold con
punteggiatura conservata, matching uno a uno bboxIoU>=0.5 e tutte le care GT.

Il preflight CUDA fallisce nella sessione Codex; il run dell'host ha invece
completato tutte le 40 regole per ognuno dei tre checkpoint su TRAIN125,
con una RTX 3090. La verifica CPU ha ricalcolato tutti i sei report dai
750 shard e controllato che codice, input, selezioni e output originali
restino coerenti. L/14 con fusione PARSeq/CLIP ottiene 386/983; con candidate
renderizzate 383/983; con il gate conservativo PARSeq/crop/crop ampio 376/983,
otto recuperi e zero danni osservati. La descrizione aggiunta al prompt del
crop peggiora l'exact matching per tutti i checkpoint. Questi sono risultati
di sviluppo APPROXIMATE, non conferma indipendente o qualifica della RTX 3070.
