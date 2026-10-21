# Guida al lavoro e ai confronti

Questa è la porta d'ingresso al workspace di riproduzione. Gli audit originali
rimangono le evidenze: l'indice e le tabelle sono viste derivate, senza nuova
inferenza, training o modifiche alle implementazioni scientifiche.

Per una nuova clone collaborativa partire da [GETTING_STARTED.md](GETTING_STARTED.md),
[REPOSITORY_MAP.md](REPOSITORY_MAP.md) e [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md).
La clone contiene codice e metadati; i report esterni richiedono un ripristino
esplicito. L'inventario storico completo è conservato in
[history/WORKSPACE_AUDIT_INDEX.json](history/WORKSPACE_AUDIT_INDEX.json).

## Dove iniziare

| Necessità | Fonte o comando |
| --- | --- |
| Macchina corretta, directory, ambienti, SSH | [HOST_OPERATIONS.md](HOST_OPERATIONS.md) |
| Confronti leggibili già preparati | [RESULTS_OVERVIEW.md](RESULTS_OVERVIEW.md) |
| Elenco completo dei confronti registrati | `python3 -m research groups` |
| Tutti i documenti e gli audit nel perimetro indicizzato | [AUDIT_INDEX.md](AUDIT_INDEX.md) |
| Posizioni delle sezioni, hash e metadati originali | [AUDIT_INDEX.json](AUDIT_INDEX.json) |
| Esito del refactor e verifiche di conservazione | [REFACTOR_AUDIT.md](REFACTOR_AUDIT.md) |
| Stato STRICT e qualifiche storiche | [STRICT_REPRODUCTION_STATUS.md](STRICT_REPRODUCTION_STATUS.md) |
| Specifica e chiarimenti degli autori | [REPRODUCTION_SPEC.md](REPRODUCTION_SPEC.md), [chiarimenti](<Author Clarifications and Reproduction Update.md>) |
| Decisioni, assunzioni e ambiguità | [PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md) |
| Identità delle fonti e dei pesi | [SOURCES.yaml](../provenance/SOURCES.yaml), [ARTIFACTS.yaml](../provenance/ARTIFACTS.yaml) |
| Nuove prove su istanze word e controllo delle duplicazioni | [WORD_INSTANCE_EXTENSIONS.md](WORD_INSTANCE_EXTENSIONS.md), [audit di novità locale](WORD_INSTANCE_NOVELTY_AUDIT.json) |
| Piccoli correttori RGB/maschere/probabilità: architetture e training | [MASK_PROBABILITY_REFINER.md](MASK_PROBABILITY_REFINER.md), [audit di novità locale](MASK_PROBABILITY_REFINER_NOVELTY.json) |

La parte iniziale di `STRICT_REPRODUCTION_STATUS.md`, il README upstream e alcune
note di implementazione descrivono milestone storiche. Non dedurre lo stato di
un esperimento recente dalla sola data di un vecchio paragrafo. Seguire audit,
receipt e report specifici, conservando eventuali contraddizioni come tali.

## Cercare senza caricare tutta la cronologia

La CLI richiede soltanto Python3 standard. Non importa PyTorch, non installa
pacchetti e non modifica gli esperimenti. Dalla directory del repository:

```bash
python3 -m research find 'native128'
python3 -m research section NATIVE128-GT-REVIEW-001
python3 -m research section TABLE5-SOURCE-METRIC-AUDIT-001
python3 -m research find 'DeepSolo' --limit 10
```

`section` restituisce il testo originale con percorso e numero di riga; se il
titolo non è univoco richiede una ricerca più precisa o `--file`. `find` cerca
letteralmente, senza interpretare una query come comando o espressione regolare.
Le normali ricerche `rg` escludono artifact ed environment tramite `.rgignore`.
Per un audit escluso usare `rg --no-ignore 'PATTERN' percorso/esatto/audit.json`.

## Confrontare i risultati

```bash
python3 -m research groups
python3 -m research compare native128-pad0 --sort f1
python3 -m research compare native128-padoriginal6
python3 -m research compare refinements --format json
python3 -m research compare multitask-geometry --format csv
```

Le sorgenti esterne richiedono radici esplicite. Su PAPER:

```bash
export CONTEXT_STR_RUNS_ROOT=/extra1/bmartelli/context-str-runs
export CONTEXT_STR_ARTIFACTS_ROOT=/extra1/bmartelli/context-str-artifacts
python3 -m research compare kernel-fresh
python3 -m research compare research-v2-heads
python3 -m research compare yolo-test-equal
python3 -m research compare clip-encoders-budget
python3 -m research overview
python3 -m research check
```

Sono accettate anche `--runs-root` e `--artifacts-root` **prima** del sottocomando.
Nessun percorso esterno viene indovinato. In assenza dei file o con hash diverso,
il comando fallisce indicando la fonte; non ricrea risultati tramite inferenza.
`groups` funziona anche senza dataset, checkpoint e directory dei run.

`--format json` conserva valori numerici completi, protocollo, riferimenti e
puntatori alle righe originali. Il CSV ripete per ogni riga protocollo, split,
classe e hash della fonte. Il Markdown arrotonda soltanto la visualizzazione.
Per salvare aggiungere `--output /percorso/nuovo.csv` oppure `.json` al confronto;
la creazione esclusiva impedisce di sovrascrivere fonti o esportazioni esistenti.

## Quali confronti sono disponibili

| Famiglia | Gruppi | Limite essenziale |
| --- | --- | --- |
| Checkpoint a uscita128 | `native128-pad0/1/2/4/6/8`, `native128-padoriginal6` | 15 checkpoint/14 stati; TEST300 esplorato; variante storica bbox-area150, cap10, nessun ignore sulle predizioni |
| Text, Kernel, Center | `multitask-geometry` | Geometria delle istanze native; non OCR né F1 pixel |
| Word research | `research-v1`, `research-v1-top10`, `research-v2-heads`, `research-v2-postprocess` | Controlli e budget distinti; epoch50 postprocessing separato dalle nuove teste |
| Crop e copertura | `crop-links`, `frozen-crops`, `learned-crops`, `refinements`, `center-votes` | TRAIN125 riutilizzato; ignore e coordinate specifici della suite |
| Gap e Kernel-only | `gap`, `kernel-extracted`, `kernel-fresh` | Estrarre la testa di una rete joint e addestrare una nuova rete sono prove diverse |
| YOLO e router | `yolo-test-equal`, `yolo-train-equal`, `yolo-test-transferred` | Matching lessicale senza IoU; budget descrittivo distinto da soglia trasferita |
| Encoder CLIP specializzati | `clip-encoders-budget`, `clip-encoders-threshold` | Famiglia prompt `margin_scene_en` fissata, non i diversi winner di ogni encoder |
| GT crop | `gt-crop-diagnostic` | Oracle diagnostico, escluso dalla panoramica dei metodi di inferenza |

Il registro iniziale contiene **26 gruppi / 225 righe** selezionate da report
strutturati ispezionati. Non ogni audit è un risultato numerico importabile:
fonti mancanti, risultati storici non rieseguibili, qualifiche, fallimenti e
proposte restano consultabili nell'indice e nei documenti originali. Le altre
righe dei report sorgente restano disponibili e non vengono eliminate.

Non aggregare i gruppi in una classifica globale. Precisione, recall, F1 e
`TN0=TP/(TP+FP+FN)` vengono ricavati soltanto dai conteggi del protocollo
dichiarato, verificando le rate originali quando presenti. `None`/`UNRESOLVED`
indica una divisione senza denominatore; non viene trasformata in successo.
Le predizioni vuote contano secondo il valutatore originale, non vengono
rimosse dal comparatore. `images` viene verificato solo quando disponibile
nel riepilogo sorgente; non viene inventato negli altri riepiloghi.

Le colonne VRAM di `multitask-geometry` sono picchi **training** originali;
i secondi di `clip-encoders-*` riguardano l'export delle feature. Non sono
tempi end-to-end confrontabili. Modello, precisione, batch, shape, checkpoint,
device e altre misure restano nei report originali identificati dall'hash.
Non viene fabbricato un benchmark di latenza/FLOP quando manca un protocollo comune.

## Aggiornare senza perdere evidenze

1. Leggere la fonte specifica, il relativo audit e la sezione del paper prima
   di cambiare un componente scientifico. Un confronto descrittivo non autorizza
   la promozione automatica di un modello nella pipeline.
2. Conservare report, configurazione risolta, dati per campione e qualifiche.
   Aggiornare `research/experiments.json` solo dopo aver verificato protocollo,
   provenienza, SHA256, split, denominatori, budget e selezione delle righe.
   Una nuova comparazione deve restare separata se cambia uno di questi confini.
3. Eseguire `python3 -m research check` con le radici necessarie. Ogni errore
   rimane un errore; il comando non salta le righe fallite per produrre una tabella.
4. Per la panoramica creare un nuovo export con `overview --output`, confrontarlo
   con `RESULTS_OVERVIEW.md`, quindi aggiornare consapevolmente la vista derivata.
   `overview --check docs/RESULTS_OVERVIEW.md` ne verifica la freschezza.
5. Dopo gli aggiornamenti documentali rigenerare l'indice:

```bash
python3 -m research catalog
python3 -m research catalog --check
```

Per mantenere anche le fonti esterne nell'indice usare le stesse due variabili
PAPER dei comandi precedenti. Il catalogo registra radici selezionate, profondità
massima ed esclusioni; non dichiara di indicizzare ogni file di ogni cache o disco.
Gli hash del catalogo verificano la freschezza dei documenti, non le affermazioni
scientifiche dei loro autori. L'indice ignora se stesso, evitando dipendenze circolari.

Decisione di reporting: `REPORTING-INDEX-001` in `PAPER_AMBIGUITIES.md`.
