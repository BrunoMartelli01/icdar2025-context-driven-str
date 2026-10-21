# Refactor di navigazione e reporting — 17 ottobre 2026

Scelta di reporting classificata `ASSUMPTION`, `REPORTING-INDEX-001`.
I risultati proiettati restano **approximate reproduction** con i protocolli
originari. Il refactor non modifica modelli, valutatori, checkpoint o ambienti.

## Problemi rilevati e interventi

| Problema osservato | Intervento |
| --- | --- |
| Ricerche generiche attraversano milioni di artifact | `.rgignore` e scoperta documentale con radici/profondità/esclusioni dichiarate |
| Registro ambiguità di oltre 560KB, storia distribuita in molti audit | Indice con hash, sezioni e righe; `find` e `section` leggono le fonti originali |
| Vecchie milestone descrivono come pendenti esperimenti poi completati | Guida per seguire la fonte specifica senza cancellare i paragrafi storici |
| “V1” designa pipeline e reti diverse | Gruppi espliciti con percorso del report, protocollo e identità delle varianti |
| Accuracy, membership, F1 pixel, geometria e word OCR confusi | Tabelle separate per unità/metrica; conteggi e denominatori visibili |
| Copia manuale di numeri e percorsi non verificati | Registro di JSON pointer e SHA256; fallimento esplicito su evidenza assente o incoerente |
| Host di esecuzione implicito | Istruzioni permanenti in AGENTS e guida locale/SSH per PAPER |

La raccolta comprende documenti in `docs/`, `provenance/`, audit/report negli
artifact locali e nelle due radici esterne dichiarate. Il perimetro preciso e
gli hash sono nel catalogo. Cache, campioni e file fuori profondità non sono
qualificati come audit indipendenti. Nessun risultato storico viene promosso
perché possiede un receipt che riporta PASS.

## Conservazione verificata

Confronto con gli hash acquisiti prima delle modifiche: **123 documenti
scientifici/provenance preesistenti identici**. README e AGENTS ricevono solo
aggiunte di navigazione e host. La cronologia delle ambiguità è conservata:
durante il lavoro un'altra attività ha inserito il record `WORD-GT-003`
(3.021 byte), che non è stato alterato o rimosso da questo refactor.

Rimuovendo soltanto quell'inserimento concomitante e la nuova appendice
`REPORTING-INDEX-001`, il contenuto ricostruito coincide esattamente con lo
SHA256 iniziale:
`bb2b3e4da65c3e9e578a880ce679464632406006090dccb7ae199f86913153a9`.
Le aggiunte concomitanti di implementazione/config/test GT V3 sono estranee
al refactor e sono state lasciate intatte.

## Verifica eseguita

- **51 test** del nuovo reporting/indice superati nell'ambiente main esistente.
  Coprono fonti mancanti/cambiate, modalità, popolazioni, contatori impossibili,
  formule, rate non finite, recuperi/perdite, pointer, path, fonti esterne,
  esportazione senza sovrascrittura, pruning delle cache e freschezza degli indici.
- **26 gruppi / 225 righe** reali verificati sui report salvati: SHA256,
  qualifiche dichiarate, campi e popolazioni attese, rate originali disponibili.
  Questa verifica non riesegue il matching per campione o qualifica i checkpoint.
- Smoke della CLI con Python standard senza site-packages, query/sezioni reali,
  panoramica, export JSON/CSV e controllo degli indici.
- Diff delle modifiche ispezionato; `git diff --check` superato.

Il primo giro dei test ha avuto un fallimento: l'asserzione contava anche una
variante chiamata `variant` come intestazione Markdown. Corretto il conteggio
del test, rieseguita l'intera suite; nessun controllo è stato disabilitato.
La verifica iniziale della sola appendice delle ambiguità ha rilevato la modifica
concomitante; il confronto ricostruito sopra ne conferma la conservazione.

Nessuna nuova inferenza/training GPU, installazione, cancellazione di artifact
o commit. I blocchi STRICT e l'equivalenza con le tabelle del paper restano
quelli delle fonti scientifiche, senza nuove rivendicazioni di riproduzione.
