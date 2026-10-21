# Stato della ricerca e direzioni: maschere, istanze e crop

2026-10-21. **Approximate reproduction / analisi esplorativa di output salvati.**
Nessun nuovo training o inferenza neurale. I risultati degli altri filoni sono
letti dai run completati e legati agli SHA256 dei rispettivi audit, non dedotti
dalle sole note di avvio. In particolare la suite word-instance è ora
SUCCESS/complete,2026-10-21T01:35UTC, anche se la sua guida descrive ancora lo
stato al lancio. Le configurazioni STRICT e la produzione restano invariate.

## Conclusione operativa

La direzione meglio sostenuta è **generare alternative locali e scegliere un
insieme di crop completo e compatibile, con budget esplicito**. Il nostro test
V3 passa795→870 parole senza cambiare U-Net/PARSeq. Deformare automaticamente
tutti i crop è invece negativo in due esperimenti di supervisione OCR e nel
proxy RGB Mask-Align. Questo non dimostra che allineamento o supervisione OCR
non possano funzionare in altre architetture; identifica le priorità attuali.

La famiglia Text+Kernel è un riferimento importante da aggiungere. Text e Kernel
sono regioni di testo e nuclei delle parole, non due reti lettere/parole.
Il ramo Gap full1 arriva1311/2204, contro1293 del suo controllo M/K. Le cifre
non formano una classifica controllata con V3: cambiano risoluzione512/native256
contro256/native128, geometria letterbox/stretch, target, training, budget e
trattamento degli ignore. Il beneficio misurato del Gap è18 parole sul proprio
controllo; il divario1311–870 non è un effetto isolato del Gap.

## Aggiornamento sui filoni paralleli

| Filone e popolazione | Confronto interno verificato | Interpretazione |
| --- | --- | --- |
| V3, TEST300/2204 | Selettore795→870; F1 37,06→39,60%; +92/-17 parole | Miglioramento effettivo di precisione e recall, ancora esplorativo |
| V3, stesso TEST | Cap rimosso795→904; F1 37,06→36,96% | Recupero reale, con più falsi positivi |
| M/K e Gap, TEST300/2204 |1293→1311; F1 53,96→54,31% | Piccolo guadagno; decoder Gap aggiunge32,6M parametri ai58,2M congelati |
| Mask-Align, stesso controllo Gap | RGB originale1311, bilineare1300, ruotato1307, mask1170 | Il warp RGB guidato dalla maschera peggiora; OCR sulle feature allineate non è stato addestrato/testato |
| Sei moduli locali, TRAIN holdout125/983 | Prompt e affinity/maximin scelgono off; geometria522, geometria+OCR520, baseline520 |20epoche reali per braccio e audit PASS; nessun vantaggio OCR della nuova loss rispetto al controllo abbinato |
| YOLO/DeepSolo, TEST300, metrica senza IoU | Coverage640 F1 77,32→77,60%;960/1280 peggiorano | Guadagno del gate incerto: intervallo F1[-0,46,+1,03]punti. Non confrontare questo F1 con quello dei crop localizzati |

La prova Gap10 diagnostica legge1324 parole ma ha F1 54,25%, inferiore a full1.
Non sostituisce l'epoca1 predefinita osservando il TEST. Il precedente +2 del Gap
sul holdout125 era concentrato in una sola immagine; il refit1255 non lascia
quel gruppo fuori dal training e non è una nuova valutazione su tale holdout.

Mask-Align ha realmente prodotto feature64x8x32/64x16x64 e verificato gradienti;
la sua lettura PARSeq riguarda invece RGB ricampionato. I due risultati non
vanno confusi. La copertura del supporto della maschera predetta non dimostra
copertura dei veri caratteri, né identifica da sola la causa delle regressioni.

Il gate YOLO continua ad accettare solo41/150 immagini completamente corrette
al budget50%, anche con coverage: confidence alta non certifica completezza.
La policy a budget è un ordinamento sul batch, distinto da una soglia scalare
trasferita; il suo costo non è il costo della sola U-Net.

Sul riferimento Gap1 il matching geometrico massimo copre1612 delle2204 parole,
mentre1311 vengono anche lette esattamente. Restano592 annotazioni senza un
abbinamento geometrico uno-a-uno. Il divario residuo fra localizzazione e lettura
comprende anche crop incompleti/contaminati: non è una misura dell'errore puro
di PARSeq e non costituisce una tassonomia causale disgiunta per singola parola.

## Analisi aggiuntiva effettuata

Bootstrap appaiato per immagine,10000estrazioni,seed20261021, intervalli95%
percentili. Modelli e politiche sono fissi: gli intervalli non includono
variabilità del training, ricerca dei parametri o riuso del TEST.

| Confronto contro il proprio controllo | Delta F1, punti | Intervallo95% condizionato | Immagini con più/meno parole corrette |
| --- | ---: | --- | ---: |
| V3 selettore | +2,54 | [+1,41,+3,69] |60/14 |
| V3 senza cap | -0,10 | [-1,73,+1,75] |34/0 |
| V3 rifinitore appreso | -1,26 | [-2,22,-0,29] |28/46 |
| Gap full1 contro M/K | +0,34 | [-0,09,+0,85] |14/2 |
| Mask-Align RGB contro originale | -5,84 | [-7,12,-4,58] |12/104 |

Il guadagno V3 non dipende da un'unica immagine: il massimo incremento netto
per immagine è5. Il conteggio per immagine è distinto dai92 recuperi/17 perdite
di singole parole: nella stessa immagine possono verificarsi entrambi.
Il Gap porta un aumento di parole corrette, ma il miglioramento F1 rimane
compatibile con zero in questo bootstrap condizionato. La sua maggiore capacità
non va giustificata con la sola diminuzione della loss TRAIN.

### Complementarità dei crop: potenziale, non nuovo risultato

Con le letture già salvate ho ricostruito matching massimi uno-a-uno e verificato
il replay di tutte300 immagini:

| Insieme di candidati | Parole abbinate correttamente |
| --- | ---: |
| Baseline cap10 |795 |
| Tutte le componenti originali |904 |
| Selettore tagli/unioni attuale |870 |
| Unione componenti originali + uscite del selettore, oracolo |996 |
| Tutto il pool locale, oracolo |1032 |
| Pool locale + tutte le componenti originali, oracolo |1141 |

Il996 indica che i recuperi da componenti scartate e quelli da trasformazioni
locali sono complementari. Nessun selettore attuale ottiene996 o1141. Si usano
le GT per il matching e non si impongono tutti i vincoli di esclusività fra
gruppi alternativi: questi sono potenziali del pool, non configurazioni fattibili
già validate. Non è lecito sommare semplicemente i guadagni dei due sistemi.

Un ulteriore confronto descrittivo trova116 parole lette da V3+selettore e non
dal Gap1,754 comuni e557 solo Gap1. GT e ordine sono verificati identici. La
unione delle identità GT è1427, ma richiede un oracolo su due sistemi con costi
e protocolli diversi; non propone di eseguire due U-Net come nuovo riferimento.

### Il costo finora non quantificato

Il selettore usa le letture PARSeq delle alternative. **2190 crop emessi non
significano2190 crop da riconoscere**: il pool contiene5537 proposte e5165
ritagli RGB unici, contro2086 della baseline. A freddo sono **2,48 volte** i
crop da leggere, senza attribuire a questo rapporto un uguale aumento di
latenza. Le cache storiche riusate nell'esperimento non sono un risparmio
garantito su nuove immagini. Questo è un motivo forte per studiare una
preselezione economica prima di PARSeq e riportare qualità a parità di budget.

## Direzioni promettenti, in ordine

1. **Trasferire il selettore discreto al ramo Text+Kernel e misurare il contributo
   del Gap.** Confronto2x2: M/K vs M/K+Gap1, assemblaggio originale vs alternative
   selezionate. Stessi campioni, PARSeq, matching, ignore e budget in ogni coppia;
   scelte solo TRAIN/calibrazione. Riutilizzare i generatori letterbox esistenti:
   il nostro generatore nativo128/stretch non è direttamente compatibile con
   le mappe256/letterbox. Il trasferimento non è ancora stato eseguito e il suo
   guadagno non si deduce dai75 ottenuti su V3. Tenere M/K come controllo meno
   costoso invece di assumere necessario il decoder Gap aggiuntivo. Prima di
   allenare un nuovo selettore, misurare il potenziale del pool su TRAIN: se
   mancano le parole nei candidati, occorre migliorare la generazione. Con il
   Gap full1255 il vecchio holdout è già stato visto dalla rete: una nuova
   validazione richiede dati/ruoli sigillati non usati da quel modello.

2. **Budget adattivo e selezione prima del riconoscitore.** Mettere nello stesso
   pool componenti oltre il cap e poche alternative locali. Addestrare su TRAIN
   una qualità del crop da RGB/maschera/provenienza, senza usare l'OCR del crop
   ancora da leggere; poi PARSeq su un budget ristretto e confronto col selettore
   completo. Ottimizzare gruppi alternativi e conflitti, con negativi troncati,
   multiword e duplicati, non solo correttezza indipendente di ogni proposta.
   I numeri996/1141 motivano il test, non ne fissano il risultato atteso.
   Misurare curve F1/recall contro crop OCR unici e tempo reale, più recuperi/
   perdite dei successi originali. Coverage e confidence sono segnali possibili,
   non certificati di completezza.

3. **Solo come ricerca successiva: OCR sulle feature già allineate.** Il lavoro
   Mask-Align rende possibile un test della stessa piccola testa OCR allenata
   su feature assiali vs feature Mask-Align, con stesso backbone congelato,
   capacità, dati, budget e selezione. È il passo già indicato nel filone AdaSpot,
   non un risultato esistente. Un'eventuale testa deve apprendere il dominio
   delle feature: PARSeq congelato su RGB deformato non ne misura il potenziale.
   Lasciare gate/deep head/joint training a esperimenti successivi per isolare
   il fattore che cambia.

La calibrazione a livello di parola ha motivazione in
[On Calibration of Scene-Text Recognition Models](https://arxiv.org/abs/2012.12643),
che studia l'overconfidence dei recognizer. Calibrare la probabilità di una
lettura non risolve automaticamente caratteri mancanti o parole mai proposte:
la qualità/complettezza del crop deve avere un obiettivo distinto.
[Mask TextSpotter v3](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560681.pdf)
e [FOTS](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_FOTS_Fast_Oriented_CVPR_2018_paper.html)
supportano proposte da segmentazione e feature allineate per un riconoscitore
addestrato. Non dimostrano l'efficacia dei nostri warp RGB congelati.
[PSENet](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html)
motiva l'uso di nuclei separati ed espansione: questa idea è già in parte
rappresentata dal ramo M/K e non va ripresentata come novità locale.
Il più recente [SemiETS, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Luo_SemiETS_Integrating_Spatial_and_Content_Consistencies_for_Semi-Supervised_End-to-end_Text_CVPR_2025_paper.html)
studia consistenza spaziale/contenuto e pseudo-label affidabili; è una pista
per nuovi dati non annotati o trasferimento di dominio, non la prima necessità
del pool attuale già annotato. Questi collegamenti motivano ipotesi; non sono
prove del guadagno delle combinazioni qui proposte.

## Cosa de-prioritizzare e come verificare

Non ripetere senza un nuovo fattore isolato: rifinitore continuo con loss OCR,
prompt locale, maximin sulle stesse mappe, warp RGB generalizzato, aumento
globale della risoluzione YOLO o nuova rete Kernel-only. I controlli esistenti
sono negativi o modesti. Non concludere che i principi siano impossibili;
evitare di rinominare prove già fatte come nuove direzioni.

Prima di investire in altra capacità: fissare una valutazione a livello di
parola con ruoli dati separati e verificarla su dati non usati nella ricerca
delle scelte; ripetere il selettore su più seed. Il bootstrap attuale non
sostituisce nessuno dei due. Esistono due file immagine identici nei TRAIN/TEST
locali (212↔578,81↔656), già registrati: il primario resta300/2204 e la
sensibilità298/2187 resta separata. Il nostro guadagno persiste su quel sottoinsieme,
ma neppure il298 storicamente esplorato diventa un benchmark intatto.

## Evidenze e riproducibilità

Fonti principali:

- [V3: confronto completo](</extra1/bmartelli/context-str-runs/mask-crop-three-tests-20261021-v1/comparison.md>).
- [Gap fullTRAIN](ADASpot_GAP_FULLTRAIN.md) e [Mask-Align](ADASpot_MASK_ALIGN.md).
- [Sei moduli locali: run completo](</extra1/bmartelli/context-str-runs/word-instance-extensions-20261021-v1/comparison.md>) e relativo `independent_audit.json`.
- [YOLO/coverage: confronto e incertezza](YOLO_TEXT_COVERAGE_RESOLUTION.md).

Nuova analisi:
`artifacts/diagnostics/research-synthesis-20261021-v1/`.
`results/bootstrap.json`, `results/candidate_oracles.json`,
`results/cross_family_overlap.json`, `candidate_workload.json`,
`source_verification.json`. Sono verificati319 SHA256 di file sorgente e il
legame fra risultati e audit dei quattro altri filoni. Self-test numerici e
replay dei conteggi originali PASS, nessun originale modificato.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/context-str-main-py311/bin/python \
  artifacts/diagnostics/research-synthesis-20261021-v1/analyze.py \
  --mode approximate --config configs/approximate/research_synthesis_20261021_v1.json \
  --output /percorso/nuovo/report
```

Classificazioni: `REPOSITORY_SPECIFIED` per protocolli/fonti riusati;
`ASSUMPTION` per bootstrap condizionato e oracoli descrittivi;
`UNRESOLVED` per trasferimento, varianza fra training, causalità fra famiglie e
beneficio dell'OCR sulle feature. Il ranking delle prossime ricerche è un
giudizio motivato da questi dati, non un risultato scientifico già ottenuto.
