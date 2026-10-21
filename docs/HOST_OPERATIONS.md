# Macchina PAPER e comandi

Destinazione indicata dall'utente: **`bmartelli@paper.dinfo.unfi.it`**.
La grafia è conservata esattamente. Alcuni documenti storici riportano
`paper.dinfo.unifi.it`; questa differenza non è stata risolta cambiando
silenziosamente la destinazione. Verificare DNS/connessione quando serve SSH.

Il controllo di questo refactor ha restituito `hostname -f = paper` e
`whoami = bmartelli`: in questa sessione i comandi vengono già eseguiti su PAPER.
Non occorre una seconda connessione SSH per eseguirli sulla stessa macchina.

## Prima di eseguire

```bash
hostname -f
whoami
pwd
```

Repository su PAPER:
`/extra1/bmartelli/projects/icdar2025-context-driven-str`.
Se ci si trova già su PAPER:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
python3 -m research groups
```

Da una macchina diversa, per un comando remoto di sola lettura:

```bash
ssh bmartelli@paper.dinfo.unfi.it \
  'cd /extra1/bmartelli/projects/icdar2025-context-driven-str && python3 -m research groups'
```

Se il nome indicato non risolve, riportare l'errore e verificare la destinazione;
non modificare alias, chiavi SSH o configurazioni di rete per aggirarlo.

## Posizioni operative

| Contenuto | Posizione storica su PAPER |
| --- | --- |
| Repository | `/extra1/bmartelli/projects/icdar2025-context-driven-str` |
| Run persistenti | `/extra1/bmartelli/context-str-runs` |
| Pesi e artifact esterni | `/extra1/bmartelli/context-str-artifacts` |
| Dataset | `/extra1/bmartelli/datasets/context-str` |
| Main Python esistente | `.venvs/context-str-main-py311/bin/python` nel repository |
| YOLO Python esistente | `.venvs/context-str-yolo-py311/bin/python` nel repository |
| DeepSolo legacy esistente | `.venvs/context-str-deepsolo-host/` nel repository |

Sono riferimenti operativi, non default da incorporare nei modelli o nei config
scientifici. Non tutti i run vivono qui: alcune prove usano `/var/tmp` o `/tmp`;
la loro presenza deve essere verificata, non assunta.

Per i report senza modelli usare `python3`, senza attivare ambienti pesanti:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
export CONTEXT_STR_RUNS_ROOT=/extra1/bmartelli/context-str-runs
export CONTEXT_STR_ARTIFACTS_ROOT=/extra1/bmartelli/context-str-artifacts
python3 -m research check
python3 -m research compare yolo-test-equal
```

Per inferenza e training scegliere l'ambiente documentato dal componente.
Prima di intervenire su DeepSolo leggere [DEEPSOLO_RUNTIME.md](DEEPSOLO_RUNTIME.md),
il manifest M6 e le voci DeepSolo dello stato STRICT. Non unificare gli ambienti
e non installare pacchetti durante una semplice consultazione.

Prima di un run GPU leggere `nvidia-smi` e verificare i processi attivi.
Le qualifiche RTX 3070, i run su RTX 3090 e la configurazione del paper restano
evidenze hardware distinte. Registrare il device effettivo e le deviazioni nel run.
