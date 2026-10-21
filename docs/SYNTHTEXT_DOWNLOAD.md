# Automatic acquisition for the detection experiment

This helper acquires the original scene-image SynthText dataset from the
[torrent linked by its authors](https://github.com/ankush-me/SynthText#pre-generated-dataset).
It selects only `SynthText.zip` (about 40.72GB on
[Academic Torrents](https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c)),
excluding the background/depth/renderer datasets. The official Oxford web page
no longer exposes a direct dataset download. This is acquisition for the existing
**approximate reproduction** extension, not a new scientific method or strict result.

Run on `bmartelli@paper.dinfo.unifi.it`:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
bash scripts/run_unet_detection_methods_with_download.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/context-str-runs/unet-word-research-v1 \
  /extra1/bmartelli/context-str-runs/unet-detection-methods-synthtext-small-v1 \
  /extra1/bmartelli/context-str-artifacts/parseq \
  /extra1/bmartelli/datasets/context-str \
  --subset-only --discard-archive
```

The fifth argument is the dataset **parent**. The resulting root is
`/extra1/bmartelli/datasets/context-str/SynthText`, containing `gt.mat` and the
image directories directly. Download/control files and receipts live separately
under `synthtext-downloads/`. The recommended options extract only the configured
50,000 eligible images and then remove this helper's verified cached ZIP. See
the storage details below. If `aria2c` is unavailable, the helper downloads a
portable Linux x86_64 build into this cache: release 1.37.0 from
[abcfy2/aria2-static-build](https://github.com/abcfy2/aria2-static-build/releases/tag/1.37.0),
archive SHA256 `e0a09b12ef67f35f8a8e4fdddbec851d235b7c31da549d0578bff459032b499a`.
It uses no sudo, package installation, Python environment changes, or mutable
`latest` reference. A preinstalled/explicit aria2 executable is recorded with its
version and hash. Automatic portable acquisition requires Linux x86_64.

The helper validates torrent info hash `2dba9518166cbd141534cbf381aa3e99a087e83c`,
derives the archive index instead of guessing it, enables torrent piece checks,
rejects incomplete transfers, hashes the downloaded ZIP, and checks member CRCs
while extracting. Extraction uses atomic per-file publication and resumes valid
files; conflicting user files are preserved and cause a failure. Torrent and ZIP
paths and symbolic links are checked before extraction. Existing unmanaged
nonempty dataset roots cause an explicit error. No alternate dataset or automatic
label filtering is substituted. The original `gt.mat` remains unchanged.

The two options independently select reduced extraction and ZIP removal:

- `--subset-only` retains the original `gt.mat` and the exact configured 50,000
  eligible images. It also retains overlap/duplicate candidates visited before
  completing that subset, so normal scientific preparation encounters identical
  bytes and makes identical exclusions. It reuses `prepare_synthtext` with an
  optional image-materialization callback; selection, seed, coordinate origin,
  RGB checksums, annotations and image validation remain shared. Default behavior
  without this callback is unchanged. Selected IDs/words/RGB hashes and exclusions
  are tested against full extraction and replay without the callback.
- `--discard-archive` removes only the helper's cached `SynthText.zip` after a
  successful verified acquisition, checking its path, byte count and SHA256.
  Annotation/crop/model files are never deleted. Completed acquisition can be
  reused after ZIP removal. A failed selection or mismatching ZIP cannot trigger
  removal. The retained subset manifest and archive SHA stay in the receipt.

Reduced acquisition initially checks archive bytes plus 28GiB free: 16GiB of
extraction headroom and the existing 12GiB experiment reserve. It checks actual
free space for each original file before extracting it; this is a disk budget,
not a measured dataset size or an accuracy change. Without `--subset-only`, the
previous full extraction and three-archive-size reservation remain available.
Peak dataset storage in reduced mode is the downloaded ZIP plus original GT and
retained images; after removal it is only GT and retained images. Actual retained
bytes are printed and recorded in `complete.json`; they have not yet been
measured on the real external corpus. The ~41GB download itself remains necessary
because the torrent contains one ZIP rather than independent image entries.

Downloads and extraction are locked and resumable. Stop with Ctrl+C and rerun
the **same command**, with the same acquisition options. Full/subset selections
are immutable contracts; a different selection requires a different dataset
parent. Completed acquisition rechecks the annotation hash and reuses images;
the scientific runner independently hashes and verifies its selected 50,000
images and binds the manifest. Completed extraction CRC checks are not repeated
for every image on each launch. Availability and speed still depend on the
external torrent peers and the host network.

The wrapper verifies existing Total-Text, source checkpoint/report, and PARSeq
artifact hashes before downloading. It passes the acquired root to the existing
launcher, retaining the same 5-epoch SynthText/50-epoch real-data schedule, held-out
split, scientific configuration, baseline and result protection. The final
`comparison.md/json/csv` report is in the third argument's directory.

Offline qualification:

```bash
.venvs/context-str-main-py311/bin/python -m pytest -q tests/scripts/test_synthtext_download.py
bash -n scripts/run_unet_detection_methods_with_download.sh
```

These checks use tiny generated engineering fixtures, not genuine SynthText.
Live download and GPU training cannot be qualified from the network-restricted
development session. Actual dataset acquisition/training results are not claimed.
Qualification on 2026-10-12: 13 focused tests passed; CLI and Bash syntax checks
passed; the wrapper's actual local source/PARSeq SHA preflight passed and resolved
the configured 50,000-image subset. The live HTTP probe failed with curl exit 6
(DNS unavailable in the sandbox), so it is not counted as a download smoke pass.
An intermediate combined test run passed 23 tests and failed one assertion that
expected a generic MAT parse exception. The corrupted-MAT assertion now checks
SciPy's specific `MatReadError`; production behavior continues to propagate it.
The final combined run passed all 24 tests (13 acquisition tests and 11 existing
detection-runner tests), with no remaining failures. The complete four-file task
diff was reviewed; existing scientific modules/configurations were not edited.

Storage extension on 2026-10-12: 43 focused acquisition/target/runner tests
passed, including full-versus-subset ID/label/RGB-hash equivalence, exclusion
replay, ZIP removal and cached resume. `prepare_synthtext` now has the optional
acquisition callback described above; ordinary preparation remains read-only.
Its source SHA therefore changes: use a new experiment output root for this
code revision. Old qualified receipts/checkpoints are retained as historical
evidence, not relabeled as qualification of the new source. Actual retained
space and host GPU results remain unmeasured. See UDM-STORAGE-001 in
`docs/PAPER_AMBIGUITIES.md`.

Original-archive compatibility correction on 2026-10-12: the real downloaded
ZIP contains a root-level `license.txt` as well as the `SynthText/` tree. The
initial path gate rejected this legitimate member before extraction. Both full
and subset extraction now preserve its exact CRC-checked bytes at
`SynthText/license.txt`, with that mapping recorded in the acquisition receipt.
Arbitrary root files, symbolic links and colliding mapped filenames still fail.
This corrects acquisition compatibility without changing GT, subset selection,
training configuration or scientific source contracts. Rerun the same command
to reuse the downloaded archive and continue; no fresh ZIP download is required
if aria2's integrity verification passes. The pipeline does not automatically
restart a user's stopped job.
Qualification of this correction: 47 acquisition/target/runner tests passed.
Read-only checks of all 858,976 original ZIP central-directory members found no
collision under the corrected mapping. The real 1,327-byte license was read with
ZIP CRC verification; representative original paths passed the corrected gate.
No real image extraction or training was performed in that qualification.
