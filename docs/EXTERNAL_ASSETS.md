# External assets: obtain, verify, recreate

Git contains source code, configurations, small manifests and provenance.
Everything excluded below remains external, including files small enough for
GitHub when they are data, weights, generated experiment outputs or secrets.
This also keeps the repository below [GitHub's file-size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

The machine-readable [EXTERNAL_ASSETS.json](../provenance/EXTERNAL_ASSETS.json)
copies the source/artifact registries with their original statuses and hashes,
lists saved-comparison inputs, and inventories excluded workspace families.
It does not promote a candidate to paper evidence or imply that a local path
can be downloaded. Canonical scientific records remain
[SOURCES.yaml](../provenance/SOURCES.yaml), [ARTIFACTS.yaml](../provenance/ARTIFACTS.yaml),
and [PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

To update this derived inventory on the full workspace, using its existing main
environment (this scans local artifact metadata, not model inference):

```bash
.venvs/context-str-main-py311/bin/python -m scripts.collaboration inventory \
  --output provenance/EXTERNAL_ASSETS.json
```

Review the diff before replacing the committed inventory. Running this on a
code-only clone cannot rediscover excluded historical files; preserve the
original inventory rather than overwriting it with an empty local scan.

## What is excluded

| Asset | Obtain / recreate | Verification / limitation |
| --- | --- | --- |
| Paper PDF | [Paper retrieval commands](../papers/README.md) | Exact arXiv v2 SHA256; no silently accepted new PDF bytes |
| Main Python, uv and packages | `bash scripts/create_main_environment_host.sh` | Exact installed-version snapshot, then new host qualification; not a wheel-hash lock |
| DeepSolo/Detectron2/native builds | Separate environment/source procedure below | Original M6 base construction is incomplete; independent rebuild needs new qualification |
| Datasets, ZIPs, images, labels | Dataset-specific official portal/mirror below | Terms, split, archive hashes, per-file manifests; mirrors are explicit candidates |
| Pretrained source/checkpoints | Pinned source URLs/revisions below and JSON inventory | Compare hashes before loading; strict identities remain blocked where unresolved |
| Trained project checkpoints | Restore a maintainer-provided run archive, or train anew with its frozen config | New training is a new experiment; the original bytes need the original artifact/hash |
| HF/torch caches | Re-download pinned artifacts into explicit external cache roots | Caches are disposable containers, not artifact identity |
| Masks, probabilities, crops, OCR/captions/features | Re-run the producing experiment with its exact config/inputs/checkpoints | Save intermediate outputs and hashes; never regenerate in place over historical evidence |
| Large results, visualizations, archives, logs | Restore an external run pack or execute the documented producer anew | No public historical pack URL currently recorded; missing exact inputs remain `UNRESOLVED` |
| Credentials and populated `.env` files | Each collaborator authenticates locally; copy empty templates only | Never obtain another contributor's credentials or include them in an archive |
| Python/native caches, compiled extensions | Interpreter/build tools regenerate them in the isolated environment | Requalify extensions on new CUDA/compiler/hardware |
| Accidental root shell logs | Historical files stay on the original machine; use a new external run log | No scientific input or setup dependency |

## Workspace layout

Choose absolute external roots, then pass them explicitly to programs:

```text
artifacts-root/
  parseq/                         pinned source archive, extracted source, checkpoint
  deepsolo/                       external source and upstream candidate weights
  huggingface/                    HF_HUB_CACHE
  huggingface-home/               HF_HOME
  torch/                         TORCH_HOME
datasets-root/
  icdar13/                       raw Task 1/2 images/annotations
  icdar2015/downloads/            original archives
  total_text/downloads/           verified mirror/official tree
  ctw1500/downloads/              detection archives, separate recognition labels
  scene-segmentation-v1/          generated targets, validity, manifests
runs-root/                       one immutable directory per experiment
```

These are layout conventions, not scientific defaults. Existing host paths are
listed in [HOST_OPERATIONS.md](HOST_OPERATIONS.md). Do not require access to
PAPER to contribute code.

## Source and candidate weights

### PARSeq-S candidate

This candidate uses v1.0.0 checkpoint bytes with the separately qualified
PyTorch-2-compatible official runtime. It is **APPROXIMATE**, not proof of the
paper's exact checkpoint. With `CONTEXT_STR_ARTIFACTS_ROOT` explicitly set:

```bash
mkdir -p "$CONTEXT_STR_ARTIFACTS_ROOT/parseq"
curl --fail --location --output "$CONTEXT_STR_ARTIFACTS_ROOT/parseq/parseq-bb5792a6.pt" \
  https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt
curl --fail --location \
  --output "$CONTEXT_STR_ARTIFACTS_ROOT/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz" \
  https://codeload.github.com/baudm/parseq/tar.gz/1902db043c029a7e03a3818c616c06600af574be
printf '%s  %s\n' \
  bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00 \
  "$CONTEXT_STR_ARTIFACTS_ROOT/parseq/parseq-bb5792a6.pt" \
  f57289cb1fc7fe41f488a52ea48f8a7f114252ef405b0b64efc1d528c29b48ef \
  "$CONTEXT_STR_ARTIFACTS_ROOT/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz" \
  | sha256sum --check
```

Only after both checks pass, extract the archive into that `parseq/` directory:

```bash
tar -xzf "$CONTEXT_STR_ARTIFACTS_ROOT/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz" \
  -C "$CONTEXT_STR_ARTIFACTS_ROOT/parseq"
```

Alternatively clone `https://github.com/baudm/parseq.git` and detach-checkout
`1902db043c029a7e03a3818c616c06600af574be`, selecting `git_checkout` source mode
in the adapter. Do not install v1.0.0's incompatible package constraints into
the main environment. Runtime/checkpoint provenance must remain separate.

### BLIP-2, MPNet and encoder candidates

After creating the main environment, use its installed `huggingface_hub`:

```bash
export HF_HOME="$CONTEXT_STR_ARTIFACTS_ROOT/huggingface-home"
export HF_HUB_CACHE="$CONTEXT_STR_ARTIFACTS_ROOT/huggingface"
.venvs/context-str-main-py311/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo, revision in (
    ("Salesforce/blip2-flan-t5-xl", "0eb0d3b46c14c1f8c7680bca2693baafdb90bb28"),
    ("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130"),
):
    print(snapshot_download(repo_id=repo, revision=revision,
                            cache_dir=os.environ["HF_HUB_CACHE"]))
PY
```

This downloads large model repositories; it does not load the models. Verify
the two BLIP-2 safetensor shards against SHA256
`23bb7a7a78644afa9a138dd6c4360edf51ca5fc6da1c5d01f147b964ad7c0c6d` and
`c77bd5bcdbc0945c57444d34d28bd4452dc715c3d8aef54e2e91459fdafb7e03`.
MPNet `model.safetensors` must have SHA256
`78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658`.
The snapshot directories printed by the command locate those files. The paper's
BLIP-2 variant, decoding and MPNet identity remain unresolved; these are pinned
repository candidates. Some research paths use different captioners/encoders;
their component configs and inventory identities take precedence for that run.

The ResNet-50 candidate is `torchvision.models.ResNet50_Weights.IMAGENET1K_V2`,
file `resnet50-11ad3fa6.pth` from
`https://download.pytorch.org/models/resnet50-11ad3fa6.pth`. Set `TORCH_HOME`
externally before the documented training launcher initializes it. The older
registry establishes only the `11ad3fa6` hash prefix: record the full downloaded
SHA256 in a new run, and do not treat that prefix as a strict identity resolution.

BLIP-2 single-device FP32 does not fit the historical 8 GB target. Offload,
FP16, quantization and sequential loading require explicit `HARDWARE_DEVIATION`
records and qualification, not a silent setup optimization.

### DeepSolo and Detectron2

Read [the full runtime record](DEEPSOLO_RUNTIME.md),
[M6 manifest](../docker/deepsolo/runtime_manifest.json), and DeepSolo entries in
[strict status](STRICT_REPRODUCTION_STATUS.md) before touching this stack.

Clone the official source outside the project and freeze it:

```bash
git clone https://github.com/ViTAE-Transformer/DeepSolo.git "$CONTEXT_STR_ARTIFACTS_ROOT/deepsolo-source"
git -C "$CONTEXT_STR_ARTIFACTS_ROOT/deepsolo-source" checkout --detach dbadae995035246bad3376c7a44c015c69e9b313
```

Use the `DeepSolo/` subdirectory as the adapter's source root. The Total-Text
R50 candidate `tt_res50_finetune_synth-tt-mlt-13-15.pth` is linked from that
frozen upstream model-zoo README and `ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE` in
the asset registry. Its SHA256 is
`ac2a01b3d8df0ee42e4632ff57a0d3a20a5706fb0ac1c5ae33b2455d0d90da68`.
Other model-zoo candidates are separate entries; do not substitute them by name.

The M6 Docker image ID is recorded, but no public image registry or complete
original base construction is available. A maintainer may provide a private
`docker save` archive with its SHA256 and `docker image inspect` identity;
otherwise use the documented reconstruction wrapper with a supplied legacy base
and qualify the new image independently. The wrapper alone cannot build the
missing base. Do not claim the original image has been reconstructed.

The separate host reconstruction tool exposes all required paths:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.prepare_deepsolo_host --help
```

It needs a separate Python 3.8, CUDA 11.1 toolkit/compiler and frozen source/
checkpoint inputs. Follow [environments/deepsolo/environment.yml](../environments/deepsolo/environment.yml)
and the later host qualification in the runtime record. Detectron2 v0.6 identity
is `d1e04565d3bec8719335b88be9e9b961bf3ec464`. Rebuild native extensions in the
isolated stack; never put Detectron2 into the modern main environment.

## Datasets and prepared targets

| Dataset | Obtain | Prepare / verify |
| --- | --- | --- |
| ICDAR13 | [RRC Challenge 2](https://rrc.cvc.uab.es/?ch=2); 229 train / 233 official test. Two training mirror URLs and archive hashes are recorded in `SOURCES.yaml` | `segmentation.icdar13_manifests --help`; frozen `manifests/icdar13_*`; Task 1 don't-care/Task 2 masks are distinct inputs. Paper recognition split remains ambiguous |
| ICDAR2015 | [RRC Challenge 4](https://rrc.cvc.uab.es/?ch=4&com=downloads), or `bash scripts/download_icdar2015_host.sh "$CONTEXT_STR_DATA_ROOT/icdar2015/downloads"` | Archive checks, then scene preparation below; 1000 train / 500 public test |
| Total-Text | [Official dataset](https://github.com/cs-chan/Total-Text-Dataset); historical mirror helper `bash scripts/download_total_text_host.sh "$CONTEXT_STR_DATA_ROOT/total_text/downloads"` | 1255 train / 300 test; mirror helper is unpinned and requires fresh manifest/checksum qualification |
| CTW1500 detection | [Official source](https://github.com/Yuliang-Liu/Curve-Text-Detector); historical mirror helper `bash scripts/download_ctw1500_host.sh "$CONTEXT_STR_DATA_ROOT/ctw1500/downloads"` | 1000 train / 500 test; out-of-paper text-line dataset; mirror download is unpinned |
| CTW1500 recognition | Separate upstream end-to-end transcription archive; `scripts/download_ctw1500_recognition_host.sh` and `scripts/prepare_ctw1500_recognition.py --help` | Geometry reconciliation and missing-label counts in `docs/SCENE_TEXT_DATASETS.md`; detection-only labels are insufficient |
| COCO-TS | Official source URL in `SRC-COCOTS-OFFICIAL` | Paper's counts conflict with the named dataset; exact mapping/protocol `UNRESOLVED`, no strict-ready recipe |
| SynthText | [Project](https://www.robots.ox.ac.uk/~vgg/data/scenetext/), `python -m scripts.download_synthtext --help` | [SYNTHTEXT_DOWNLOAD.md](SYNTHTEXT_DOWNLOAD.md); research extension; use its own pinned manifest/qualification |

Links record acquisition sources; this onboarding change does not download or
requalify all datasets. Registration/terms may be needed. Existing mirror
helpers preserve historical behavior and may follow mutable revisions; a fresh
download is **not** the old qualified bytes until hashes/manifests agree. For a
new acquisition pin the dataset revision and retain the original archives.

`scripts.prepare_scene_text_segmentation` expects the exact source archive/tree
names in `segmentation/scene_text_preparation.py`. Inspect that preparer and
[SCENE_TEXT_DATASETS.md](SCENE_TEXT_DATASETS.md), then run:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.prepare_scene_text_segmentation \
  --download-root "$CONTEXT_STR_DATA_ROOT" \
  --output-root "$CONTEXT_STR_DATA_ROOT/scene-segmentation-v1"
```

It creates image/target/validity trees, per-file hashes and manifests without
mutating downloaded archives. ICDAR15/CTW polygon-filled targets are approximate
assumptions; they are not automatically paper-equivalent glyph masks. For OCR
manifests use `python -m scripts.prepare_scene_text_dataset build --help` and
`verify --help`, with explicit root/image/annotation/output paths. Do not use
`--allow-partial` for a complete official-population qualification.

## Project checkpoints and generated experiment outputs

The authors' trained AG-U-Net checkpoint is unavailable in the inspected source.
Local approximate checkpoints are different artifacts. Restore an exact one
using its run directory and SHA256 in `ARTIFACTS.yaml`/its component audit, or
create a new run from the same resolved configuration. The baseline training
entry point and a one-step qualification are in [GETTING_STARTED.md](GETTING_STARTED.md).

| Output family | Producer / guide |
| --- | --- |
| Baseline segmentation checkpoints/masks | `segmentation.train`, `segmentation.evaluate`; `docs/AG_UNET_IMPLEMENTATION_DECISIONS.md` |
| Dataset-specific and combined segmentation | `scripts.prepare_scene_text_segmentation`, `segmentation.train`; `docs/SCENE_TEXT_DATASETS.md` |
| Text/Kernel/Center, crop links, word research | Component `run_*/train_*` modules in `scripts/`; `docs/WORD_MULTITASK_RESULTS_V1.md`, `docs/UNET_WORD_RESEARCH.md`, `docs/UNET_WORD_RESEARCH_V2.md` |
| Gap / kernel-only | `scripts/train_adaspot_gap_fulltrain.py`, `scripts/train_kernel_only_fresh.py`; `docs/ADASpot_GAP_FULLTRAIN.md` |
| Word-instance extensions | `scripts/run_word_instance_extensions.py`; `docs/WORD_INSTANCE_EXTENSIONS.md` |
| Probability/mask refiners | `scripts/run_mask_probability_refiner.py`; `docs/MASK_PROBABILITY_REFINER.md` |
| PARSeq/DeepSolo outputs and captions | `scripts/profile_parseq.py`, `scripts/profile_deepsolo.py`, `scripts/export_train_blip2_captions.py`; component configs/runtime guides |
| CLIP/YOLO/router research | Specific `scripts/run_*/evaluate_*` module and frozen config; `docs/CLIP_ROUTER_LITERATURE_MODELS.md`, `docs/YOLO_PARSEQ_FOUR_DATASET_COMPARISON.md` |
| Other historical analyses/reviews | Find their exact path/title with `research find`, then read the preserved workspace inventory and producing audit/config |

Use each producer's `--help` and exact component config. The table is a route to
the procedure, not an assertion that all historical artifacts can be recreated.
Some experiments depend on outputs of earlier runs. Keep those dependencies,
source-map receipts, IDs and hashes intact; do not fill missing predictions with
dummy outputs or silently skip failed samples.

### Restore saved comparisons

`provenance/EXTERNAL_ASSETS.json` lists each registered comparison input as
`root`, `path`, `sha256`, groups and `included_in_source_export`. Repository
sources outside the source allowlist (usually `artifacts/...`) must be restored
at the same relative paths in the clone. Other sources use explicit external
roots:

```bash
export CONTEXT_STR_RUNS_ROOT=/absolute/path/to/restored-runs
export CONTEXT_STR_ARTIFACTS_ROOT=/absolute/path/to/restored-artifacts
python3 -m research check
python3 -m research compare native128-pad0 --format json
```

Obtain a missing historical pack from the maintainer with its checksum,
relative-path manifest and required source reports. **No downloadable pack or
unrecorded recovery command is invented here.** Missing packs, deleted temporary
outputs and incomplete runtime construction remain explicit recovery gaps.
New inference/training may produce a scientifically useful new result, but must
not replace an old report under its registered hash. Static result summaries
remain readable without those packs and do not validate missing raw evidence.

## Access and licenses

Dataset and model terms are listed separately in the registries; an unresolved
license stays unresolved. Mirrors do not grant new rights. Do not publish data,
trained artifacts or a Docker image until its terms and intended distribution
are established. Credentials are supplied independently by each contributor.
The upstream README's MIT claim has no accompanying upstream license file in
this workspace; no license file for the combined repository is fabricated.
