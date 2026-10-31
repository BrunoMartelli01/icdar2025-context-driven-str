# Shared pipeline: code tour for a research presentation

This guide follows the common U-Net/PARSeq/router/DeepSolo implementation on
ICDAR2013, ICDAR2015 and Total-Text. Its result class is **approximate
reproduction / research extension**. The method is shared; neural weights,
fitted CPU models, vocabularies and selected operating points are explicit
dataset-specific resources. This does not mean one identical trained model
across all three datasets.

## Suggested presentation order

| Step | Open | Explain |
| --- | --- | --- |
| 1. Entire execution | [pipeline/uniform_word_pipeline.py](../pipeline/uniform_word_pipeline.py), `run` | Start from RGB images, run the light branch, freeze routing, call DeepSolo only for selected images, then combine words and save receipts. |
| 2. Input and neural frontend | Same file, `input_records` and `frontend` | Input accepts only IDs and image paths. The frontend saves predicted maps, genuine PARSeq readings and optional features from the existing U-Net encoder. |
| 3. Model and training | [scripts/run_uniform_unet_training.py](../scripts/run_uniform_unet_training.py), `make_model` and `train`; [segmentation/icdar13_word_refit.py](../segmentation/icdar13_word_refit.py), `WordRefitUNet` | Follow model construction and strict checkpoint loading. The shared training recipe produces separate dataset checkpoints. The historical IC13 filename is also used by the common pipeline. |
| 4. Crops and recognition | [scripts/train_icdar13_word_refit.py](../scripts/train_icdar13_word_refit.py), `read_rows`; [recognition/parseq.py](../recognition/parseq.py), `ParseqApproxRecognizer.recognize` | Trace predicted text/kernel maps to crops, then to actual PARSeq decoding and confidence. Follow the imported crop helper for the exact extraction rule. |
| 5. Routing decision | [recognition/uniform_utility_deployment.py](../recognition/uniform_utility_deployment.py), `observe`; [recognition/uniform_utility_fusion.py](../recognition/uniform_utility_fusion.py), `score`; [recognition/uniform_risk_fusion.py](../recognition/uniform_risk_fusion.py), `route` | Observable word, geometry, map and encoder features feed the fitted CPU heads. Deterministic batch ranking chooses the fallback population before loading DeepSolo outputs. |
| 6. Budget enforcement | [recognition/hard_fallback_budget.py](../recognition/hard_fallback_budget.py) | Validate the requested policy before inference; rescale hard quotas with floor rounding; verify the complete selected population. Legacy profiles without hard budgets use the separately documented ceiling rule. |
| 7. Heavy branch and word fusion | Back to `run`, then [recognition/uniform_fallback.py](../recognition/uniform_fallback.py) and [recognition/uniform_word_union.py](../recognition/uniform_word_union.py), `combine` | The isolated DeepSolo process receives only the selected RGB inputs. Qualified postprocessing and the frozen union model resolve overlapping light/heavy candidates. |
| 8. Evidence and limitations | [UNIFORM_HARD_BUDGET.md](UNIFORM_HARD_BUDGET.md), [UNIFORM_UNET_FUSION.md](UNIFORM_UNET_FUSION.md), [provenance/UNIFORM_PIPELINE_FINAL_SCOPE_AUDIT_V1.json](../provenance/UNIFORM_PIPELINE_FINAL_SCOPE_AUDIT_V1.json) | Distinguish complete TEST evidence, validation screens, live RGB smokes and cached routing replays. Read each dated stage rather than treating an older status paragraph as current. |

The compact flow to explain verbally is:

```text
RGB images
  -> shared U-Net architecture -> text/kernel maps -> crops -> PARSeq words
  -> observable features + existing encoder context -> CPU utility score
  -> deterministic batch route with an explicit DeepSolo image budget
       light images: retain admitted light words
       fallback images: isolated DeepSolo -> qualified word union
  -> predictions + routing counts + timing/memory/provenance receipts
```

The light branch runs on every image, including images later sent to DeepSolo.
The encoder context reuses the frontend encoder; it is not a new captioning
network. [recognition/uniform_encoder_context.py](../recognition/uniform_encoder_context.py)
contains the verified feature-bank and projection boundary.

## Entry point, profiles and outputs

The command-line entry point is
[scripts/infer_uniform_word_pipeline.py](../scripts/infer_uniform_word_pipeline.py).
Use `--help` in the existing main environment to inspect its required arguments:
`--config`, `--manifest`, `--dataset`, `--policy-quota` and `--output`.
The manifest contract is:

```json
{"samples": [{"sample_id": "example-1", "image_path": "/absolute/path/image.jpg"}]}
```

Read [uniform_pipeline_hard_budget_v9.json](../configs/approximate/uniform_pipeline_hard_budget_v9.json)
for the constrained runtime and
[uniform_pipeline_deployment_v6b.json](../configs/approximate/uniform_pipeline_deployment_v6b.json)
for the preceding deployment profile. Follow `utility_config` and
`fallback_profiles` to the CPU models and heavy resources. These are preserved
PAPER profiles with explicit paths and pinned hashes; a source clone alone is
not sufficient to execute them. Restore the exact assets using
[EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md). Do not rewrite recorded hashes to make
missing resources pass validation.

During a code demonstration, show the writes of `contract.json`,
`decisions_before_fallback.json`, `fallback_plan.json`, `predictions.json` and
`complete.json` inside `run`. The completion receipt reports DeepSolo images,
total images, percentage, actual neural calls, per-stage time and frontend peak
allocated/reserved GPU memory. Failures propagate; the command-line wrapper
records a failure receipt. Generated outputs stay outside Git.

## Latest candidate versus the qualified runtime

The v21 word-addition experiment in
[recognition/uniform_multiscale_words.py](../recognition/uniform_multiscale_words.py)
and [uniform_multiscale_words_shared_v21.json](../configs/approximate/uniform_multiscale_words_shared_v21.json)
is a subsequent validation candidate. Its receipt is
[UNIFORM_MULTISCALE_WORDS_SHARED_VALIDATION_V21.json](../provenance/UNIFORM_MULTISCALE_WORDS_SHARED_VALIDATION_V21.json).
It passed the declared shared validation screen using authenticated cached
observations. Matching decoder-heldout observations, head qualification, full
RGB integration and validation-frozen full TEST evaluation remain required.
It has not replaced the deployed runtime. A newer version number alone does
not establish a better qualified complete pipeline.

## Tests and scientific boundaries to show

- [tests/pipeline/test_uniform_word_pipeline.py](../tests/pipeline/test_uniform_word_pipeline.py): RGB-only input, identity collisions, legacy budget rounding and preservation of a frozen route when the fallback worker fails.
- [tests/recognition/test_hard_fallback_budget.py](../tests/recognition/test_hard_fallback_budget.py): hard quota limits, floor rounding and complete routing-population checks.
- [tests/recognition/test_uniform_encoder_context.py](../tests/recognition/test_uniform_encoder_context.py): real feature identity, dimensionality and finite projection inputs.
- [tests/recognition/test_uniform_utility_fusion.py](../tests/recognition/test_uniform_utility_fusion.py), [test_uniform_word_union.py](../tests/recognition/test_uniform_word_union.py) and [test_uniform_fallback.py](../tests/recognition/test_uniform_fallback.py): observable scores, conditional fusion and fallback boundary.

These focused tests use small fixtures; they do not establish recognition
accuracy. Full evaluation entry points include
[scripts/evaluate_uniform_utility_fusion.py](../scripts/evaluate_uniform_utility_fusion.py).
Metric boundaries are in
[segmentation/uniform_unet_fusion.py](../segmentation/uniform_unet_fusion.py),
`normalize`, `word_counts` and `aggregate_counts`.

The current lexical micro word F1 is distinct from paper recognition accuracy
and official spatial Hmean. Any displayed router result must include DeepSolo
images / total images / percentage, and distinguish logical cached routing
from real online inference. STRICT paper reproduction remains blocked by the
evidence listed in [STRICT_REPRODUCTION_STATUS.md](STRICT_REPRODUCTION_STATUS.md)
and [PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

For the paper-style caption/scoring comparison, use the separate
[PAPER_CAPTION_ROUTER_COMPARISON.md](PAPER_CAPTION_ROUTER_COMPARISON.md).
The shared utility/union router is a learned research extension; presenting it
as the paper's training-free BLIP-2/MPNet/fuzzy router would change the claim.
For other historical experiments, start at [experiments/README.md](../experiments/README.md)
and [experiments/CODE_INDEX.md](../experiments/CODE_INDEX.md).
