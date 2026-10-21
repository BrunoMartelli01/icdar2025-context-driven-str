# New CLIP uses and GPU comparison command

This **approximate reproduction** battery uses Total-Text TRAIN125 already
used for development. It compares three frozen checkpoints on identical PARSeq
words/boxes; it is neither independent TEST nor reproduced Table5. The
[previous battery](CLIP_WORD_VERIFICATION.md) and its results remain separate.

## One command from the host terminal

```bash
bash /extra1/bmartelli/projects/icdar2025-context-driven-str/scripts/run_clip_verification_suite_host.sh
```

The command runs tests, checks CUDA/scientific inputs, reuses local ViT-B/32,
and downloads missing pinned ViT-B/16 and ViT-L/14 snapshots. It installs no
packages. Processes use one GPU, float32, batch 16; models load separately.
Actual GPU, input shapes, allocated/reserved peaks, failures/OOM are recorded.
Precision, device, model, and batch never change automatically after an error.

Downloads include only required model/processor/tokenizer files, excluding
TensorFlow/Flax weight copies. The downloader checks full commit, LFS SHA256
ETag, and weight bytes. A verified receipt is retained for B/16 too, whose
upstream checksum was unresolved in the old config; that config is not changed.
In `--offline` mode, an absent upstream checksum remains explicitly unverified.

Log: `/tmp/clip-verification-host-suite.log`, configurable with `CLIP_SUITE_LOG`.
Results under `/tmp/clip-verification-host-suite/<protocol-hash>/`:

* `COMPARISON.md`: rule table per model;
* `comparison.json`: complete comparison and download provenance;
* `classic/`: all v1 comparisons, including description prompts;
* `innovative/`: new comparisons with prompts/scores/choices/rendering hashes;
* `downloads/`, `hardware.json`, `launch.json`: provenance/configuration;
* `suite_failure.json` and/or `failure.json`: propagated errors when present.

The path is printed in the final `COMPLETED` line. Code, config, software,
device, and image limit define a new output tree; repeated identical protocols
reuse compatible intermediates. Old CPU experiments are preserved. Change
output location with `--output-root /path/of/your/choice`.

For a local check without downloads, with the model already available:

```bash
bash scripts/run_clip_verification_suite_host.sh --models vit_b32 --device cpu --offline --limit 1
```

For an initial three-checkpoint GPU smoke, add `--limit 1` to the full command;
results stay labeled smoke. An unlimited run uses all 125 images and 983 care
GT. The script does not open TEST.

## Implemented hypotheses

1. **Dependence on the written region.** For candidate `w`:
   `score(w) = cos(wide_crop,w) - cos(wide_crop_with_target_covered,w)`.
   A high score measures support from the target's presence. Alternatives share
   both inputs. The whole box is covered, including background within it: this
   does not causally prove every letter was recognized. Raw/quoted/ensemble,
   contextual residual, and a PARSeq agreement gate are tested.
2. **Candidate words as images.** Each complete PARSeq string is rendered in
   DejaVu Sans, Serif, Mono, black-on-white/white-on-black. Compare the real crop
   with their normalized mean visual features. Candidates also enter the visual
   branch, avoiding text tokenization for this score. Semantic bias is not
   guaranteed to disappear: CLIP's visual encoder can encode text meaning.
   Font/shape/length may become shortcuts.
3. **Letter order.** Subtract similarity of the same reversed rendered string:
   identical characters/fonts/polarities, different order. `STOP`/`POTS` is a
   control example, not a new OCR output. Reversed strings never enter the
   selectable pool. Palindromes/repeated characters identical to their reverse
   provide zero order evidence. Meaning/kerning can still change, so this is
   not a pure spelling measure.
4. **More input space for text.** Also stretch crops to fill the official CLIP
   resolution's square using bicubic resize. With very wide words, the previous
   square canvas leaves few vertical text pixels. Test enlargement with textual
   templates and visual renderings, separately from aspect-preserving canvases.
   Stretching is an experimental assumption, not a STRICT-preprocessor change.

Selection always retains PARSeq boxes/raw strings. The glyph gate requires
agreement among PARSeq confidence, written-region effect, and multiview visual
comparison, with fixed margins. No GT/caption text generates candidates. Text,
font/checksum, polarity, geometry, and RGB hashes recreate every synthetic probe.
Rendering charset is printable ASCII: empty/invisible/unsupported inputs fail
explicitly, without replacement glyphs.

The complete host comparison is now finished for **B/32, B/16, and L/14**:
[verified results and interpretation](CLIP_VERIFICATION_RESULTS.md).
Report: `/tmp/clip-verification-host-suite/6920d0b0c030/comparison.json`.

The complete battery has **40 rules per checkpoint**, including all previous 28
controls. The counterfactual-only battery has 33 rules and does not claim to
evaluate rendering/stretching. Protocol, assumptions, hardware deviations:
[config v2](../configs/approximate/clip_innovative_verification_train_v2.json)
and `CLIP-INNOVATIVE-VERIFICATION-V2` in `PAPER_AMBIGUITIES.md`.

## Sources and further ideas

[FDP, ACM MM2024](https://arxiv.org/abs/2408.00441) identifies limited text
perceptual scale and entangled visual/semantic concepts as CLIP scene-text
retrieval problems. It includes learned components; these scale/spelling tests
are hypotheses inspired by that diagnosis, not reproductions of its architecture
or results.

[T-MARS, ICLR2024](https://arxiv.org/abs/2307.03132) masks text/recalculates CLIP
similarity to filter pretraining data dominated by text. Here before/after
comparison studies target contribution for each candidate: task/selection differ.

[Disentangling visual and written concepts in CLIP](https://joaanna.github.io/disentangling_spelling_in_clip/)
also studies meaningless words and projections separating spelling/visual
concepts. Official pretrained projections could be frozen after qualifying
compatibility/version/checkpoint; they have not been downloaded/integrated.
Another extension could compare single-letter-difference word pairs using visual
patches in the differing region; CLIP-patch-to-character mapping remains undefined.

## Checks performed in this session

**53 tests pass**, including downloaded-checkpoint integrity, rendering order/
charset, font pooling, immutable pools, preserved v1 selections, empty
 denominators, and simulated/propagated OOM. An initial test detected a bracket
error in the new branch; it was fixed before inference. Full host script
completed a real B/32 CPU smoke with all 40 rules under
`/tmp/clip-verification-host-suite/47395c888c47/`. This is engineering verification
on one image, not evidence of glyph benefit.

Only the counterfactual portion was evaluated on all prior B/32 TRAIN125 scores,
without new inference or changing the original 28 controls:

| Rule | Correct /983 | Gained | Lost | Changed words |
|---|---:|---:|---:|---:|
| PARSeq baseline |368|0|0|0|
| PARSeq confidence over alternatives |383|26|11|351|
| Before/after difference, written prompt |362|14|20|360|
| Before/after difference, contextual residual |366|19|21|378|
| PARSeq + written-region-effect agreement |372|4|0|23|

The gate recovers 4 words without observed losses; the other 19 changes do not
increase exact matching. Unrestricted maximum-score selection remains
insufficient; PARSeq confidence reaches a higher total. Among 67 one-edit pairs,
written-prompt effect chooses correctly in 68.7%, versus 59.7% for the previous
crop written prompt and 82.1% for PARSeq control. These are correlated development
data, not significance or correction-safety guarantees.

Complete report:
`/tmp/clip-innovative-verification-v2/vit_b32/counterfactual/results.json`,
SHA256 `0901ef57307bac7b0b361ad0b4844906cfc5d4a1d3e273609b5448075ab138c6`.
Metrics/matching/denominators follow v1: exact casefold with punctuation,
one-to-one bboxIoU>=0.5 matching, all care GT.

CUDA preflight fails in the Codex session; the host run completed all 40 rules
per checkpoint on TRAIN125 using RTX 3090. CPU verification recalculated all six
reports from 750 shards and checked consistency of code, inputs, original
selections/outputs. L/14 PARSeq/CLIP fusion reaches 386/983; rendered candidates
383/983; conservative PARSeq/crop/wide-crop gate 376/983, eight recoveries, zero
observed harm. Adding descriptions to crop prompts worsens exact matching for
all checkpoints. These are APPROXIMATE development results, not independent
confirmation or RTX 3070 qualification.
