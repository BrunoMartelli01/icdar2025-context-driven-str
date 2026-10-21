# One GT-word group per annotated word

This is an **oracle TEST diagnostic** in APPROXIMATE mode. It asks how well the
frozen crop selector chooses a reading if localization/group membership is
provided by annotations. It does not estimate deployable text-spotting accuracy.

For each of the 2,204 care words in the 300-image Total-Text TEST artifact,
make exactly one group. A frozen original crop belongs to that group when its
original-coordinate axis-aligned bbox IoU with the GT bbox is at least 0.5.
One crop can belong to multiple GT groups, preserving the earlier multiword
membership rule. Empty groups remain in the denominator.

The primary selector is the existing `mask_completeness_confidence` rule. Its
reference bbox is selected from the group's member crops: the candidate closest
to their coordinate-wise median. **The GT bbox is not used to compute the
selector score.** The existing maximum-score assignment allows one output per
group and one use per original crop. `confidence_balanced` is an unchanged
PARSeq-only control. All masks, crops and OCR readings are frozen; no model is
rerun.

Run the reproducible diagnostic with:

```bash
bash scripts/test_gt_word_group_selector.sh
```

## Full TEST result

Machine-readable report:
`/tmp/totaltext-crop-group-test-suite/30643625ace4/gt-word-groups-420282ed5f05/results.json`.

| Quantity | Count |
|---|---:|
| GT groups | 2,204 |
| Groups with at least one crop at bbox IoU>=0.5 | 1,997 |
| Groups containing at least one correct crop reading | 1,823 |
| Oracle ceiling with one crop per group/crop | 1,823 |
| Primary selector outputs | 1,997 |
| Primary selector correct within assigned GT group | **1,659** |
| PARSeq-confidence control correct within assigned GT group | 1,525 |

The decomposition is 207 groups without a crop at the declared spatial match;
174 more have crop(s) but no exact normalized reading; 164 have an available
correct reading that the primary selector does not select (including any
competition for shared original crops). The local one-to-one evaluator can
cross-match two primary outputs to a different nearby GT, yielding 1,661 global
matches; the requested *within-group* answer is 1,659.

The previous predicted-group run produced 1,722 matches from 18,052 outputs.
That higher raw count comes with many overlapping guesses per GT; a single
GT-defined group forces one choice. Neither result is an official polygon
Total-Text spotting score. Earlier TEST diagnostics motivated this oracle
question; the fixed IoU convention and selector were not fitted to the result.
