# PARSeq visual-margin contextual gate — TRAIN offline diagnostic

`PRACTICAL_APPROXIMATION / approximate reproduction`, not paper routing or Table5.
Paper section3.5 uses semantic/lexical confidence and fallback, not this candidate
selector. Existing crop generation, recognizers, captions and prior runs unchanged.

## Predeclared experiment

Reuse AR top10 candidates and both cached Qwen likelihood conditions (`local`,
`local_florence`). No imports/loads of model runtimes for inference, no training,
no new candidates or captions. All operations CPU, no TEST data.

Baseline:502/745 eligible GT rows attached to700 unique V1 crops; the full TRAIN
denominator is849, with104 unavailable predictions unchanged. Not lexical503/849.
If multiple GT rows share a crop, the selector must make ONE decision for that
crop; only the post-hoc oracle may use labels.

For an exact-string scored KEEP reference:

```
dv = ar_logprob(candidate) - ar_logprob(V1)
dc = qwen_mean_logprob(candidate) - qwen_mean_logprob(V1)
eligible iff dv >= -m
gain = dv + beta * dc
choose greatest gain only if gain > tau, otherwise KEEP
```

The V1 string is post-refinement and need not be AR rank1. Recover its score from
the existing beam20, but NEVER add beam11..20 alternatives. Positive visual margins
are allowed when the actual V1 string has lower AR probability. Missing exact
reference (including capitalization-only mismatch) => KEEP; no substituted score.
AR probabilities are sums including EOS when completed, not length-normalized.
Qwen is mean subword log probability. Their combination is NOT a calibrated
posterior; length bias remains a limitation. Beam export contains no full token
paths for independent verification of truncation, so only saved-score consistency
is verified here.

Grid is exactly the user-requested 128 combinations:
m=.10,.25,.50,1; beta=.05,.10,.25,.50; tau=0,.05,.10,.25; two conditions.
No parameters added based on results. Ten controls including KEEP: visual_only
(strict positive dv), and eight visual_margin_gate_only controls (positive dc
within each margin, no additive visual weighting). KEEP wins zero/tau ties;
other ties use frozen candidate order. Diagnostic marginals also report m=2,
but this margin does NOT enter the grid.

Potentially interesting: recovered>=5, damaged<=2, net>0. Best TRAIN entries sort
by exact, fewer damaged, fewer changed crops, smaller m/beta, larger tau. This is
TRAIN selection, not independent validation or automatic promotion.

## Provenance and outputs

Source report/inputs/labels/protocol SHA256 pinned in the separate approximate
config. Each of700 Qwen shards is checked against candidate identities and score
mean/sum/token counts, then hashed in a frozen protocol before grid evaluation.
This does not retroactively prove the old runtime provenance: old artifacts lack
a complete cryptographically bound inference contract.

Output directory: artifacts/diagnostics/parseq-visual-margin-context-gate-train-v1/.
protocol.json and results.json are sealed, never overwrite conflicting outputs.
Report retains128 metrics, controls, every chosen candidate index, crop order,
observable cached evidence, separate evaluation labels and recoverable GT margins.

