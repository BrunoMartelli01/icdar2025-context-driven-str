# Navigation and reporting refactor — 17 October 2026

Reporting choice classified as `ASSUMPTION`, `REPORTING-INDEX-001`.
Projected results remain **approximate reproduction** under their original
protocols. This refactor does not change models, evaluators, checkpoints, or environments.

## Findings and changes

| Observed problem | Change |
| --- | --- |
| Generic searches traverse millions of artifacts | `.rgignore` and document discovery with declared roots/depth/exclusions |
| Ambiguity log larger than 560KB, history spread across audits | Index with hashes, sections, and line positions; `find` and `section` read source evidence |
| Old milestones mark subsequently completed experiments as pending | Guide to specific sources without deleting historical paragraphs |
| “V1” names different pipelines and networks | Explicit groups with report paths, protocols, and variant identities |
| Accuracy, membership, pixel F1, geometry, and word OCR are confused | Separate tables by unit/metric, with visible counts and denominators |
| Manual copying of numbers and unverified paths | JSON-pointer/SHA256 registry; explicit failure on missing or inconsistent evidence |
| Execution host is implicit | Permanent AGENTS instructions and local/SSH guide for PAPER |

The collection covers `docs/`, `provenance/`, local artifact audits/reports, and
the two declared external roots. Exact scope and hashes are in the catalog.
Caches, samples, and files beyond the depth limit are not qualified as independent
audits. A receipt reporting PASS does not promote a historical result.

## Verified preservation

Compared with hashes captured before the changes: **123 existing scientific/
provenance documents remain identical**. README and AGENTS receive only
navigation/host additions. Ambiguity history is preserved: another concurrent
activity inserted `WORD-GT-003` (3.021 bytes), which this refactor neither altered
nor removed.

Removing only that concurrent insertion and the new `REPORTING-INDEX-001`
appendix reconstructs content matching the initial SHA256 exactly:
`bb2b3e4da65c3e9e578a880ce679464632406006090dccb7ae199f86913153a9`.
Concurrent GT V3 implementation/config/test additions are outside this refactor
and were left intact.

## Checks performed

- **51 tests** for new reporting/indexing passed in the existing main environment.
  They cover missing/changed sources, modes, populations, impossible counts,
  formulas, non-finite rates, gains/losses, pointers, paths, external sources,
  exclusive exports, cache pruning, and index freshness.
- **26 groups / 225 real rows** verified against saved reports: SHA256,
  declared qualifications, expected fields/populations, and available original
  rates. This does not rerun per-sample matching or qualify checkpoints.
- Standard-Python CLI smoke without site-packages, actual queries/sections,
  overview, JSON/CSV exports, and index checks.
- Change diff inspected; `git diff --check` passed.

The first test run had one failure: the assertion also counted a variant named
`variant` as a Markdown header. The assertion was corrected and the entire suite
rerun; no check was disabled. The initial ambiguity-appendix-only check detected
the concurrent modification; the reconstruction above confirms preservation.

No new GPU inference/training, installation, artifact deletion, or commit.
STRICT blockers and equivalence to paper tables remain as recorded in the
scientific sources, without new reproduction claims.
