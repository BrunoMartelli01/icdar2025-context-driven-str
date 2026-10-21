# English experiment-library refactor

This change translates maintained prose and report labels, and organizes
experiments through a shared navigation layer. It does not claim a new paper
result. See `REPORTING-ENGLISH-LIBRARY-001` and
[LANGUAGE_MIGRATION.json](../provenance/LANGUAGE_MIGRATION.json).

## What to read

- [Experiment families](../experiments/README.md): guides grouped by purpose,
  with links to implementation, configuration, tests, and results.
- [Code index](../experiments/CODE_INDEX.md): Python modules grouped by package,
  with source docstrings and links back to experiments.
- [Saved numerical results](../experiments/results/README.md): 26 comparisons,
  225 rows, in Markdown, JSON, and CSV. Every group retains its own protocol,
  split, denominator, source hash, and limitations.

```bash
python3 -m research experiments --family word-instances
python3 -m research experiment mask-crop-three-tests
python3 -m research results native128-pad0 --format json
```

`results` reads checked local summaries. `compare` rechecks original reports;
external roots are required for those sources. Neither command starts inference.
Experiments without a registered projection link to their dated results/audits;
they are not represented as missing successes or zero-valued results.

## Preservation checks

Nineteen guides were translated, together with historical display titles and
remaining report/visualization labels. Existing model, training, localization,
recognition, scoring, dataset, environment, and config bytes were preserved:
444 pre-existing scientific files match the archived source exactly. The
historical inventory retains all 368 evidence identities, source hashes, byte
counts, and heading positions; its display titles are English translations.

All 225 published rows were compared again with their original hash-verified
reports. The three Python report/visualization scripts retain identical AST
structure and non-string constants; changes are presentation strings. Existing
package paths and executable entry points remain stable. No weights, datasets,
environments, or large raw outputs were added to Git.

Completed-result additions and residual-VQA test updates arriving during this
work were retained. Those additions explain extra numbers in the two YOLO guides;
no original numerical tokens were removed. Old launch-time notes retain their
dated meaning rather than being rewritten as fresh results.

Original-language model inputs remain exact scientific payloads under the
[language policy](LANGUAGE_POLICY.md). A lexical English check covers maintained
prose and labels; it complements review rather than proving language universally.

## Validation

151 focused tests pass: evidence navigation/comparison, experiment references,
missing/tampered summaries, mode/config boundaries, localization, mocked DeepSolo
interfaces, offline imports, CPU segmentation smoke, router reporting, language
policy, and source-only export. This is not a full-suite GPU qualification.

A new saved-view test initially failed because text reads normalized CSV CRLF
line endings. The check now compares raw bytes; `.gitattributes` preserves CSV
bytes through Git storage. A regression test covers that boundary. The first
staged Git check also treated the intended CSV CRLF as trailing whitespace.
CSV attributes now declare carriage returns at line endings while retaining
the normal whitespace checks; tests verify that actual trailing spaces fail.
The source
policy initially rejected CSV; it now accepts small reviewed result tables only
under `experiments/results/`, retaining rejection elsewhere. Remaining historical
titles and a detector test-fixture issue were corrected. Temporary translation
helpers had variable/import-path errors, corrected before completion. No failing
check was disabled or replaced with dummy scientific output.

```bash
python3 -m scripts.check_repository_language
python3 -m scripts.build_experiment_library --check
python3 -m research catalog --source-only --check
python3 -m scripts.collaboration check
```

To independently verify the migration, obtain the maintainer's original-source
pack, verify its SHA256 from `LANGUAGE_MIGRATION.json`, and extract it into a
separate archive directory containing `original-source/`, `original-manifest.json`,
and the downloaded `original-source.tar.gz`. Restore the pinned original reports,
then run:

```bash
python3 -m scripts.audit_english_refactor \
  --archive-root /absolute/path/to/archive \
  --runs-root /absolute/path/to/restored-runs \
  --artifacts-root /absolute/path/to/restored-artifacts \
  --output /absolute/path/to/new-verification.json
```

The output is created exclusively. The tool verifies unchanged scientific files,
original numerical tokens, report-script structure, historical identities, and
all saved numerical rows. No original pack or missing report is fabricated.
STRICT blockers, metric ambiguities, and hardware limitations remain unchanged.
