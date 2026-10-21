# English source and evidence policy

Maintained documentation, comments, interfaces, report labels, generated navigation,
and result explanations use English. Experiment IDs, file paths, model revisions,
hashes, metric definitions, scientific values, and configuration keys remain stable.

Transcriptions, labels, captions, model prompts, quoted identifiers, and other
scientific inputs retain their original language and exact bytes. Translating
these would change the experiment. For example, the Italian prompt family in
`configs/approximate/clip_router_prompt_ablation_v2.json` is a measured multilingual
condition; its content must remain intact. The historical filename `confronto`
in the cleanup script is a file identity, not an untranslated interface label.

Before translation, a complete source copy was saved outside Git. Its manifest
records each path, byte count, and SHA256. The migration record at
`provenance/LANGUAGE_MIGRATION.json` links changed files to their archived
originals and records preservation checks. Obtain the original-source pack from
the maintainer and verify the listed hashes when exact historical text is needed.
No public archive URL is currently recorded. Current English text is a
translation; the archive preserves original source wording and authorization.

Historical audit inventories retain their original paths, source hashes, byte
counts, and line positions. Their display titles are translated and labeled as
such. The current source-only index is regenerated separately:

```bash
python3 -m research catalog --source-only
python3 -m research catalog --source-only --check
```

Experiment organization and saved numerical summaries are presentation choices
classified as `ASSUMPTION`; see `REPORTING-ENGLISH-LIBRARY-001` in
`docs/PAPER_AMBIGUITIES.md`. They do not establish new scientific results,
resolve strict blockers, or make different protocols comparable.

Regenerate navigation with `python3 -m scripts.build_experiment_library`.
Use `--check` to check it without changing files. Numerical summaries require
`--refresh-results` plus the original pinned sources; local summary checks alone
do not revalidate original predictions, matching, checkpoints, or hardware.
