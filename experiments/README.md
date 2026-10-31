# Experiment library

Browse a family, then follow its guide to code, configurations, tests, and saved results.
The catalog does not infer completion or merge incompatible protocols.

| Family | Guides | Scope |
| --- | ---: | --- |
| [paper](paper/README.md) | 6 | Paper baseline and scientific specification |
| [datasets](datasets/README.md) | 8 | Datasets, targets, and annotation preparation |
| [word-instances](word-instances/README.md) | 29 | Word instances, kernels, links, centers, and Gap |
| [crops](crops/README.md) | 16 | Crop generation, selection, and refinement |
| [recognition](recognition/README.md) | 21 | Recognition, language correction, and candidate verification |
| [routing](routing/README.md) | 41 | Context, captioning, CLIP and YOLO routing |
| [runtime](runtime/README.md) | 2 | Environments and isolated model runtimes |
| [analysis](analysis/README.md) | 62 | Diagnostics, audits, ablations and research synthesis |

```bash
python3 -m research experiments --family word-instances
python3 -m research experiment mask-crop-three-tests
python3 -m research results native128-pad0
```

[Code by package](CODE_INDEX.md) · [Saved numerical results](results/README.md) · [Language and evidence policy](../docs/LANGUAGE_POLICY.md)

Regenerate navigation: `python3 -m scripts.build_experiment_library`. Add `--check` to detect stale views.

Refresh numerical summaries only with the original reports: `--refresh-results --runs-root PATH --artifacts-root PATH`.
A missing source remains an error. No training or inference runs.
