# Table5: source verification and meaning of accuracy

Audit on 16 October2026 of `papers/2503.15639v2.pdf`, section4.1, Table5,
references42/46/48/66/71. Source analysis without new inference, evaluator
changes, or reproduction claims.

**Finding:** all four Total-Text baseline numbers occur as end-to-end F1/Hmean
in primary sources, with a Mask TextSpotter identity/version error. This strongly
suggests Table5 intends F1; it does not establish how either “Ours” row was
computed. Their operational metric remains `UNRESOLVED`. Some ICDAR numbers
also differ from model rows in cited publications.

## Total-Text correspondences

| Table5 row | Value | Primary source/location | Verified interpretation |
|---|---:|---|---|
| Mask TextSpotter[42] |71.2|[Mask TextSpotter v3, Table3, PDF p.13](https://arxiv.org/pdf/2007.09482v1#page=13)|End-to-end F-measure, None. Our paper's reference42 is instead Mask TextSpotter2018.|
| MANGO[48] |72.9|[MANGO, Table2, PDF p.6](https://ojs.aaai.org/index.php/AAAI/article/view/16348/16155#page=6)|End-to-end, None, long input side1600;83.6 with Full.|
| SPTS[46] |74.2|[SPTS, Table6, PDF p.6](https://arxiv.org/pdf/2112.07917v6#page=6)|Point-based end-to-end Hmean, None;82.4 with Full.|
| TESTR[71] |73.3|[TESTR CVPR2022, Table1, PDF p.7](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Text_Spotting_Transformers_CVPR_2022_paper.pdf#page=7)|End-to-end F-score, TESTR-Polygon, None;83.9 with Full.|

“None” means no external lexicon. “Full” uses test-set words as lexicon: a distinct
setting requiring separate reporting. No lexicon does not remove benchmark
care/ignore filtering.

## ICDAR row verification

Columns below: ICDAR2013 / ICDAR2015 / Total-Text. Original ICDAR values use
end-to-end Generic unless noted.

| Method | Our paper | Primary source | Outcome |
|---|---|---|---|
| Mask TextSpotter2018 |84.1 /70.3 /71.2|86.5 /62.4 /52.9|Does not match cited model;71.2 belongs to v3.|
| MANGO |88.7 /72.9 /72.9|88.7* /67.3 or73.9* /72.9|IC13/Total-Text match; IC15 differs.|
| SPTS |88.5 /73.5 /74.2|88.5 /65.8 /74.2|IC13/Total-Text match; IC15 differs.|
| TESTR |86.4 /73.3 /73.3|IC13 not reported /73.6 /73.3|Only Total-Text matches. IC15 without lexicon is65.3.|

Mask2018: [Tables1,2,4, PDF pp.12,12,14](https://openaccess.thecvf.com/content_ECCV_2018/papers/Pengyuan_Lyu_Mask_TextSpotter_An_ECCV_2018_paper.pdf).
MANGO: Table1; asterisk uses Mask TextSpotter2019-specific lexicons. IC15
73.9 is MANGO*,67.3 is conventional-lexicon MANGO. There is no unique ICDAR
number independent of lexicon, scale, task. SPTS: Tables4/5, confirmed by
[official repository](https://github.com/shannanyinxiang/SPTS#performance).
TESTR: CVPR PDF Table3 as well as Table1.

Numerical coincidences illustrate row-confusion risks: SPTS paper's73.5 is the
Mask TextSpotter IC15 row; MANGO paper's70.3 is MANGO IC15 Word Spotting Generic,
and 84.1 is Boundary IC13 E2E Generic. TESTR's86.4 is **Total-Text detection
F-score** in resolution1000 ablation, not IC13. These coincidences do not prove
where authors obtained values. Undocumented independent reruns might exist,
but evidence is insufficient to reconstruct those rows.

## DeepSolo and stated differences

[Official DeepSolo CVPR2023 Table5](https://openaccess.thecvf.com/content/CVPR2023/papers/Ye_DeepSolo_Let_Transformer_Decoder_With_Explicit_Points_Solo_for_Text_CVPR_2023_paper.pdf#page=7)
reports Total-Text None: R50 without TextOCR79.7, R50 with TextOCR82.5,
ViTAEv2-S with TextOCR83.6. Table1 explicitly defines None/Full as F1-measure;
prose sometimes calls these same results accuracy. Supplementary Table8 reports
IC13 Generic R50=90.1.

Our paper's78.4+4.1=82.5 exactly matches R50 **with TextOCR**, unlike our
qualified candidate without TextOCR. Conversely79.8+3.9=83.7 differs from the
ViTAEv2-S table's83.6. DeepSolo Table6 gives IC15 Generic76.9/79.1 for R50
without/with TextOCR,79.5 for ViTAEv2-S with TextOCR. Generic “82%” in our paper
identifies none of these settings. Checkpoint/lexicon must not be selected to
force comparison agreement.

## Score calculation

With TP=correctly localized/transcribed matched words, FP=incorrect care
predictions, FN=unrecovered care words:

`P = TP/(TP+FP)`; `R = TP/(TP+FN)`;
`F1 = Hmean = 2PR/(P+R) = 2TP/(2TP+FP+FN)`.

Counts aggregate across the dataset, not mean per-image accuracy.
[Official DeepSolo evaluator, commit dbadae9](https://github.com/ViTAE-Transformer/DeepSolo/blob/dbadae995035246bad3376c7a44c015c69e9b313/DeepSolo/adet/evaluation/text_eval_script.py#L451)
computes precision/recall from global counts and returns hmean. Polygon matching
uses IoU **strictly greater than0.5**; a prediction is neutralized when more than
half its area lies in a single ignore region. Text/care rules depend on task:
E2E/Word Spotting are not interchangeable, nor equivalent to bbox-IoU>=0.5 or
plain casefold.

The aggregate formula alone does not establish protocol equivalence:

- MANGO declares adapted IoU>0.1 matching for coarse localization; this does
  not make it equivalent to polygon0.5 matching.
- SPTS assigns points to nearest GT centers, checks transcription, counts
  duplicates as FP. Its [official eval.py, commit3f61ec9](https://github.com/shannanyinxiang/SPTS/blob/3f61ec9d3cfd6d69d84086ee975e588f4ec1a926/evaluation/eval.py#L184)
  returns global-count Hmean; Total-Text enables Word Spotting filters,
  including minimum length3/character restrictions. CLI selects maximum Hmean
  over confidence0.80..0.94 sweep. This describes inspected code, not proof of
  the “Ours” protocol.
- TESTR specifies polygon IoU after converting Bezier points.
- Our paper omits recognition-accuracy formula, lexicons, matching, ignore, evaluation
  code needed for equivalence. IC13 count168/229 also conflicts with 233 TEST
  in its Table1; without a manifest, actual split cannot be established.

## Implications for our experiments

Our TN=0 accuracy is `J = TP/(TP+FP+FN)`, distinct from F1. **Only with identical
counts/protocol** do `F1=2J/(1+J)` and `J=F1/(2-F1)` hold. If78.4/79.8 were F1,
they would correspond to J=64.47/66.39%, not J=78.4/79.8%. This conditional
conversion does not reconstruct author results.

The earlier YOLO/CLIP comparison remains locally valid among gates sharing a
backbone/fallback: TEST CLIP F1=73.45%, J=58.04% with 150 descriptive acceptances.
Its IoU-free multiset matching, punctuation-retaining casefold, native 256 ignore
are not official polygon benchmarks. Converting J to F1 alone cannot establish
paper equivalence. External comparison requires official evaluator/GT, identical
checkpoint, lexicon, normalization, ignore, threshold protocol. Unlocalized T1 output
also needs an author-unspecified rule.

## Evidence and reproducibility

Pinned arXiv author PDFs, published CVF PDFs, text extracts, visually checked table
images, official SPTS code, SHA256 manifests remain outside Git:
`/extra1/bmartelli/context-str-artifacts/paper-table5-source-audit-20261016/`.
`source_audit.json` and `comparison.csv` distinguish confirmed, different, and
different-version cells without changing historical results. MANGO was checked
against author preprint and browser-accessible AAAI PDF tables; failed direct
AAAI download remains recorded without substituted values. No model, dataset,
or package was installed/modified. Scientific reference:
`TABLE5-SOURCE-METRIC-AUDIT-001` in `docs/PAPER_AMBIGUITIES.md`.
