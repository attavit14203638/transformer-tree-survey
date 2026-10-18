# Author decisions and judgement calls (B01 to B08)

Status: decisions applied on 2026-10-14 and 2026-10-15 at the author's request (sections "Decisions applied" below), the
dataset-key policy (C) and the date rule included. The final corpus is 133 papers (132 studies), all extracted and cross-checked.

This file collects in one place the decisions taken after the full-text extraction, (0) the checklist that was prepared for them,
(1) every paper that an agent proposed for exclusion, (2) the questions raised during extraction, and (3) the judgement calls made while
extracting each batch. The evidence for each is in `reports/B##.md`, in the `notes` field of the paper in the workbook, and in
`compiled/conflicts.csv` (value choices). Sections 0 to 3 are kept as the record of what was weighed; where they recommend something
different from "Decisions applied", the decision applies.

## Decisions applied on 2026-10-14

The author asked the agent to re-check the proposed exclusions (A1), the kept closest calls (A2) and the scope groups (B) against the
selection criteria and to decide them. Every paper was re-read where the decision depended on it and judged against the five criteria
of `Scripts/search/screening.py` and against the author's own full-text exclusions (the excluded rows of `papers.csv` without a paper ID).
The criteria were read as follows, for every paper alike:

- Criterion 1 (transformer): a transformer-based model must be scored quantitatively, or its output must be what the tree-level score
  measures, whether it is the proposed model, a baseline or a tool. A transformer shown only qualitatively, used only to annotate
  without a score, or applied to a non-tree output does not count (as the author's exclusions R0194, R0777 and R1088).
- Criterion 2 (individual tree): segmenting tree crowns or trees in high-resolution imagery or point clouds counts, also when it is
  scored per pixel or per point (as P033). Area-level products do not count: cover, extent or canopy-loss maps, plantation maps scored at
  sample points, labels of whole image patches, scene classes, and generic tree masks inside general-purpose land-cover benchmarks (as
  R0049, R0566, R0983, R1438, R1432 and R1504).

Corpus after full text: 139 papers (138 studies; P032 and P035 are two reports of one study), 132 found by the database searches and
7 by other methods, all extracted (`compile.py`: 139 of 139 included papers extracted; 3113 results, 1384 primary; 143
transformer-versus-baseline comparisons; 193 transfer pairs; 57 value choices). PRISMA (`screening.py prisma`): 2,522 records, 976
duplicates removed, 1,546 screened, 1,372 excluded on title and abstract, 174 reports sought, 4 not retrieved, 170 assessed, 38
excluded (Area-level mapping, not individual trees 18; No transformer-based model trained or evaluated 8; Not a tree task 7; Not
remote sensing or close-range tree imagery 2; No individual-tree output evaluated 2; No quantitative evaluation 1), 132 included;
other methods: 10 assessed, 3 excluded (Area-level mapping), 7 included.

### A. Inclusion (A1 and A2)

| ID (batch) | Decision | Reason |
|---|---|---|
| P003 (B04) | Exclude: Full text not retrievable | No PDF could be obtained for the extraction (H1). |
| P006 (B04) | Exclude: Area-level mapping, not individual trees | Each resized aerial image gets a palm or non-palm label; no tree is detected, counted or located (as R0566 and R1438). |
| P028 (B04) | Exclude: Area-level mapping, not individual trees | Olive plantation map from pixel tree segmentation and 32 x 32 patch labels, scored by pixel F1, sample-point map accuracy and canopy cover per hectare; no tree is delineated or counted. |
| P037 (B04) | Exclude: Area-level mapping, not individual trees | Urban canopy-loss mapping; the SAM crown partitions are not evaluated (as R0049). |
| P038 (B05) | Include; extracted (242 rows) | SAM-HQ crown masks are scored by IoU against manual annotations (Fig. 9, p.12) and the Mask R-CNN crowns on a test set. The B05 proposal had missed the SAM-HQ IoU values. |
| P046 (B05) | Exclude: No transformer-based model trained or evaluated | CCD-YOLO is convolutional and no transformer baseline is scored (as R1088). |
| P055 (B05) | Exclude: No transformer-based model trained or evaluated | SAM appears only in a qualitative comparison; the scored models are convolutional. |
| P070 (B06) | Exclude: No transformer-based model trained or evaluated | The only learned model is FastSAM, a YOLOv8-seg CNN. |
| P071 (B06) | Exclude: No individual-tree output evaluated | Completion of 1 m3 point cloud segments, scored only by Chamfer distance and F-score (as R1020). |
| P089 (B06) | Include; extracted (68 rows) | A transformer completion network is trained on single-tree point clouds and the species classification of individual trees is scored; the scored classifier uses multi-head self-attention pooling. |
| P097 (B07) | Exclude: Area-level mapping, not individual trees | Scene-class point segmentation scored over all classes of S3DIS and DALES (as R0983). |
| P155 (B04) | Exclude: Area-level mapping, not individual trees | General vision-language model whose only tree result is a binary tree segmentation ablation on m-NeonTree with an unnamed metric (as R1504 and EarthDial). Its rows were written and then removed on 2026-10-14. |
| P156 (B04) | Exclude: No quantitative evaluation | The SAM tree segmentation is judged visually. |
| P092, P098, P103, P104, P109, P125, P131, P136, P137, P140, P143, P144 | Include (no change) | Each scores a transformer (proposed model, baseline or tool) and a tree-level output. |

### B. Scope groups

| Group | Decision |
|---|---|
| B1. Pixel masks of diseased or dead crowns (P125, P129, P130, P131, P133, P135, P140, P142, P151, P152) | Include all: criterion 2 names the classification of crowns by health, disease or mortality, and the masks delineate the crowns. |
| B2. Component papers (P064, P065, P067, P073, P079, P088, P093, P095) | Include: criterion 2 names tree components. |
| B3. P086 and P068 | Include both: P086 segments trees in 3D in the same way as the kept 2D crown segmentation papers (P033, P039, P043, P052, P054). This reverses the earlier recommendation to exclude P086. |
| B4. Close-range imagery (P043, P049, P065) | Include: criterion 3 names close-range tree images. |
| B5. General-purpose models | P143 and P154 Include (each scores a tree task); P155 and P156 Exclude (A). |
| B6. Completion papers | P089, P092 and P098 Include (a tree-level score measures the completed trees or uses them); P071 Exclude. |

### C to H

| Item | Outcome |
|---|---|
| C. Dataset keys | Applied on 2026-10-14 (policy below): own-data keys that had been merged by inference were split (`ajman_date_palm_uav` into P011, P018 and P051; `kroof_uav` into P024 and P057; `huangshan_pwd_2022` into P127 and P139; P129 no longer uses `own_p125_qingdao_pwd`; P094's joint plots have their own key), every note that compared papers was removed from the rows of 19 papers and from `datasets.csv`, and protocol rule V9 was reworded. P144's sensor is now not reported (it had been taken from another paper's description). 14 public datasets stay shared because each paper names them. |
| D. Values not stated | The 27 papers with a primary result of split `not_stated` and the 5 with level `not_stated` were searched again for a statement. Changed: P099 tree heights, split `validation` (the paper calls its field reference trees a validation dataset, p.6). Kept `not_stated` because the paper names no split for the table: P004, P017 (its test data are all labelled images, training regions included), P026, P039, P052, P067, P069, P073, P081 (automated testing includes the training plot), P083, P093, P094, P095, P098, P100, P115, P117, P131, P137, P138, P141, P143, P144, P146, P147, P154. P039, P098 and P100 describe a split elsewhere, and in P141 the FCL-RTDETR row of Table 8 equals the Sanmen test row of Table 10; none of them names the split of the table, so none is tagged. Level `not_stated` kept for P065, P067, P109, P122 and P137. For the manuscript: 26 of the 139 papers do not name the data split of their main results and 5 do not state the unit that is scored; their results are kept in the tables but left out of comparisons that need a known split (protocol 6.4). |
| E. Identical or copied rows | No change. Values a paper quotes from earlier work keep `copied_from` and stay out of the statistics; values two papers print identically stay as printed. |
| F. Value choices | No change: the 57 value choices follow rules V1 and V2 (the main results table wins). |
| G. Source values that look like errors | No change: the recorded values are those of the main tables; the observations stay in section 0 G for reference. |
| H1. P003 | Excluded (Full text not retrievable). |
| H2. compile.py | Extended: kappa may be -1 to 1, and the review kind `percent_below_one` is read (the seven P154 warnings are now reviewed notes). P047's ablation kappa of -0.003 is now recorded. |
| H3. Rows and files | `Scripts/extraction/kit/append.py B## --remove P###` deletes the rows of a paper set to Exclude (it refuses any other paper); used once (P155). No other excluded paper had rows. `corpus.py sync` moved the 10 PDFs of the excluded papers to `PDF_Library/Excluded_Full_Text` (P003 has none). Every script of B01 to B08 was re-run (`append.py B## --all`) and all checks pass. |
| H4. PRISMA | Recomputed after the decisions (numbers above). |
| H5. Transfer pairs | No change: 193 pairs, 120 complete; 73 without an in-domain value stay out of paired figures (P005 6, P099 1, P114 1, P118 4, P133 7, P151 54). |

### C. Policy for dataset keys and relations between papers (applied on 2026-10-14)

Why the keys exist (protocol V9): to count benchmark use with one rule, to compare results on the same benchmark, and to catch
numbers that two papers print identically (for example, two reports that print the same 64.84 percent wmAP). Proposal, so that each paper is one independent data point:

1. A dataset key records the dataset as the paper itself names or cites it: public benchmarks by their name (shared by every paper that
   names them, because that is what each paper states); the paper's own data get a key of their own, never merged with another paper's
   on the basis of authors, places or matching numbers. Keys merged by inference (for example `own_p125_qingdao_pwd` for P129) would be
   split.
2. Notes that compare one paper with another (matching class counts, shared authors, the same county) are removed from the scripts and
   from `datasets.csv`; they are not used by any analysis.
3. `copied_from` stays only where the paper itself says that a value comes from another work; those values stay out of the statistics.
4. Manuscript wording: each paper is treated as an independent data point, datasets are identified as the papers name them, and values a
   paper quotes from earlier work are not counted.

### Decisions applied on 2026-10-15: date rule

The author's rule: anything dated 2026 is excluded; the published version is preferred to a preprint and is used when it is dated
2025 or earlier; a paper whose version of record is dated 2026 stays only if a 2025 preprint is the extracted version.
Crossref gives a 2026 version of record for seven included papers; arXiv and the search's arXiv records were checked for 2025
preprints.

| Paper | Version of record | 2025 preprint | Outcome |
|---|---|---|---|
| P033 BARE | CCIS book chapter, online 13 February 2026 | none | Excluded (Outside date range); cited with FRTS in Future directions |
| P045 FM-SAM | Computers and Electronics in Agriculture, January 2026 issue | none found | Excluded (Outside date range) |
| P061 Sonneratia apetala | Forest Ecology and Management, February 2026 issue | none found | Excluded (Outside date range) |
| P112 LGINet | Information Fusion, May 2026 issue | none found | Excluded (Outside date range) |
| P128 drought phenotyping | Expert Systems with Applications, March 2026 issue | none found | Excluded (Outside date range) |
| P114 multispectral ALS benchmark | ISPRS J. 233 (2026), the extracted version | arXiv 2504.14337, 19 April 2025 | Excluded for now; re-included if the author obtains the preprint (values then extracted from it) |
| P101 WHU-STree | ISPRS J. (2026) | arXiv 2509.13172, 16 September 2025, the extracted version | Kept; the bib cites the arXiv version |

P045, P061, P112 and P128 were online in 2025 (the database searches retrieved them before 1 January 2026); under the earlier
reading (the online date counts) they would have stayed. Their rows were removed (`append.py --remove`), their scripts moved to
`scripts/excluded/` and the dataset keys used only by them to `scripts/excluded/datasets_removed.csv` (both deleted on 2026-10-16
at the author's request). PRISMA: databases 170
assessed, 43 excluded (5 outside the date range), 127 included; other methods 10 assessed, 4 excluded (P033), 6 included; 133
reports, 132 studies. Of the cross-check sample, 189 values come from the 133 papers of the final corpus, all correct (one-sided 95
percent upper bound 1.57 percent; `crosscheck.py summary`).

Published versions for papers extracted from preprints: P030 (TGRS 2023), P035 (NeurIPS 2025), P078 (ICCV 2025), P117 (IGARSS
2023) and P135 (JAG 2025) had been extracted from their arXiv versions. On 2026-10-16 the author replaced the five PDFs with the
published versions; the text was re-cached and every row checked again against them:
- P030: the published tables are images; all 106 table values are found again by OCR on their pages (locations now say "(image)"),
  and the quotes of 8 text fields were updated to the published wording and pages.
- P035: every value is unchanged; the published Table 1 adds Mask2Former baselines (Swin-base and Swin-L, RGB and RGB plus DSM),
  now extracted (4 models, 12 results), with the trainable parameters of Table 10 (p.23); three quotes moved (p.4, p.6, p.16).
- P078: the dataset statistics (forest types, species, countries, sensors, tree counts of the test sets) are in the arXiv appendix
  and, for the ICCV paper, in its separate supplementary material, not in the cited main text; those fields are now not reported or
  cited from the main text, and the transfer targets are named as in the main text (Wytham woods, species absent from training;
  LAUTx, collected using MLS).
- P117: unchanged (all quotes and the 6 image values found on the published pages).
- P135: the published Table 2 (p.8) prints the same results rounded (for example 0.5895 becomes 0.59) with three more columns; the
  values are now as printed in the published version, and 10 quotes were updated to the published wording.
All five pass corpus.py check, verify.py, linkcheck.py, pagecheck.py and ocrcheck.py.

Study categorisation (author decision of 2026-10-16, option A): a study's architecture family is the family of the transformer it
proposes; if it proposes none, that of its best-scoring transformer on its preferred metric; if no transformer has a comparable
score, the most frequent family among the transformers it uses. Result for the 132 studies: 93 by a proposed transformer, 29 by
the best-scoring transformer baseline and 10 by the transformer used; Pure ViT 19, Hierarchical ViT 38, CNN-transformer hybrid 52, foundation model 19, vision-language model 4.

### Author cross-check (next step)

`Scripts/extraction/crosscheck.py sample` drew 200 recorded numbers at random (seed 20261014) into
`Literature_Extraction/Extraction/crosscheck/crosscheck_sample.xlsx`: one result from every one of the 139 papers (a primary result
where the paper has one) and 61 further numbers drawn by kind (20 results, 8 transfer values, 13 model sizes or costs, 13 dataset
sizes, 7 GSD or point density values); 42 of the 200 were read from images. Each row gives the page, the place, a plain description
of the value and the quote to look for; the author marks it OK, Wrong or Not found. Rule: every value must be correct. A value marked
Wrong or Not found is corrected in its script, the whole paper is checked again, the same kind of value is checked across its batch,
and a new sample is drawn for that batch. `crosscheck.py summary` counts the marks and writes `crosscheck/crosscheck_result.json`;
if all 200 are correct, the one-sided 95 percent upper bound on the error rate of the sampled values is 1.5 percent (for the
methodology section).

Outcome (2026-10-15): all 200 values are correct (`crosscheck/crosscheck_result.json`: 200 of 200 OK from 139 papers; one-sided 95
percent upper bound on the error rate 1.49 percent). The author noted three location problems, none affecting a value: P084 cited
Sec. 3.4 for the canopy-area RMSE of Sec. 3.3 (corrected); P073 cited p.17, where the text layer holds a hidden copy of p.18 (corrected
to p.18); P079 cites p.9, the ninth page of the file, which prints the number 11 (the page convention, now stated in the protocol and
on the sample sheet). Prompted by P073, every quote that the text layer holds on several pages, or off its cited page, was checked
for visibility by OCR (`kit/pagecheck.py`, new): 88 page citations in 8 papers pointed to a hidden copy or a neighbouring page and were
corrected (P032 10, P056 1, P121 1, P128 4, P135 1, P073 66, P124 4, P131 1); all 912 such quotes are now visible on their cited page.
No value changed.

## 0. Reconciliation checklist (B01 to B08)

Prepared after B08 (2026-10-13); the decisions taken are in "Decisions applied on 2026-10-14" above. Decide everything from this section. Each row is one decision: the question, my recommended default (a recommendation only, nothing
is applied), what changes if you choose otherwise, and where the evidence is. The recommendations apply the five screening criteria
that `Scripts/search/screening.py` gives the screeners: (1) at least one transformer-based model is trained or tested; transformers used
only as baselines count, generic attention modules alone (SE, CBAM, coordinate attention) do not; (2) the unit of analysis is the
individual tree (detect, count or locate; delineate or segment crowns or trees in 2D or 3D; segment components; classify individual
trees or crowns by species, health, disease or mortality; estimate attributes), and general-purpose models qualify if evaluated on at
least one such task; (3) remote sensing or close-range imagery of trees; (4) a quantitative evaluation; (5) English, 2017 to 2025.
Area-level mapping (cover or extent maps, stand maps, disturbance maps, wall-to-wall canopy height or biomass maps, pixel-wise species
maps made like land-cover classification) is excluded.

State after B08 (from `compile.py`): 137 of the 150 included papers are extracted (3,091 results, 1,377 primary, 143
transformer-versus-baseline comparisons, 191 transfer pairs, 57 value choices). The other 13 are the 12 proposed exclusions (no rows)
and P003 (no PDF). "Rows removed" in a table means the paper's rows must also leave the workbook if you exclude it; see "How to apply
my decisions" at the end of this section, because changing `papers.csv` alone does not remove them from the compiled tables.

Decisions that touch the most papers, in the order I would take them: B1 (pixel-level crown condition masks, 10 papers), D1 (split not
stated, 27 papers), A1 (12 proposed exclusions), D3 (single-head attention models), C1 (shared `mopad` key).

### A. Inclusion

A1. Proposed exclusions (12 papers, no rows exist for any of them; all are `Pending` and counted as included):

| ID (batch) | Question | Recommended default | If you decide otherwise | Evidence |
|---|---|---|---|---|
| P006 (B04) | Four classifiers (EfficientNet, DenseNet, InceptionResNetV2, ViT) label 224 x 224 aerial images palm or non-palm tree; no individual tree is detected, counted or located. Keep? | Exclude (criterion 2). | Keep: set `Include` and extract it; the values would be image-level classification metrics of the four models. | Part 1 row P006; `reports/B04.md` |
| P028 (B04) | Wall-to-wall pixel-level olive tree map of six provinces, scored by pixel and sample-point F1 and overall accuracy, though the paper calls the output individual olive trees and compares transformers with CNNs. Keep? | Exclude (area-level mapping made like land-cover classification). Closest call of B04. | Keep: extract the pixel and sample-point scores of the map. | Part 1 row P028; `reports/B04.md` |
| P037 (B04) | A DeepLabv3+ CNN segments canopy pixels and SAM only partitions the mask into crown polygons; accuracy is pixel-based canopy loss and loss area per property. Keep? | Exclude (criterion 1: no transformer is trained or scored; area-level mapping). | Keep: extract the canopy loss accuracy values. | Part 1 row P037; `reports/B04.md` |
| P038 (B05) | The four scored models are ResNet-based Mask R-CNN variants; SAM-HQ (ViT-H) only helps annotate and is not scored. Keep? | Exclude (criterion 1: SAM-HQ is neither trained nor tested). | Keep: extract the four Mask R-CNN rows (all CNNs). | Part 1 row P038; `reports/B05.md` |
| P046 (B05) | CCD-YOLO is convolutional: CReToNeXt is a transformer in its name only and the dynamic head uses convolution-based attention; no multi-head self-attention block. Keep? | Exclude (criterion 1). Closest call of B05. | Keep: extract Tables 2 and 3; the proposed row is 81.2, 74.1, 77.5, 82.9 (boxes) and 82.4, 72.8, 77.9, 83.0 (masks). | Part 1 row P046; `reports/B05.md` |
| P055 (B05) | SPGN is a graph convolution network; SAM appears only in a qualitative figure and the scored comparison models of Table IV are convolutional. Keep? | Exclude (criterion 1). | Keep: extract the comparison table (Table IV). | Part 1 row P055; `reports/B05.md` |
| P070 (B06) | The only learned model is FastSAM, a YOLOv8-seg CNN that imitates SAM; trunk diameter comes from circle fitting and canopy volume from a convex hull. Keep? | Exclude (criterion 1: FastSAM is convolutional; the criteria name SAM, not FastSAM). | Keep: extract the trunk and canopy measurement results of the paper. | Part 1 row P070; `reports/B06.md` |
| P071 (B06) | A fine-tuned PoinTr (a transformer) completes 1 m3 tree segments, scored by Chamfer distance and F-score; no tree is measured (the effect on height or volume is named as future work). Keep? | Exclude (criterion 2). | Keep: extract the Chamfer distance and F-score tables (completion only, no tree-level score). | Part 1 row P071; `reports/B06.md` |
| P089 (B06) | A transformer-based completion network is trained and scored (Chamfer distance, F-score) and the only individual-tree score is the species accuracy of a PointNet++ classifier trained on the augmented data (Table 4: 87.2, 88.1, 90.1). Keep? | Keep: criteria 1 and 2 hold when read independently (a transformer is trained and scored, a single-tree species classification is scored), consistent with the kept P092, P098, P109, P103 and P104. I proposed exclusion under the stricter reading that the scored tree-level model must itself be a transformer. | Exclude: no rows exist, only `papers.csv` changes. Keep: extract the per-species accuracies of Table 4. | Part 1 row P089; `reports/B06.md` |
| P097 (B07) | PointRes (ResMLP replaces self-attention) and a Point Transformer baseline are scored by mAcc and mIoU over all classes of S3DIS and DALES; no individual tree is delineated, detected or measured. Keep? | Exclude (criterion 2). | Keep: extract the class-averaged values of Tables 2 and 3 (PointRes 71.6 mIoU on S3DIS). | Part 1 row P097; `reports/B07.md` |
| P155 (B04) | General vision-language model for Earth observation that you added; its only tree result is an m-NeonTree ablation (three values, metric not named). Keep? | Keep: criterion 2 admits general-purpose models evaluated on one tree task, and you added it as such; the values would be non-primary (metric `other`). | Exclude: only `papers.csv` changes. Keep: extract Table 6 (SigLIP 52.3, DOFA-CLIP-B without MaKA 58.1, with MaKA 59.0). | Part 1 row P155; `reports/B04.md` |
| P156 (B04) | Qualitative demonstration of an LLM expert system with SAM tree segmentation; no quantitative result (judged by eye in Fig. 2). Keep? | Exclude (criterion 4). | Keep: nothing to extract. | Part 1 row P156; `reports/B04.md` |

A2. Kept papers that a stricter reading would catch (nothing changes unless you exclude; "rows removed" counts the rows that would leave):

| ID (batch, rows) | Question | Recommended default | If you decide otherwise | Evidence |
|---|---|---|---|---|
| P098 (B07, 215) | A completion network; its tree scores (height, DBH, crown width) use reference values computed from the complete clouds, not field data. | Keep (a transformer-based network is trained and tree attributes are scored). | Exclude: rows removed. | `reports/B07.md`, closest calls |
| P109 (B07, 192) | Species classification is by ResUnet CNNs; SAM ViT-H only delineates crowns zero-shot and is scored by a boundary F1 (98 percent). | Keep (SAM is evaluated and part of the extraction; contrast P038, where SAM was not scored). | Exclude: rows removed. | `reports/B07.md` |
| P125 (B07, 219) | The proposed RCANet is a CNN; SegFormer and Swin-UNet are baselines. | Keep (baselines count under criterion 1). | Exclude: rows removed. | `reports/B07.md` |
| P103, P104 (B07, 139 and 189) | SAM is one stage (P103) and ViT is a comparison classifier (P104); the proposed models are CNNs. | Keep. | Exclude: rows removed. | `reports/B07.md` |
| P092 (B06, 167) | A completion transformer whose DBH, height and crown width are scored against field data. | Keep. | Exclude: rows removed. | `reports/B06.md`, part 1 row P071 |
| P137 (B08, 68) | The only scored learned output is the street tree detection of Table 1 (Mask2Former has the best F1); the vitality index is computed without a reference and the scored unit and split are not stated. | Keep (Mask2Former is scored on tree detection). Closest call of B08. | Exclude: rows removed. | `reports/B08.md`, closest calls |
| P140 (B08, 78) | One SegFormer-B5 model, no baseline, scores standing deadwood pixels (dead crowns and partial canopy dieback) with crowd-sourced labels. | Keep, unless you exclude the pixel-level group (B1). | Exclude: rows removed. | `reports/B08.md` |
| P143 (B08, 291) | A multimodal router selects a segmentation model; the scored dead tree polygons match by any overlap and can be groups of trees; SegFormer and Mask2Former are baselines inside the pipelines. | Keep. | Exclude: rows removed. | `reports/B08.md` |
| P131, P136, P144 (B08, 392, 436 and 270) | The transformer is only a baseline or comparison (a fine-tuned SAM, DETR, Deformable DETR); the best or proposed model is a CNN. | Keep (baselines count under criterion 1). | Exclude: rows removed. | `reports/B08.md` |

### B. Scope groups (one decision per group)

| Group | Question | Recommended default | If you decide otherwise | Evidence |
|---|---|---|---|---|
| B1. Pixel-level masks of discolored or dead crowns | P125, P129, P130 (B07), P131, P133, P140, P142, P151 (B08), P152 (B04) and P135 (B03) score pixel masks of diseased or dead crowns or dead tree pixels, tagged `health_classification` or `mortality_detection` at level `pixel`. Do they stay? | Keep all: criterion 2 includes segmenting tree crowns and classifying crowns by health or mortality, and the area-level exclusion names cover, extent and disturbance maps, not crown condition masks. | Exclude the group: ten papers leave (rows removed; P125 219, P129 281, P130 853, P131 392, P133 480, P140 78, P142 209, P151 1582, P152 512, P135 see B03). | `reports/B07.md`, `reports/B08.md`; part 2 question 2 |
| B2. Component and point-level papers | P064, P065, P067 (B05), P073, P079, P088, P093, P095 (B06) segment stem, branch, wood or leaf of trees. Do they stay? | Keep: criterion 2 lists tree components. | Exclude: rows removed for the chosen papers. | `reports/B05.md`, `reports/B06.md` |
| B3. Tree points without individual trees | P086 (B06) extracts tree points from urban multispectral LiDAR (tree-class IoU; the mIoU includes the non-tree class) and has no individual trees; P068 (B06) also segments individual trees. | Exclude P086 (closest to the area-level exclusion), keep P068. | Keep P086: nothing changes. Exclude P086: 189 rows removed. | `reports/B06.md` |
| B4. Close-range and non-aerial imagery | P043 (street-view and smartphone photographs), P049 and P065 (close-range images). | Keep: criterion 3 names close-range and ground-level tree images. | Exclude: rows removed. | `reports/B05.md` |
| B5. General-purpose models | P143 (B08) and P154 (B04, only its detection task is recorded) are evaluated on a tree task; P155 and P156 are in A1. | Keep P143 and P154 (criterion 2). | Exclude: rows removed. | `reports/B04.md`, `reports/B08.md` |
| B6. Completion papers | P071 and P089 are proposed (A1), P092 and P098 are kept. | One rule: keep when a transformer is trained and a tree-level score is computed (P089, P092, P098), exclude P071 (no tree-level score). | Stricter rule (the scored model must be the transformer): P089 and P092 and P098 are decided together. | `reports/B06.md`, `reports/B07.md` |

### C. Dataset keys to merge or split

| Key or papers | Question | Recommended default | If you decide otherwise | Evidence |
|---|---|---|---|---|
| C1. `mopad` (P138, P144, P147; B08) | One key for three papers: P138's healthy and unhealthy counts (114,842 and 1,884) equal the Healthy and Yellowish counts of MOPAD Site 2 in P144 (the same 2,303 images); P147 combines Sites 1 and 2 (about 363,877 trees). | Keep merged. | Split: change `dataset_key` in the scripts and `datasets.csv` (groupings only change). | `reports/B08.md`; `datasets.csv` |
| C2. P141 Longyou keys and `own_p027_longyou` (P027, B01) | Same county and a shared author (Qing Li); the papers do not say the images are the same. | Keep separate. | Merge: set P141's two Longyou keys to `own_p027_longyou`. | `datasets.csv` notes |
| C3. `own_p143_naip_mortality` and `treefinder` (P143, P151) | Same group and the same NAIP and Google Earth Engine labelling approach (75 against 1,000 sites). | Keep separate. | Merge: set P143's key to `treefinder` if its sites are a subset. | `datasets.csv` notes |
| C4. `own_p136_dahuofang_early` and `own_p123_kp_pwd`, `own_p123_cp_pwd` (P136, P123) | Same forest (Dahuofang), different UAVs; the papers do not say images are shared. | Keep separate. | Merge if you know the images are shared. | `datasets.csv` notes |
| C5. `own_p133_yantai_h2` and `own_p129_yantai_pwd` (P133, P129) | Both Yantai, Shandong; hyperspectral strips against RGB tiles. | Keep separate. | Merge if the same site and flights. | `datasets.csv` |
| C6. `own_p125_qingdao_pwd` (P125, P129; B07) | Merged because both use 2,981 Qingdao tiles from the same authors and equipment; they print 17,458 and 17,000 raw images and split 8:1:1 and 9:1. | Keep merged. | Split the key if you do not accept them as the same data. | `reports/B07.md` |
| C7. `forestsemantic_evo` (P081, P095) and `forestsemantic_ms` (P063) | Evo (Finland) against the multi-scan variant; the registry keeps them apart. | Keep separate. | Merge if the two share the plots. | `reports/B06.md` |
| C8. `stpctls` and `lewos` (P093 inferred; P102, P107 and P088 use them) | P093 names none of its datasets; the keys come from its citations ([52] Xi et al., [53] Momo Takoudjou et al.). | Keep. | Rename or split if the citations are wrong. | `reports/B06.md` |
| C9. `xi_2024_als_canada` (P066, P094) | P094's ten ALS plots are the dataset P066 cites as Xi et al. (2024); P066 uses four plots. | Keep merged. | Split if not the same data. | `reports/B06.md` |
| C10. `neon_aop_rgb_p059` (P059) and `neon_treeevaluation` (P050) | P059's NEON data may be the NeonTreeEvaluation benchmark. | Keep separate. | Merge if it is. | `reports/B05.md` |
| C11. `weiser_2022_als_trees` (P117) and `als_four_plots_p096` (P096) | P117's key is inferred from its reference [15]; P096 uses plots G1 and G2 of the same publication, so plots may overlap. | Keep apart. | Merge if the plots are the same. | `reports/B07.md` |
| C12. P114's two keys and P149's key | P114's HeliALS and Optech Titan keys hold the same 6,326 segments and one Zenodo record; P149's source is not identified (`forest_health_uav_p149`, kind `other`). | Keep. | Merge P114's keys if you treat them as one dataset. | `reports/B07.md`, `reports/B04.md` |

### D. Tag choices to confirm

| Decision | Question | Recommended default | If you decide otherwise | Evidence |
|---|---|---|---|---|
| D1. Split not stated (27 papers) | These papers have primary results with `split = not_stated`, so they are left out of the figures and form no transformer-versus-baseline comparison (protocol 6.4): P017 (B01), P083 (B02), P004, P026, P154 (B04), P039, P052, P067 (B05), P069, P073, P081, P093, P094, P095 (B06), P098, P099, P100, P115, P117 (B07) and P131, P137, P138, P141, P143, P144, P146, P147 (B08). | Keep `not_stated` (no inference). | Per paper, change the split tag in the script. Likely splits in B08: P141 test (Table 8 equals the Sanmen test row of Table 10), P143 test (fixed 70, 10 and 20 split), P144 and P146 validation (no test set, or a test set scored only for one model); P131, P137, P138 and P147 have no basis. P141, P143, P144 and P146 would then form comparisons. | `reports/B08.md`; batch reports |
| D2. Level not stated | P065, P067, P109, P122 and P137 have primary results with `level = not_stated` (the unit scored is not stated). | Keep `not_stated`. | Set a level per paper if you know it. | batch reports |
| D3. Single-head self-attention models | The DSI versions of P134 (spatial self-attention layer) and NLNet, CCNet, DANet and GCNet of P142 are `cnn_transformer_hybrid` (protocol 4.5: self-attention layers added to a CNN). | Keep `cnn_transformer_hybrid`. | If only multi-head blocks count, set them to `cnn`: P134 then has DETR and RT-DETR as its only transformers, and P142's comparison keeps SegFormer-b2 alone against FCN and DANet. | `reports/B08.md`; P134 and P142 scripts |
| D4. Overall scores that include non-tree classes | They are never primary (protocol 6.3): P045 (B05), P047 (B05), the headline of P152 (B04), the mIoU of P068, P073, P086, P094, P095 (B06), the mIoU of P142 and the All columns of P151 (B08). | Keep non-primary. | To make one primary, change the tag in its script (P045: `scripts/B05/P045.py`). | batch reports |
| D5. Primary settings chosen in B08 | P131 RGB scheme (shared by all models, SAM included); P133 strip 2 (first listed, no overall value); P138 healthy-class F1 (first listed class); P140 mean of four biomes; P143 BigTreeForest (first listed scenario); P142 tree-count precision and recall (mIoU is not primary); P151 random 80/20 split only (scenarios are not primary). | Keep as recorded. | Change `primary=` and the setting in the script, then re-run the checks (one primary setting per task and level is enforced). | `reports/B08.md`, Other choices |
| D6. Task and level tags chosen in B08 | P137 `detection` (the vitality is not scored); P138, P144 and P147 palm condition classes as `health_classification`, the Dead class of P144 as `mortality_detection`; P143 level `instance_mask` (polygons matched by any overlap); P131 success rate as `recall` at object level; P145 box results as `object` and mask results as `instance_mask`. | Keep as recorded. | Change the tag in the script. | `reports/B08.md` |
| D7. Metric identity in B08 | P146's accuracy rate 73.1 equals 463 of 633 labels and is `recall`; P142's producer and user accuracy are `recall` and `precision` (their definitions); P145's mAP is the COCO mAP (`ap50_95`); P144's overall F1 is the mean of the five per-class F1. | Keep as recorded. | Change the metric in the script. | `reports/B08.md` |
| D8. B04 to B07 tag choices | P152 headline averages background and tree (non-primary); P114's macro-average accuracy is `recall`; P120's weighted recall is `oa` (the paper's overall accuracy); P115's AUC and P109's boundary F1 are `other`; P130 records all 18 models; P123, P124 and P125 evaluation sets are `validation` (the papers call them so). | Keep as recorded. | Change the tag in the script. | part 2 question 5; `reports/B07.md` |

### E. Identical or copied rows

| Papers | Question | Recommended default | If you decide otherwise | Evidence |
|---|---|---|---|---|
| E1. P125 and P129 (B07) | Same authors and Qingdao data; they print an identical Swin-UNet row (IoU 0.665, F1 0.799, precision 0.817, recall 0.781, 98.30 M parameters, 26.33 GFLOPs) while every other model differs. | Keep both as reported, no `copied_from` (neither says the row was taken from the other). | Set `copied_from` in `scripts/B07/P129.py`; the row then leaves the cross-paper statistics. | `compile_review.csv`; `reports/B07.md` |
| E2. P147 (B08) | The MOPAD row (reference [7]) and the YOLOv5, YOLOv7, YOLOv8 and HR-YOLOv8 rows (reference [20]) are taken from other papers; they carry `copied_from` because the paper says so. | No action. | none | `reports/B08.md` |
| E3. Already handled | P066's ForAINet numbers and P100's two cited methods are `copied_from`; P035 repeats 28 results of P032 (same study, counted once); one equal value of P011 and P018 was reviewed as a coincidence. | No action. | none | `compile_review.csv` |

### F. Value choices whose rule application is debatable

The other value choices (57 in all, listed by batch in part 3) follow protocol rules V1 and V2 and need no action.

| Paper | Question | Recommended default | If you decide otherwise | Evidence |
|---|---|---|---|---|
| P084 (B06) | The abstract, the text and Table 2 print three sets of height errors, and the Average row of Table 3 differs from the mean of the printed per-tree values; five cf items choose the table and text values over the abstract and the per-tree means over the Average row (rule V2). | Keep. | Use the other printed value in the script and swap the cf item. | `compiled/conflicts.csv`; part 3 B06 |
| P150 (B04) | Parameter counts: the 5-band whole-detector count is used over the 3-band backbone-sized count (V1b); the Swin Transformer value (924.97 M) equals its GFLOPs. | Keep. | Use the 3-band counts. | part 3 B04 |
| P116 (B01) | The printed average F1 (94.99) is not the mean of the three printed folds (91.66); the folds are the transfer pairs (V2). | Keep. | Record the printed average. | part 3 B01 to B03 |
| P138 (B08) | The abstract's 276,883 annotations match neither the class counts nor their sum (116,726); the sum is used (V1c). | Keep. | Use the abstract value. | `reports/B08.md` |

### G. Source values that look like errors (information only; recorded as printed or not recorded)

| Paper | Observation | Evidence |
|---|---|---|
| P150 (B04) | Swin Transformer 5-band parameters 924.97 M (equal to its GFLOPs). | `reports/B04.md` |
| P045 (B05) | Sophora japonica recall 91.83 in Table 2 against 89.83 in the text and Table 3 (not recorded). | `reports/B05.md` |
| P124 (B07) | The abstract's "3.89 MFLOPs" is a parameter count. | `reports/B07.md` |
| P126 (B07) | The text says the same FPS as the original Faster-RCNN, Table 6 gives 10.8. | `reports/B07.md` |
| P102 (B07) | Table II total 998 against 728 in its rows (not recorded). | `reports/B07.md` |
| P130 (B07) | SETR precision and F1 are both printed as 84.52 on the Hubei data. | `reports/B07.md` |
| P137 (B08) | The abstract says U-Net had the best performance; Table 1 ranks Mask2Former first on all metrics. | `reports/B08.md` |
| P133 (B08) | The PWD proportion of strip 1 is 0.013 in the text and 0.041 in Table 1. | `reports/B08.md` |
| P141 (B08) | The recall of YOLOv3-tiny is printed as 715 (not recorded). | `reports/B08.md` |
| P153 (B08) | Shanxi Province (Sec. 2.2) against Shaanxi (Sec. 2.1); the F1 gain over YOLOv10 is printed as 1.4 points, Table 5 gives 2.4; 21,752 augmented images against 16,552 plus 5,191. | `reports/B08.md` |
| P147 and P144 (B08) | P147 prints F1 63 percent for the MOPAD baseline; P144 quotes 87.91 percent (Site 1) and 99.04 percent (Site 2) for the same source. | `reports/B08.md` |
| P138 (B08) | The abstract's 276,883 annotations match no printed count. | `reports/B08.md` |

### H. Technical items

| Item | Observation | Recommended default | Evidence |
|---|---|---|---|
| H1. P003 (B04) | No PDF in the library (open access, PubMed Central PMC8783740; the publisher blocks automated downloads); it cannot be extracted until you supply the PDF. | Supply the PDF, then `corpus.py file` and extract. | part 1 |
| H2. compile.py limits | It cannot take a negative kappa (P047 baseline -0.003 is not recorded) and does not read the review kind `percent_below_one`, so seven P154 warnings still print although the values are correct. | Extend compile.py or accept both. | part 2 question 6 |
| H3. Rows of an excluded paper | `corpus.py check` reports every row of an excluded paper; `compile.py` builds results, models, datasets, comparisons, transfer pairs and value choices from all workbook rows and skips non-review papers only in `studies.csv`; no tool deletes a paper's rows (read from the code, see "How to apply"). | Remove the rows as described under "How to apply my decisions". | `Scripts/extraction/corpus.py`, `compile.py`, `kit/append.py` |
| H4. PRISMA while papers are Pending | `screening.py prisma` counts a Pending database paper as neither assessed nor included (it prints a warning) and a Pending paper found by other methods as assessed but neither included nor excluded, so the flow counts balance only after all 12 decisions. | Run `prisma` after your decisions. | `Scripts/search/screening.py` |
| H5. Transfer pairs without an in-domain value | 73 of the 191 transfer pairs have no in-domain value with the same metric and stay out of paired figures: P005 (6), P099 (1), P114 (1), P118 (4), P133 (7) and P151 (54); 118 pairs are paired. | Keep (rule V8). | `compile.py` notes |

### How to apply my decisions

Verified by reading `Scripts/extraction/corpus.py`, `Scripts/extraction/compile.py`, `Scripts/extraction/kit/append.py`,
`Scripts/search/screening.py` and `Literature_Extraction/Extraction/protocol.md`; steps marked "unverified" were not run. Run every
command from the project root with `Scripts/.venv/bin/python` (`py` below). No decision has been applied.

1. Exclude a paper (a proposed one or a kept one).
   - `Literature_Extraction/Corpus/papers.csv`: set `ft_decision` to `Exclude` and `ft_reason` to a reason that starts with one of the
     reasons in `EXCLUSION_REASONS` of `screening.py` (for example "No transformer-based model trained or evaluated", "Area-level
     mapping, not individual trees", "Not a tree task", "No quantitative evaluation"), optionally followed by " (detail)"; the text before
     " (" is the category that `screening.py prisma` and the tracker count. Remove the "(proposed) " prefix.
   - `py Scripts/extraction/corpus.py sync` moves the PDF to `PDF_Library/Excluded_Full_Text` and updates `pdf_file`.
   - If the paper has rows: remove them with `py Scripts/extraction/kit/append.py B## --remove P###` (added on 2026-10-14; it only
     deletes rows of papers set to Exclude in `papers.csv`). Without this, `corpus.py check B##` reports each row of an excluded paper
     ("paper is not in the review") and `compile.py` keeps them in `results.csv`, `models.csv`, `datasets.csv`, `comparisons.csv`,
     `transfer.csv` and `conflicts.csv` (only `studies.csv` skips non-review papers). Afterwards move or delete the paper's script,
     because `append.py B## --all` would write its rows again.
   - Dataset keys used only by that paper then print the warning "datasets.csv keys used by no paper" in `compile.py`; remove the key rows
     from `datasets.csv` or accept the warning.
   - Then run: `py Scripts/extraction/corpus.py check B##`, `py Scripts/extraction/kit/verify.py B##`, `py Scripts/extraction/kit/linkcheck.py B##`,
     `py Scripts/extraction/compile.py`, `py Scripts/extraction/corpus.py tracker` and `cd Scripts/search && ../.venv/bin/python screening.py prisma`.
2. Keep a proposed paper and have it extracted: set `ft_decision` to `Include` and clear `ft_reason`; then write
   `Literature_Extraction/Extraction/scripts/B##/P###.py` as the batch-extraction procedure describes, add any new dataset key to
   `datasets.csv`, run `py Scripts/extraction/kit/append.py B## P###` and the checks above. The skill works on whole batches and treats a
   batch as finished when its report says `Status: complete`, so for one paper either ask an agent to extract only that paper with the
   skill's per-paper steps or reopen the batch (unverified).
3. Merge or split a dataset key: edit `R("dataset_key", ...)` in the script of each affected paper, add or edit the row in `datasets.csv` (the
   key must exist there, or `compile.py` reports an error), run `append.py B## P###` for each paper, then the checks and `compile.py`; the
   list of shared keys printed by `compile.py` shows the result.
4. Change a tag or a primary choice: edit the `Tags(...)` or `primary=` arguments in the paper's script (B04 to B08). For B01 to B03 the
   protocol 6 tags come from `Literature_Extraction/Extraction/scripts/legacy_tags.py` (as the `append.py` header says), not from the script.
   Run `append.py`, then `corpus.py check B##` (it enforces one primary result per model, task, level and metric and one primary setting
   per task and level), `verify.py`, `linkcheck.py` and `compile.py`.
5. Set `copied_from`: add `copied_from=` to the `result(...)` call in the script, run `append.py` and `compile.py` (protocol V9: the result
   stays out of cross-paper statistics). If `compile.py` printed a "same value in several studies" note for it, update the matching row in
   `compile_review.csv` (kind `same_value`).
6. Accept the other printed value of a value choice (cf item): put the other value in the recorded value and the previously used value in
   the `conflict(...)` item (same page and quote rules), run `append.py`, the checks and `compile.py`; `compiled/conflicts.csv` is rebuilt.
7. After any change, also repeat the quote check for the paper (`verify.py` and `linkcheck.py` do not check that quotes are verbatim; the
   temporary `qc.py` helper of the B04 to B08 sessions did, it is not part of the project).
8. What `compile.py` does with an excluded paper's rows: it keeps them (see step 1) unless they are removed from the workbook.

## 1. Proposed exclusions (12 papers; decided on 2026-10-14: P038 and P089 included, the other ten excluded)

| Batch | ID | Title | Proposed reason | Note |
|---|---|---|---|---|
| B04 | P006 | Advancements in Integrating Deep Learning into Geographic Information Systems for Palm Tree Analysis | The four models (EfficientNet, DenseNet, InceptionResNetV2, ViT) are binary image classifiers (palm tree versus non-palm tree, Fig. 6) on 224 x 224 resized aerial images; the paper does not say that a sample is one tree and evaluates no individual-tree output (no boxes, counts or locations). |  |
| B04 | P028 | Sub-national scale mapping of individual olive trees integrating Earth observation and deep learning | The output is a wall-to-wall pixel-level olive tree map for six provinces (semantic segmentation of tree pixels constrained by 32 x 32 sub-patch olive or non-olive classification), scored by pixel and sample-point F1 and overall accuracy; no individual tree is delineated, detected or counted. | Closest call of B04: the paper calls its output "individual olive trees" and compares transformers with CNNs, but the map is evaluated on pixels and sample points, not on trees. |
| B04 | P037 | Detecting and measuring fine-scale urban tree canopy loss with deep learning and remote sensing | Area-level mapping of urban tree canopy loss: a DeepLabv3+ CNN segments canopy pixels, SAM only partitions the canopy mask into crown polygons as an intermediate step, and the accuracy is pixel-based for canopy loss and loss area per property; no individual-tree output is evaluated. |  |
| B04 | P155 | Geolangbind: Unifying earth observation with agglomerative vision-language foundation models | General vision-language foundation model for Earth observation (zero-shot scene classification and retrieval); the only tree-related result is an ablation on the m-NeonTree segmentation set whose metric and target are not described, and no individual-tree output is evaluated. | Added by you as a general-purpose model; its PDF is the arXiv version retitled DOFA-CLIP. If kept, the only numbers are the three m-NeonTree values of Table 6 (SigLIP 52.3, DOFA-CLIP-B without MaKA 58.1, with MaKA 59.0), whose metric is not named. |
| B04 | P156 | TREE-GPT: MODULAR LARGE LANGUAGE MODEL EXPERT SYSTEM FOR FOREST REMOTE SENSING IMAGE UNDERSTANDING AND INTERACTIVE ANALYSIS | Qualitative demonstration of an LLM expert system with SAM tree segmentation: no quantitative result is reported (segmentation is judged by eye against manual annotation in Fig. 2). | Added by you as a general-purpose model. It has no numbers to record. |
| B05 | P038 | Domain-Aware Transfer Learning With SAM-Assisted Mask R-CNN for Urban Tree Crown Delineation From UAV Orthophotos | The four evaluated models are Mask R-CNN variants with ResNet50 or ResNet101 backbones (CNNs that differ in pre-training); SAM-HQ (ViT-H) is used only as a semi-automatic annotation tool and is neither trained nor scored. |  |
| B05 | P046 | From Crown Detection to Boundary Segmentation: Advancing Forest Analytics with Enhanced YOLO Model and Airborne LiDAR Point Clouds | CCD-YOLO is a YOLOv8 variant whose modules are convolutional: CReToNeXt (re-parameterized convolutions and residual connections, a transformer only in its name), CBAM, and a dynamic head with convolution-based scale, spatial and task attention; no multi-head self-attention block is described, so no transformer is trained or evaluated (protocol 4.5). | Closest call of B05: the abstract calls CReToNeXt a "cross residual transformer network" and the dynamic head is described as a "self-attention mechanism". If kept: Tables 2 (crown detection and boundary segmentation counts) and 3 (the proposed row is 81.2, 74.1, 77.5, 82.9 for boxes and 82.4, 72.8, 77.9, 83.0 for masks). |
| B05 | P055 | Scale Pyramid Graph Network for Hyperspectral Individual Tree Segmentation | SPGN is a graph convolution network with a scale pyramid module and positional encoding but no transformer block; SAM appears only in a qualitative visual comparison (Fig. 13) with no scored result, and the scored comparison models of Table IV (Mask R-CNN, SS-FPN and SOLO variants) are convolutional. |  |
| B06 | P070 | Assessing ornamental tree maturity and spray requirements using depth sensing and LiDAR technologies | The only learned model is FastSAM, a YOLOv8-seg based convolutional network trained to imitate SAM, used to segment trunks in RGB images; trunk diameter comes from circle fitting and canopy volume from a convex hull, so no transformer is trained or evaluated (protocol 4.5). |  |
| B06 | P071 | Completing 3D point clouds of individual trees using deep learning | Point cloud completion of 1 m3 tree segments with a fine-tuned PoinTr (a transformer), scored by Chamfer distance and F-score against complete synthetic or real segments; no individual tree is detected, segmented, classified or measured, and the effect of completion on tree height, diameter or volume is named as future work (Sec. 4). | Contrast with P092 (kept): P092's DBH, height and crown width taken from the completed clouds are scored against field data; P071 names that as future work. |
| B06 | P089 | Research on Tree Point Cloud Enhancement Based on Deep Learning | Tree point cloud enhancement: a transformer-based completion network scored by Chamfer distance and F-score on completed point clouds (Tables 1 to 3), and an adversarial augmentation network whose only individual-tree score is the species accuracy of a PointNet++ classifier with attention pooling trained on the augmented data (Table 4: 87.2 original, 88.1 conventional augmentation, 90.1 proposed); the scored classifier is not a transformer and no tree is detected, segmented or measured. | Closest call of B06: it scores a species classification of single trees, but the scored classifier is PointNet++ with attention pooling and the transformer only completes the training samples. If kept: Table 4 per-species accuracies (average 87.2 original, 88.1 conventional augmentation, 90.1 proposed). |
| B07 | P097 | Tree Barriers and Power Lines Segmentation from Point Clouds Based on Improved Point Transformer | PointRes replaces the self-attention of Point Transformer with ResMLP layers, and it and the Point Transformer baseline are scored by mAcc and mIoU over all classes of S3DIS (indoor rooms) and DALES (eight classes, vegetation being one class); tree barriers are vegetation points near power lines, so no individual tree is delineated, detected, counted or measured. | A Point Transformer baseline is evaluated, so the proposal rests on scope (no individual-tree output), not on the absence of a transformer. If kept: Table 2 (S3DIS) and Table 3 (DALES) mAcc and mIoU over all classes (PointRes 71.6 mIoU on S3DIS, abstract); the tree barrier reports (Table 4) list coordinates without an accuracy. |

Not an exclusion, but open: P003 (B04, olive tree detection and counting in satellite imagery, Computational Intelligence and
Neuroscience 2022, DOI 10.1155/2022/1549842) has no PDF in the library. The paper is open access (PubMed Central PMC8783740) and is not
retracted; the publisher's servers block automated downloads. It cannot be extracted until you supply the PDF.

B08 adds no proposed exclusion: all 16 papers were extracted. The IDs in the table are exactly the papers with `ft_decision = Pending` in
`papers.csv` and in the list of `Paper_Tracker.md` (checked 2026-10-13). The recommended default for each row is in section 0, A1.

## 2. Questions that need your decision

1. **The 12 proposed exclusions above.** Closest calls: P028 (pixel map called individual trees), P046 ("transformer" in the name of a
   convolutional module, a "self-attention" dynamic head), P089 (scores a single-tree species classification, but only with a PointNet++
   classifier). P092 (B06) is kept because the tree attributes it scores come from the transformer's own output; if you apply a stricter
   reading (the scored model itself must be the transformer), P092 and P089 would be judged together. B07 kept papers that would fall
   under such a stricter reading: P109 (CNN species classification; SAM ViT-H only delineates crowns zero-shot and is scored by a boundary
   F1), P098 (a completion network whose tree scores use reference values computed from the complete point clouds, not field data; compare
   P071 and P089), P125 (the proposed RCANet is a CNN; SegFormer and Swin-UNet are baselines), P103 (SAM is one stage, ResNet-101 classifies)
   and P104 (ViT is a comparison classifier, EfficientNetB3 the proposed one). They are kept because a transformer is trained or scored in
   each (protocol 4.5). B08 adds no proposal; its closest calls among the kept papers are P137 (only the street tree detection is
   scored, the vitality index is not, and the scored unit and split are not stated), P140 (one SegFormer model, no baseline, deadwood
   pixels) and P143 (a multimodal router selects the segmentation model; dead tree polygons match by any overlap), and P131, P136 and
   P144, where the transformer (a fine-tuned SAM, DETR, Deformable DETR) is only a baseline and the best or proposed model is a CNN
   (the screening criteria say that transformers used only as baselines count).
2. **Scope of point-level and component papers.** Recorded under the protocol 6 tags, but not tree-level extraction in the strict sense:
   B05 P064 (stem regions of eucalyptus point clouds), P065 (leaves and branches for pruning, `component_segmentation`), P067 (trunk and
   branch); B06 P068 and P086 (tree points from urban multispectral LiDAR; P086 has no individual trees at all), P073, P079, P088, P093
   and P095 (wood, leaf, trunk and branch separation of single-tree point clouds); B07 P125, P129 and P130 (semantic segmentation of
   discolored pine crowns, recorded as `health_classification` at pixel level, like P152 in B04; P123, P124 and P126 detect infected or
   dead trees with boxes at object level); B08 P131, P133, P142 (masks of discolored pine crowns or pixels, `health_classification`
   at pixel level), P140 and P151 (dead tree or deadwood pixels, `mortality_detection` at pixel level) and P143 (dead tree polygons
   matched by any overlap, level `instance_mask`; P135 in B03 is also a pixel-level mortality paper). Decide whether these stay in the
   review.
3. **Close-range and non-aerial imagery** (B05): P043 (street-view and smartphone photographs), P049 and P065 (close-range images).
4. **Dataset keys that rest on citations or assumptions** (merge or split in `datasets.csv` if you know better):
   - P059 uses its own key `neon_aop_rgb_p059` for NEON data; it may be the NeonTreeEvaluation benchmark (`neon_treeevaluation`, P050).
   - P081 and P095 use ForestSemantic of Liang et al. (Evo, Finland; Zenodo 15080627; `forestsemantic_evo`); P063 uses ForestSemantic-MS
     (`forestsemantic_ms`). Merge them if the two share the plots.
   - P093 names none of its datasets; `stpctls` (six temperate datasets, citation [52], Xi et al.) and `lewos` (tropical set, [53]) are
     inferred from the citations. P088 uses the same `lewos` key.
   - P094's ten ALS plots are the ALS dataset that P066 cites as Xi et al. (2024); both use `xi_2024_als_canada` (P066 uses four plots).
   - P149's dataset source is not identified in the paper (key `forest_health_uav_p149`, kind `other`).
   - P125 and P129 share `own_p125_qingdao_pwd` (B07): both use 2,981 Qingdao tiles (same authors, DB-II fixed-wing UAV with a Sony Alpha
     7R II, 256 x 256 tiles); they print 17,458 and 17,000 raw images and split 8:1:1 and 9:1. Split the key if you do not accept them as the
     same data.
   - P117's `weiser_2022_als_trees` (B07) is inferred from its reference [15] (Weiser et al., multi-platform laser scanning in German
     forests, PANGAEA 942856). P096 (`als_four_plots_p096`) uses the Bretten plots G1 and G2 of the same publication, so P117's six plots
     (BR01, BR03, BR05, KA09, KA10, KA11) may overlap with them; the keys are kept apart.
   - P114's two keys (HeliALS and Optech Titan, B07) hold the same 6,326 segments and one Zenodo record (17077255).
   - `mopad` (B08) is shared by P138, P144 and P147: P138's healthy and unhealthy counts (114,842 and 1,884) equal the Healthy and Yellowish
     counts of MOPAD Site 2 that P144 prints (the same 2,303 images), and P147 combines Sites 1 and 2; P138's unhealthy class thus
     appears to be MOPAD's Yellowish class. Split the key if you do not accept them as the same data.
   - Kept as separate keys because the papers do not say that images are shared (merge if you know better): P141's two Longyou County
     datasets and P027's `own_p027_longyou` (same county, shared author Qing Li); P143's 75 NAIP sites (`own_p143_naip_mortality`) and
     TreeFinder (`treefinder`, P151; same group and labelling approach, 1,000 sites); P136 and P123 (Dahuofang, Liaoning); P133 and P129
     (Yantai, Shandong; hyperspectral strips against RGB tiles).
5. **Tag choices you may want to change.** P045 (B05): its overall scores include the background class, so nothing is primary; to keep
   them primary, change the tag in `scripts/B05/P045.py`. P152 (B04): the paper's headline F1 averages background and tree classes, so it
   is non-primary. B06: every mIoU that averages tree and non-tree classes is non-primary (P068, P073, P086, P094, P095). B07: P114's
   macro-average accuracy (mean species-wise recall) is recorded as `recall`; P120's weighted recall, which the paper defines as its
   overall accuracy, is recorded as `oa`; P115's AUC and P109's boundary F1 are metric `other`; P130 records all 18 models per table although
   the skill allows a subset above ten; the evaluation sets of P123, P124 and P125 are called validation sets by the papers (`validation`).
   P115 and P117 do not state the split of their results, so they form no comparison in `comparisons.csv`.
   B08: (a) 27 papers in all have primary results with the split `not_stated` and are left out of the figures and the comparisons
   (P131, P137, P138, P141, P143, P144, P146 and P147 in B08; the full list is in section 0, D1); P141, P143, P144 and P146 would form
   comparisons if you accept the likely split (test, test, validation, validation). (b) The DSI versions of P134 and NLNet, CCNet, DANet
   and GCNet of P142 (single-head self-attention models) are `cnn_transformer_hybrid`; set them to `cnn` if only multi-head blocks
   count. (c) The mIoU of P142 averages the diseased and the negative class and is not primary (its tree-count precision and recall
   are); P151's All columns average the dead tree and the background class and are not recorded. (d) Primary settings: P131 RGB scheme,
   P133 strip 2, P138 healthy-class F1, P140 mean of four biomes, P143 BigTreeForest, P151 random split. (e) P137 is tagged `detection`
   with level and split `not_stated`; P143 is level `instance_mask`; P146's accuracy rate is `recall` (463 of 633 labels); P145's mAP
   is `ap50_95`.
6. **compile.py limits** (the kit is not changed by agents): (a) it cannot take a negative kappa, so the P047 baseline (-0.003, B05) is
   not recorded; (b) it does not read the review kind `percent_below_one`, so seven P154 warnings (B04) still print although the values
   are correct (checked on the rendered page); (c) it keeps the rows of an excluded paper in the compiled tables unless they are removed
   from the workbook, and `corpus.py check` reports them (see section 0, "How to apply my decisions"); (d) 73 of the 191 transfer pairs
   have no in-domain value with the same metric and stay out of paired figures (B08 adds 61: P133 7, P151 54). Extend compile.py or
   accept these.
7. **Source values recorded as printed that look like typos:** P150 Swin Transformer 5-band parameters 924.97 M (equal to its GFLOPs;
   B04); P045 Sophora japonica recall 91.83 against 89.83 in the text (not recorded; B05); P124's abstract "3.89 MFLOPs" for the parameter
   count; P126's text "the same FPS as the original Faster-RCNN" against 10.8 in Table 6; P102's Table II total of 998 against 728 in its
   rows (not recorded; B07); P130's SETR precision and F1 are both printed as 84.52 on the Hubei data. B08: P137's abstract names U-Net
   best while Table 1 ranks Mask2Former first; P133's PWD proportion of strip 1 is 0.013 in the text and 0.041 in Table 1; P141 prints
   the recall of YOLOv3-tiny as 715 (not recorded); P153 prints Shanxi (Sec. 2.2) and Shaanxi (Sec. 2.1), a 1.4 point F1 gain over
   YOLOv10 where Table 5 gives 2.4, and 21,752 augmented images against 16,552 plus 5,191; P138's abstract gives 276,883 annotations,
   which match no printed count.
8. **Identical or copied rows in two papers (B07, B08).** P125 and P129 (same authors, same Qingdao data) print an identical Swin-UNet row (IoU 0.665,
   F1 0.799, precision 0.817, recall 0.781, 98.30 M parameters, 26.33 GFLOPs) while every other model differs. Neither says the row was
   taken from the other, so `copied_from` is not set and both count in the comparisons (reviewed as `kept` in `compile_review.csv`).
   If you treat P129's row as copied, set `copied_from` in `scripts/B07/P129.py`. B08: P147 prints five rows taken from other papers (the
   MOPAD row from reference [7], YOLOv5, YOLOv7, YOLOv8 and HR-YOLOv8 from reference [20]); they carry `copied_from` because the paper
   says so, and no equal-value warning arose. Its MOPAD F1 (63 percent) differs from the 87.91 and 99.04 percent that P144 quotes for the
   same source.

## 3. Judgement calls by batch

### B01 to B03 (back-fill of the protocol 6 tags)

Tags and categories:

- Random forest, SVM, k-means, watershed and SfM+MVS are `classical`; PoinTramba (P107) is `cnn_transformer_hybrid` (it has transformer
  encoders); two baselines of P112 whose architecture the paper does not describe are `not_reported`.
- P148: the five training runs are not said to be averaged, so its results are `aggregation = not_stated`.
- P001 had two headline rows (detection mAP50 and F1 of the same model); the mAP50 row is kept.
- Papers without primary results: P042 (overall scores average over non-tree scene classes, and the SAM-YOLOv8 pipeline is scored on
  "high-quality images" while the baseline is scored on the unmodified set), P051 (every in-domain value was read off a bar chart,
  so approximate), P082 (only rates of correct detection and segmentation, metric `other`).

Value choices:

| ID | Target | Used | Not used | Rule | Reason |
|---|---|---|---|---|---|
| P015 | dataset1.n_trees | 7112 (sum of the printed splits) | 7824 | V1c | The per-split counts in the same sentence add up to 7112 |
| P017 | gsd | 0.01 m | 5 m | V1c | Table 1 also prints spatial resolution 0.01 m, which matches the UAV flights |
| P022 | res31.value (MTCDNet, Dataset_B, mAP) | 48.41 | 45.31 | V1a | Table 2 is the main comparison; the text values are those of Early Fusion |
| P022 | res32.value (MTCDNet, Dataset_B, AP50) | 94.58 | 93.22 | V1a | as above |
| P027 | res2.value (RT-DETR mAP) | 91.5 | 89.5 | V1a | 89.5 is the baseline of the ablation study |
| P042 | res5.value (YOLOv8 mask AP50:95) | 0.472 | 0.434 | V1a | 0.434 is from the model-size comparison (Table 1) |
| P090 | dataset1.n_images | 7112 (sum of the printed splits) | 7239 | V1c | The per-split counts next to it add up to 7112 |
| P116 | Table 5 average validation F1 | (folds recorded) | 94.99 | V2 | Not the mean of the three printed folds (91.66); the folds are the transfer pairs |
| P135 | gsd | 0.25 m | 0.5 m | V1c | Table 1 and the annotation text give 25 cm |
| P148 | model1.total_params | 45.89 M (whole model, Table 3) | 29.0 M | V1b | 45.89 M goes with the model's 209.70 GFLOPs; 29.0 M matches the Swin-T backbone |

Notes from compile.py: P035 repeats 28 results of P032 (same study, counted once); one equal value in P011 and P018 was reviewed as a
coincidence (`compile_review.csv`); P118's four transfer pairs have no in-domain value with the same metric; P056 model5 prints FLOPs
without a unit. P155's PDF is arXiv 2503.06312v2, retitled DOFA-CLIP by its authors; the manuscript cites it as GeoLangBind.

### B04

Value choices:

| ID | Target | Used | Not used | Rule | Reason |
|---|---|---|---|---|---|
| P005 | res61.value (AdaTreeFormer, Jiangsu test set, MAE) | 122.8 | 15.9 | V1a | Table 2, the main comparison, gives 122.8; the abstract uses 15.9 as the reduction in absolute counting error (138.7 of FusionCount+FT minus 122.8) |
| P005 | res56.value (Baseline, London test set, F1) | 15.3 | 15.5 | V1a | Table 2 gives 15.3 for the Baseline; the 'w/o' row of the ablation Table 8 (no target labels) matches the Baseline in its other five values |
| P008 | res2.value (SwinTU-net, total prediction error) | 0.95 | 0.94 | V1a | Tables 6 and 7 (the comparison table), Sec. 4.5 and the conclusion give 0.95; the introduction, Sec. 4.4 and p.11 give 0.94 |
| P008 | res3.value (commission failure rate) | 0.98 | 0.97 | V1a | Table 7, the comparison table, gives 0.98 |
| P008 | res4.value (omission failure rate) | 1.3 | 1.2 | V1a | Table 7, the comparison table, gives 1.3 |
| P008 | res3.value | 0.98 | 708 | V1a | Table 6 prints 708 under a percent label (CFG), which cannot be a percentage; Table 7 gives 0.98 percent for CFR |
| P008 | res4.value | 1.3 | 913 | V1a | Table 6 prints 913 under a percent label (OFG), which cannot be a percentage; Table 7 gives 1.3 percent for OFR |
| P149 | res1.value (SWIN Transformer, accuracy) | 95 | 93 | V1a | Table III, the results table, gives 95%; the abstract's 93% equals the F1 score of Table III |
| P149 | res1.value | 95 | 85 | V1a | Table III gives 95%; the text of Sec. 5 gives a median category accuracy of about 85% |
| P149 | dataset1.n_images | 45022 (sum of the printed splits) | 45023 | V1c | The per-split counts of Table I add up to 45,022; the printed total is 45,023 |
| P150 | res37.value (ResNeXt-101, 5 bands, class weights, F1) | 85.70 | 85.68 | V1a | Table 7, the results table, gives 85.70 (the highlights on p.1 also say 85.70); the abstract and the conclusion give 85.68 |
| P150 | model1.total_params (ResNet-50) | 41.30 M (5-band, whole detector) | 26 M | V1b | Table 4 gives a second parameter count for 3-band images (backbone-sized); the 5-band count is the whole detector that goes with the printed GFLOPs |
| P150 | model2.total_params (ResNet-101) | 60.20 M | 44 M | V1b | as above |
| P150 | model3.total_params (ResNeXt-101) | 104.60 M | 89 M | V1b | as above |
| P150 | model4.total_params (Swin Transformer Base) | 924.97 M | 88 M | V1b | as above; the printed 5-band value equals the GFLOPs printed beside it (see question 7 in section 2) |

Tagging, primary-setting and recording choices (written in the `notes` field of each paper):

- P150: the primary setting is Table 6 (no class weights) on the 5-band images; the headline is the best Table 7 result.
- P152: the primary results are the UTC-class IoU and F1 on RGB; the paper's headline F1 (0.85 and 0.82) is the average over the background and UTC classes, so it is recorded as non-primary.
- P154: results are on NeonTree (first listed dataset); the split is `not_stated` because Table 4 does not say which split it uses.
- P029: Region A with 960 x 960 blocks (first listed subset) is primary.

Open points of B04 beyond the questions in section 2:

- P154 contributes only its object detection task (Table 4); its land cover, genus-level classification, grid segmentation and
  point cloud experiments are area-level or not scored per tree. PolyNet in and out are different models, so they are not
  transfer pairs. Whether Table 4 is on the test split is not stated.
- P023: the text says the four-model fusion with the most detected trees is Swin transformer, Faster R-CNN, YOLO and DETR, while
  Table 4 lists M1, M2, M3 and M5 (M3 is R-CNN). Table 4 is followed.
- P029 shares the Yosemite data with P005, P014 and P030 (key `yosemite_trees`); it prints an 11.8 cm GSD where P005 prints 0.12 m.
- P026 does not say which split Tables I and II use and defines none of its overall accuracy, average IoU and counting error;
  they are recorded as `other` (not primary).
- P005: six transfer pairs have no in-domain value with the same metric (rule V8), as P118 tr1 to tr4.

### B05

Value choices:

| ID | Target | Used | Not used | Rule | Reason |
|---|---|---|---|---|---|
| P048 | res5.value (MAPE of the canopy area) | 4.94 | 4.8 | V1a | The abstract and the validation panel of Fig. 8 give 4.94%; the text of Sec. 3.3.2 gives 4.8% |
| P052 | res5.value (U-Net VGG19, F1) | 0.85 | 0.8568 | V1a | Table 4, the comparison table, gives 0.85; the text of Sec. 5 gives 0.8568 (which would round to 0.86) |
| P052 | res6.value (U-Net VGG19, IoU) | 0.74 | 0.7495 | V1a | Table 4 gives 0.74; the text gives 0.7495 (which would round to 0.75) |
| P054 | gsd | 4 cm | 3 cm | V1c | The orthophoto and surface model that the networks use are given at 4 cm/pix in the data section (p.8); the raw images are about 3 cm/pix (p.7) |
| P061 | res11.value (YOLACT, region-level F1 at IoU 0.5) | 0.7743 | 0.7733 | V1a | Table 1, the main comparison table, gives 0.7743; Table 4 gives 0.7733 (and precision 0.7884 and recall 0.7587 against 0.7895 and 0.7597) |
| P066 | res7.value (StemCls mIoUc, UAV) | 0.74 | 0.77 | V1a | The Average row of Table 6 gives 0.74; the text of Sec. 3.1 gives 0.77 |
| P066 | res13.value (TreeBase F1, TLS, intervened) | 0.95 | 0.92 | V1a | The Average row of Table 7 gives 0.95; the text of Sec. 3.2 gives 0.92 |
| P066 | res10.value (CrownOff3D mIoUt, TLS, intervened) | 0.86 | 0.87 | V1a | The Average row of Table 7 and the abstract give 0.86; the text gives 0.87 |
| P066 | res19.value (TreeOff2D mIoUt, ALS, hands-off) | 0.59 | 0.57 | V1a | The Average row of Table 8 and the abstract give 0.59; the text gives 0.57, which is the F1 of that column |
| P067 | res16.value (ACRGS, cherry mIoU) | 0.933 | 0.938 | V1a | Table 4, the comparison table, and the abstract give 0.933; the text of Sec. 3.1 gives 0.938 |

Tagging, primary-setting and recording choices (written in the `notes` field of each paper):

- P039 and P054: both tables label a column mIoU while the paper's formula is the IoU of the target class. P039's printed mIoU is
  recorded as `iou` (primary, as the formula says; its values do not follow the printed formulas, see its notes). P054's is recorded
  as `miou` and is not primary: its values equal the mean of the canopy IoU implied by F1 and the background IoU implied by OA and F1
  (0.9345 against 0.9346 for the proposed 3-D row), so they include the background class. F1 (canopy class) is primary.
- P041: the primary results are challenge set 1 (the first listed test set); the validation set is the in-domain value of the 12 transfer pairs.
- P043: the primary setting is Jekyll (first dataset), ADE20k pre-training, RGB input; Table V (no pre-training) is a separate model item.
- P045: Table 6 is not primary. FM-SAM's printed precision (92.45) and recall (91.81) equal the means of the five per-class values of
  Table 2 (four species and the background), so the overall scores include a non-tree class (protocol 6.3). The BAMFORESTS
  transfer pairs use the Overall rows of Table 7 against the Table 6 scores; whether the models were retrained on BAMFORESTS is not
  stated, the paper calls it unseen data.
- P047: OA, mIoU and kappa are overall scores of two classes, so nothing is primary. The ablation baseline's kappa of -0.003 (Table 4)
  is not recorded because compile.py only accepts kappa from 0 to 1.
- P053: the ratio of segmented to true trees is metric `other`; the "Accuracy rate" rows equal the mean of the five plot ratios but the paper
  does not say so (aggregation not stated).
- P059: Table 1 is printed with decimal commas; mAP and mAP at IoU 0.5 are taken from the results text, which prints the same values with
  points. TreePseCo and DF (FT) are trained on the Valle d'Aosta data, so only the original DeepForest gives transfer pairs.
- P061: the primary settings (F1 and mIoU of the crown masks, R2 and RMSE of the crown area) all use the RGB input at IoU 0.5; F1 at IoU 0.75
  and the mean over 0.50 to 0.95 are recorded without `primary`; boundary F1 is metric `other`.
- P066: the UAV dataset is primary (first listed); the hands-off TreeisoNet is the fully automatic result, the intervened one is a
  separate model item. ForAINet's numbers are marked `copied_from` Xiang et al. (2024), which is not in the corpus.
- P067: the paper calls its mIoU instance-level but defines it over m classes, and does not say which trees the deep learning baselines
  are tested on, so level and split are `not_stated`; all Table 4 values carry the same aggregation tag.

Open points of B05 beyond the questions in section 2:

- P045: its Table 2 prints a Sophora japonica recall of 91.83 where its text and Table 3 give 89.83 (the F1 of 92.73 fits 89.83),
  and its text gives a Populus nigra precision of 92.29 against 92.92 in Table 2. Neither value is recorded (per-species scores are not
  recorded).
- P059: the resolution of its Valle d'Aosta orthophotos is missing from the printed text ("cm/pixel resolution").
- P050: the abstract prints "up to 0.38 at specific sites", but its Table 3 has site values up to 0.415; only the Average row is recorded.
- P066: the ForAINet numbers are copied from Xiang et al. (2024), which is not in the corpus (`copied_from`); text and tables disagree for
  four averages (cf items). The UAV data are the original FOR-instance (`for_instance_v1`), not FOR-instanceV2.
- P052 (20 networks) and P054 (18 networks over Tables II, V and VII) compare many networks; only the models required by the skill
  (proposed, transformer-based, best CNN, the baseline it builds on) plus the ones the text discusses are recorded.
- Papers without primary results: P045 (overall scores include the background), P047 (OA, mIoU and kappa over vegetation and
  background), P053 (the only score is the count ratio, metric `other`).

### B06

Value choices:

| ID | Target | Used | Not used | Rule | Reason |
|---|---|---|---|---|---|
| P077 | res22.value (CNN fusion biomass RMSE) | 0.089 | 0.132 | V1a | Table 6, the results table, gives 0.089 g/m2; the text of Sec. 3.2 gives 0.132, which is the MLP fusion value of Table 6 |
| P084 | res10.value (canopy area MAE, D3) | 0.15 | 0.16 | V2 | The Average row of Table 3 prints 0.16; the text and the abstract give 0.15 m2, which equals the mean of the 24 printed per-tree area errors (0.155) |
| P084 | res12.value (canopy volume MAE, D3) | 0.26 | 0.13 | V2 | The Average row of Table 3 prints 0.13; the text and the abstract give 0.26 m3, which equals the mean of the 24 printed per-tree volume errors (0.263) |
| P084 | res3.value (height RMSE, D1) | 0.13 | 0.23 | V1a | Sec. 3.3 gives 0.13 m; the abstract prints 0.23, which is the D2 MAE (the abstract exchanges the D1 RMSE and the D2 MAE) |
| P084 | res5.value (height MAE, D2) | 0.23 | 0.13 | V1a | The Avg. row of Table 2 and Sec. 3.3 give 0.23 (mean of the 24 printed errors 0.228); the abstract prints 0.13, which is the D1 RMSE |
| P084 | res8.value (height MAE, D3) | 0.37 | 0.16 | V1a | The Avg. row of Table 2 gives 0.37 (mean of the 24 printed errors 0.3675); the abstract and Sec. 3.3 print 0.16 |
| P093 | res20.value (PointNet++ backbone mIoU, Larch) | 65.51 | 65.15 | V1a | Table V gives 65.51; the text of Sec. 4.D gives 65.15 |
| P094 | res7.value (SegFormer ALS tree classification, average mIoU) | 0.89 | 0.92 | V1a | Table 3 and Sec. 4.1 give 0.89; the abstract gives 0.92, which is the SegFormer mIoU for TLS tree points (Sec. 5.1) |

Tagging, primary-setting and recording choices (written in the `notes` field of each paper):

- P068: Table III's tree-class IoU is recorded as `iou` of task `component_segmentation` (point level) and is primary; Table VII's overall
  F1 is primary and the headline. OA and mIoU of Table V and the per-area table (Table IV) are not recorded.
- P069: Table 3 (after the cylinder post-processing, test data of fold 1) is primary for the trunk and branch mIoU; Table 2 (four folds,
  before post-processing) is recorded as its own setting. Only tree height is a standard attribute and primary; the other traits are not.
- P073: the mIoU of the SegFormer classifiers (tree, stem and downed log points) includes the complement class, so only the
  individual-tree clustering mIoU is primary. The QSM volume comparisons and the biases are not recorded.
- P075: no overall score exists for the 61 acquisitions. The photogrammetry row (the paper's comparison with LiDAR) is primary; the
  short-sequence and photogrammetry rows of Table II are identical.
- P077: Table 7 (label checking) is not recorded; the abstract's precision (0.943) equals the Table 5 value (94.3).
- P079: pIoU is the part IoU averaged over trunk, branch and leaf and is recorded as `miou`; the class-wise F1 values are not primary.
- P080: the target biomass is computed from field DBH with the allometric equations of Chojnacky et al. (2014); the prediction of the proposed
  model is the mean of twelve predictions per tree.
- P081: "mean accuracy" is 2 n_matched / (n_reference + n_predicted), recorded as `f1`; completeness and correctness are not recorded. Models 1,
  3 and 5 are trained on one plot each and run on all six plots (the training plot included), so the split is `not_stated`. The FOR-instance
  automated results (Table 4) are given per plot only and are not primary; the matched and predicted counts compared with ForAINet
  (Table 5) are not recorded.
- P084: SAM is used zero-shot, so every data set is `test_only`. D1 (first listed) is primary for the height; the canopy area and volume exist
  only for D3. The D3 RMSE of the text (0.21 m) is below the Table 2 MAE of 0.37 m and is not recorded. Tree counts of SAM against manual
  counts are metric `other` (the paper gives no counting metric).
- P086: the tree-class IoU is primary; the mIoU averages tree and non-tree and carries the headline (85.28, abstract) without being
  primary. The spectral ablation of Table III is recorded for SPT without the row that repeats Table II. SPT's 212K parameters are a derived
  row of 0.212 M. Table IV (error rates above 2 m) and the training times of Table V are not recorded.
- P088: the first listed dataset (rubber plantations) is primary. Precision is not recorded (an IoU is given). RepSurf-U and Point2vec are
  named in Table 4 only and have no architecture category. The tree counts are sums of the printed training and test numbers (derived).
- P092: only the QSM attribute results are recorded. The unit of the RMSE is not printed with the values (cm for DBH and m for height, from
  the stated changes).
- P093: the six temperate datasets (Table I) are recorded under the key `stpctls` and the tropical set under `lewos`, both from the
  citations ([52] Xi et al., [53] Momo Takoudjou et al.); the paper does not name them. Table V (backbone replaced inside SPTNet) is an
  ablation and not primary. The split of the results is not stated (70 percent training, the rest validation and test).
- P094: the mIoU of the classifiers averages tree and non-tree classes and is never primary. The detection rate is recorded as `recall` and the
  instance mean IoU as `miou` (Table 4, Average row of the ten plots). TLS treeiso and the TLS NN-GPR row are comparison rows and not primary.
  RMSE% of Table 5 is not recorded.
- P095: the mIoU of RubberTree and ForestSemantic averages tree and other points, so no mIoU is primary. ShapeNetPart (objects other than
  trees) and the R2 column of Table 3 (not defined clearly) are not recorded.
- P096: precision and recall are not recorded (an F-score is given); the Overall row (pooled over the four plots) is primary, the per-plot
  values are not.
- No comparison table of B06 has more than ten models, so every row of the main tables is recorded (the larger-table rule of the skill was
  not needed; the 20 model items of P094 are three separate comparisons).

Open points of B06 beyond the questions in section 2:

- P084 prints three different sets of height errors (abstract, text, Table 2); five cf items record the choices (Table 2 and the text
  over the abstract). The text value of the D3 RMSE (0.21 m) cannot be right next to a MAE of 0.37 m and is not recorded.
- P081: the Table 2 values average all six plots, including the training plot of each model, so they are not clean test results
  (split `not_stated`); the ForAINet comparison and the prompt-mode results are not recorded.
- P095: its transfer pair compares the RubberTree test mIoU (88.78) with new point clouds of the same plantation city (86.15); the paper
  does not say what differs, so the shift type `site` is a weak label. The R2 column of its Table 3 is not recorded (the quantity is
  not defined clearly); the R2 of the field test is only printed in the abstract (1.00, 0.99, 0.89).
- The text layers of P096 (tables and prose overlaid), P092 and P073 (partly duplicated) are poor. The P096 table values are typed from
  the rendered page (the OCR check found them all); spot-check the quotes of those papers against the rendered pages.

### B07

Value choices:

| ID | Target | Used | Not used | Rule | Reason |
|---|---|---|---|---|---|
| P099 | res1.value (TreeLoc F1, LiDAR only) | 0.83 | 0.82 | V1a | Table 3 and Sec. 4.1 give 0.83; the abstract gives 0.82 (the bar of Fig. 4 is about 0.815) |
| P100 | res1.value (VLM-TMN DBH RMSE) | 7.1 cm | 7.08 cm | V1a | Table 1, the main comparison, gives 7.1 cm; Table 2 (ablation) and the abstract give 7.08 cm |
| P100 | res2.value (VLM-TMN height RMSE) | 2.3 m | 2.25 m | V1a | Table 1, the main comparison, gives 2.3 m; Table 2 gives 2.25 m |
| P102 | dataset2.n_trees (STPCTLS) | 691 | 690 | V1c | Table II gives 691 samples (the seven species sum to 691); the text of Sec. III-C gives 690 |
| P114 | res2.value (FGI-PointTransformer-DL-3D, HeliALS, macro-average accuracy) | 0.712 | 74.5 percent | V1a | Fig. 6a, Sec. 4.1 and Sec. 4.2 give 71.2 percent; the abstract's 74.5 percent is the bar label of FGI-PointTransformerWeighted-DL-3D (0.745), paired there with the 87.9 percent overall accuracy of the unweighted model |
| P124 | model7.total_params (YOLOv5) | 7.1 M | 7.02 M | V1b | Table 4 (the cost table of the main comparison) gives 7.1 M with 15.8 GFLOPs and 67.0 FPS; the ablation Table 5 gives 7.02 M |
| P124 | res1.value (Light-ViTeYOLO mAP@0.5) | 97.2 | 97.27 | V1a | Table 3, the main comparison, gives 97.2; the ablation Table 5 gives 97.27 |
| P126 | model6.inference_speed (Faster R-CNN) | 10.8 FPS | 10.4 FPS | V1a | Table 6 gives 10.8; the text of Sec. 5.1 says the proposed model's 10.4 FPS is the same as that of the original Faster-RCNN |
| P126 | res11.value (Faster R-CNN mAP50) | 88.7 | 88.9 | V1a | Table 6, the main comparison, gives 88.7 (the conclusion's gains of 7.2, 2.0 and 4.5 points follow from it); the ablation Table 5 gives 88.9 |
| P129 | res1.value (EVitNet IoU, Dataset A) | 0.713 | 0.7134 | V1a | Table 3, the main comparison, and the abstract give 0.713; the text of Sec. 3.3 gives 0.7134 |

Tagging, primary-setting and recording choices (written in the `notes` field of each paper):

- P098: only Table 5 (tree parameters from the completed clouds against the real, complete clouds) is recorded; height, DBH and the
  north-south crown width are primary, the east-west width is not (same attribute). The completion scores are not recorded.
- P099: the vegetation filter is not a tree task and is not recorded. LiDAR-only delineation and LiDAR + RGB classification columns of Table 3
  are primary; the abstract's 0.94 and 0.82 are rounded bars of Fig. 4 (Table 3 prints 0.93 and 0.83).
- P100: the values of the two cited methods are `copied_from` their papers (their own images) and are not primary; the split is not stated.
- P102: Table V (self-built data, first listed) is primary, STPCTLS (Table VI) is not; kappa columns are kappa times 100; Table II prints 998
  for the original training total but its species rows sum to 728 (not recorded).
- P103: the yellow-leaf period is primary (the abstract's first result is there), the green-leaf period is not; the 5:2:2 split was made
  on augmented sub-images.
- P104: no overall value in Tables 4 and 5, so the oil palm column (first listed) is primary; F is `f1`, the per-class accuracy is not recorded.
- P108: SZU-South is primary (the abstract's headline is there), SZU-North (listed first) is not; Tables V to VIII are images.
- P109: the scored unit of the F1 and kappa is not stated (level `not_stated`); the boundary F1 of SAM is metric `other`.
- P110: species-level OA and macro F1 of Table 9 are primary; genus and foliage-type results of the proposed network and the one-branch
  ablation are not.
- P114: HeliALS is primary, Optech Titan is not; the macro-average accuracy is `recall`; the input and loss variants of Point Transformer
  other than the default and the weighted loss, and SA-DetailView (not retrained), are not primary; Point2Vec has no architecture category
  (not described); IBL-BalancedRF-ML was trained on 660 of the 1065 training segments; the macro F1 printed for the best method of each
  group is recorded without primary; a segment may hold several trees.
- P115: AUC is metric `other`; the split is not stated.
- P117: the six input settings are ablations without a stated default; the best (the abstract's result) is primary and the headline; the
  split is not stated.
- P120: the weighted recall is the paper's overall accuracy (`oa`); Pr, Re and F are support-weighted; Table 4 (parameters and FLOPs) are
  literature values for ModelNet40.
- P123: Korean pine dataset (first listed) is primary; the 30 percent part of the 7:3 split is the evaluation set (`validation`); 12 of the 14
  rows of each table are recorded (YOLOv3 and YOLOv4 are in the notes); only mAP@50 is recorded.
- P124: Table 3 is primary, the backbone comparison of Table 6 is not; Table 5 (ablation) is not recorded except in cf items.
- P125 and P129: the evaluation sets are the validation set (P125) and the test set (P129) of Dataset A; P129's Dataset B gives two transfer
  pairs (EVitNet 0.713 to 0.321 and UNet 0.669 to 0.109) and the fine-tuned model (IoU 0.735).
- P126: mortality detection; the test split (20 percent, not augmented) is evaluated.
- P130: Hubei (first listed) is primary; all 18 models of both tables are recorded; Table II's FLOPs and parameters, which differ from
  Table I for most models, are not recorded; SegFormer is `pure_vit` as the table prints (ViT-base) and DAE-former has no category.

Open points of B07 beyond the questions in section 2:

- P114: the abstract pairs 87.9 percent overall accuracy with 74.5 percent macro-average accuracy, which belong to two different variants
  (cf item). Point2Vec is named but not described; set `pure_vit` if you know its architecture from the source.
- P099: the transfer pair (TreeLoc, summer and autumn 2023 to spring 2024) has no same-season test value, so it is kept out of the paired
  figures (rule V8), as P114's pair (SA-DetailView, trained on FOR-species20K; no in-domain value in the paper).
- P123 (pp. 4 to 6), P129, P120 (Table 3 and p.3) and P130 (tables) have poor text layers; the table values were typed from the rendered pages
  and found again by OCR or checked on crops (49 values that OCR missed or read without the decimal point: P098 15, P108 18, P114 8, P130 8).
  Spot-check the quotes of those papers against the renders if you want extra assurance.
- The row and column assignment of the image tables of P114, P117, P120, P123 and P129 was re-checked on rendered crops at the start of the
  B08 session (2026-10-13); everything matched (see `reports/B07.md`, Checks).

### B08

All 16 papers are extracted; none is proposed for exclusion (416 results, 159 primary; 13 new comparisons, 70 transfer pairs, 4 value
choices; details in `reports/B08.md`).

Value choices:

| ID | Target | Used | Not used | Rule | Reason |
|---|---|---|---|---|---|
| P138 | dataset1.n_trees | 116726 (sum of the printed class counts) | 276883 | V1c | Table I and Table III give 114,842 healthy and 1,884 unhealthy palms (the Table III splits add up to the same counts); the abstract's 276,883 annotations match neither the class counts nor their sum |
| P140 | model1.trainable_params | 84.7 M (whole model) | 81 M | V1b | The paper gives 84.7 M trainable parameters for the whole model (encoder and U-Net decoder); 81 M is the SegFormer-B5 encoder alone |
| P141 | model10.total_params (RT-DETR) | 20.75 M | 20.7 M | V1a | Table 8 is the main comparison table and prints 20.75 M; the ablation Table 5 prints 20.7 M for the same model (a difference of rounding) |
| P141 | model10.total_params (RT-DETR) | 20.75 M | 20.8 M | V1a | The text of Sec. 3.6 rounds the size to 20.8 (printed as MB) while Table 8 prints 20.75 M |

Tagging, primary-setting and recording choices (written in the `notes` field of each paper):

- P131: the RGB scheme is primary (the default input that every deep model, SAM included, has in common); HSV and RGBHSV rows, the random
  forest and the HSV row of SAM are not; the headline is the abstract's best result (U-Net, RGBHSV, F1 0.9586). The success rate of the
  561 reference points is `recall` at object level (not a held-out test); the split is not stated; Tables 6 and 9 are image-read.
- P133: no overall value, so F1 of every test strip is recorded and strip 2 (first listed) is primary; 14 transfer pairs (shift site), the
  seven of the one-class models without an in-domain value; Deep SVDD has no architecture category; the sensitivity analyses are not recorded.
- P134: the DSI versions are `cnn_transformer_hybrid` (a single spatial self-attention layer); only mAP@50 is recorded, all 14 rows are primary.
- P136: the proposed method is convolutional; the only transformer is the DETR baseline; the evaluation set is the validation set; ablations
  are not recorded.
- P137: task `detection`; level and split `not_stated`; the vitality index is not scored; Mask2Former has no stated backbone
  (`cnn_transformer_hybrid`); the abstract says U-Net is best while Table 1 ranks Mask2Former first.
- P138: mAP from the text (also Fig. 11), the healthy-class F1 is primary and the unhealthy-class F1 is not; the Overall row of Table IV and
  the copied benchmark values are not recorded; the split is not stated and the two models may be scored on different splits.
- P140: the mean of the four biomes is the only overall value (primary, headline); the biome rows (best resolution bin) are not primary; the
  scored unit is a deadwood pixel including partial canopy dieback; no baseline.
- P141: split of Table 8 not stated (test likely); Table 5 backbones and Table 9 hybrid training are not primary; Table 10 gives two transfer
  pairs (shift region); the recall of YOLOv3-tiny is printed as 715.
- P142: the mIoU of Table 1 (validation) averages the diseased and the negative class and is not primary although it is the abstract's
  headline; the tree-count precision and recall of Table 3 (test) are primary; single-head attention models are `cnn_transformer_hybrid`.
- P143: eight model items (four backbones, two pipelines); BigTreeForest is primary; level `instance_mask`; router accuracy (Table II) and the
  unlabelled Fig. 5 are not recorded; split not stated.
- P144: the overall F1 (Table II) is primary (it equals the mean of the five per-class F1 within rounding); the per-class F1 (the Dead class
  as `mortality_detection`) is not; YOLOv11 is `cnn`; split not stated; the values come from the text sentences, which match the tables.
- P145: box and mask AP50 and COCO mAP (Table 2) are primary at both levels; AP75 has no metric code; validation set; the in-situ validation
  (nine sites) is recorded as recall at two thresholds (not primary); the false detection shares have no stated denominator.
- P146: AP50 (AP_PWD) of Table 1 is primary, split not stated; the printed accuracy rate 73.1 is `recall` (463 of 633) on the 400-image test set
  (not primary); the ViT-only ablation row is recorded without primary; model sizes in MB are not recorded.
- P147: F1 of 11 models is primary (integer percentages); five rows are `copied_from` references [7] and [20]; the split is not described;
  `mopad` covers Sites 1 and 2.
- P151: dead tree F1 and IoU on the random split (mean of three runs, SD in the note) are primary; the All columns are not recorded; nine
  scenarios of generalisation are recorded as non-primary results and as 54 transfer pairs without an in-domain value; Fig. 3 has no values.
- P153: Table 5 (F1 and mAP@0.5 of seven models, test set) is primary; Table 3 backbones are not primary; Table 4 is not recorded.

Open points of B08 beyond the questions in section 2:

- The text layers of P131 (Forests pages, duplicated), P145 (pp. 3 to 9, partly duplicated) and P144 (tables interleaved) are poor: P131's
  Tables 6 and 9 were read from renders, P145's p.7 and P144's Tables II and III were compared with renders, and quotes of P145 are
  short fragments that exist verbatim on their pages. Spot-check them against the renders if you want extra assurance.
- No B08 value was flagged by the OCR check (7 image values, all found); every text quote was checked as verbatim (5,311 quotes).
- P151's 54 transfer pairs and P133's 7 have no in-domain value with the same metric (rule V8) and stay out of paired figures.

## 4. How this file is kept

After each batch the agent adds a section to part 3 (value choices, recording choices, open points) and any new proposed exclusion to
part 1, and folds new open questions into the checklist of section 0 and into part 2. B01 to B08 are all in; section 0 is the page to
decide from. The tracker (`Paper_Tracker.md`) lists the proposed exclusions automatically from `papers.csv`; the judgement calls exist
only here, in the batch reports and in `compiled/conflicts.csv`. When you apply a decision (see "How to apply my decisions" in section 0),
update the matching row of section 0 and the counts at its top.
