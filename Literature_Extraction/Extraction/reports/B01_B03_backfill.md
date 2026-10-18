# B01 to B03: back-fill of the protocol 6 tags and verification

Status: complete

## What changed

- The 56 extraction scripts were moved from a temporary folder into `scripts/B01` to `scripts/B03`. Re-running every
  script reproduced its paper's rows in the workbooks exactly before any change was made.
- Every result now carries `result_task`, `level`, `split`, `aggregation` (and `attribute` for attribute estimation),
  `primary` where protocol 6.3 selects it, and `copied_from` where a value is copied from another paper (P014, 30 baseline
  values from P030). Every dataset item has a `dataset_key`; every transfer pair a `transfer_level`. The tags are defined
  per paper in `scripts/legacy_tags.py` and re-applied by `kit/append.py`.
- Architecture categories: random forest, SVM, k-means, watershed and SfM+MVS are now `classical`; PoinTramba (P107) is
  `cnn_transformer_hybrid` (it has transformer encoders); the two baselines of P112 whose architecture the paper does not
  describe are `not_reported`.
- P148: the five training runs are not said to be averaged, so its results are `aggregation = not_stated` and the
  setting text no longer says "mean of five runs".
- P001 had two headline rows (detection mAP50 and F1 of the same model); the mAP50 row is kept.

## Value choices (cf items)

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

## Papers without primary results (tables only)

| ID | Why |
|---|---|
| P042 | Its overall scores average over non-tree scene classes (lake, ground, track, obstacle), and the SAM-YOLOv8 pipeline is scored on "high-quality images" while the YOLOv8x-seg baseline is scored on the unmodified set |
| P051 | Every in-domain value was read off a bar chart (approximate); the exact values are on the unseen Al Ain data (transfer) |
| P082 | Only rates of correct detection and segmentation (metric `other`) |

## Checks

- `corpus.py check`: 0 issues in B01, B02 and B03 (5518, 4962 and 4315 rows).
- `kit/verify.py` and `kit/linkcheck.py`: 0 issues in each batch.
- `kit/ocrcheck.py` (values read from images, now including parameter and FLOP counts): B02 all found. Confirmed on the
  rendered page:
  - P027 dataset1 n_images 1160: Fig. 9 (p.9), "1160 Images".
  - P107 res10 96.43, res18 96.46, res29 91.45, res30 92.46 (Table III, p.13) and res3 94.05, res4 94.55, res8 86.49
    (Table IV, p.14): OCR lost the decimal points; the digits match the rendered tables.
  - P139 Table I (p.7), read at 250 dpi: FIDC-YOLO 80.0 mAP, 26.1 M, 88.8 G; RT-DETR 72.1, 42.8, 107; YOLOv8m 78.4,
    25.8, 78.7; YOLOv10l 77.1, 25.7, 126.3; Mamba-YOLO 78.7, 57.6, 155.9; YOLO-PWD 74.3, 42.4, 61.6; L-FIDC-YOLO 76.0,
    2.9, 9.1. All match the workbook.
- `compile.py`: 0 errors, 0 open warnings. Notes: P035 repeats 28 results of P032 (same study, counted once); one
  equal value in P011 and P018 reviewed as a coincidence (`compile_review.csv`); P118's four transfer pairs have no
  in-domain value with the same metric; P056 model5 prints FLOPs without a unit.

## Open points for the author

- P003 (B04) still has no PDF. The paper is open access (PubMed Central PMC8783740, DOI 10.1155/2022/1549842) and is not
  retracted; the publisher's servers block automated downloads.
- P155's PDF is arXiv 2503.06312v2, which the authors retitled "DOFA-CLIP"; the manuscript cites it as GeoLangBind.
