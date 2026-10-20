# Data-Extraction Protocol

**Status: in use.** The field list and sections 1 to 5 were approved before B01. Section 6 (value-selection rules) and the
result tags it relies on were added after B03 and back-filled into B01 to B03.

This protocol is also the basis of the data-extraction subsection of the manuscript's methodology.

## 1. Scope

- Unit of extraction: one report (paper), identified by its paper ID (`P001` to `P156`). `record_id` (`R####`) links it to the screening sheet; papers found by other methods have no record ID. IDs and decisions live in `../Corpus/papers.csv`.
- Unit of analysis: the study. Reports of the same study share a `study_group` in `papers.csv` (for example `teng_2025`) and are counted once in study-level statistics.
- Only values reported in the paper are recorded. Nothing is inferred, estimated from other papers, or filled in from memory.
- Version and date: values are extracted from the version that is cited, which must be dated 2025 or earlier (the year of the journal issue or proceedings, or the posting date of a preprint). The published version is preferred; a paper whose version of record is dated 2026 stays only when a 2025 preprint is the extracted and cited version (decision of 2026-10-15).

## 2. Files

| File | Content |
|---|---|
| `../Corpus/papers.csv` | One row per report assessed on the full text: IDs, title, venue, DOI, Zotero collection, decisions, batch, PDF path |
| `fields.csv` | The field dictionary: name, group, type, allowed values, unit, required or not |
| `datasets.csv` | Registry of dataset keys (rule V9): one row per dataset as the papers name it (public datasets shared, own data per paper) |
| `batches/B01.xlsx` to `B08.xlsx` | The extracted values, one Excel workbook per batch of about 20 papers |
| `scripts/B##/P###.py` | The script that wrote each paper's rows (re-run with `Scripts/extraction/kit/append.py`) |
| `../PDF_Text/` | Machine text of every included PDF with page breaks, used to locate values |
| `compiled/` | Analysis tables generated from the batch workbooks by `Scripts/extraction/compile.py`; never edited by hand |

Every value exists in exactly one batch workbook. The batch workbooks are the single source of truth for extracted data.

## 3. Batch workbook format

Each workbook has a `values` sheet with one row per value, and a `papers` sheet listing the batch's papers and PDFs:

| Column | Content |
|---|---|
| `paper_id` | `P###` |
| `item` | Empty for single-value fields; `dataset1`, `model2`, `res5`, `tr1`, `de3`, `cf1` to group repeatable fields (see `item_prefix` in `fields.csv`) |
| `field` | A field name from `fields.csv` |
| `value` | The value as printed, or one of the allowed categories |
| `unit` | Unit when the field has one (for example `%`, `M`, `cm`) |
| `page` | Position of the page in the PDF file where the value is visible (the first page of the file is 1; the number printed on the page can differ, for example in book chapters) |
| `location` | Table, figure or section (for example `Table 3`, `Fig. 7`, `Sec. 4.2`) |
| `evidence` | Short quote or table cell text that shows the value (25 words or fewer) |
| `status` | `reported`, `not_reported`, `approximate` (read off a figure) or `derived` (computed from reported values; the formula goes in `note`) |
| `note` | Anything needed to read the value correctly |
| `checked` | Left empty by the extractor; the author writes `ok`, or `fix: <correct value>` |

Required fields that a paper does not report get one row with `status = not_reported`, so missing values are explicit.

Values printed inside images (tables embedded as pictures, data labels on charts) are `reported`, with `(image)` added to `location`. Values estimated from the geometry of a plot are `approximate`: the bar or point is measured on the rendered page against the axis gridlines (and against any values the text gives for the same chart), rounded to the precision the paper uses elsewhere, and the measured value goes in `note`. Analyses can then include or exclude approximate values explicitly.

## 4. Rules

### 4.1 Results

- Record the results needed to compare models: every model in the paper's main results table on the main test set(s), for the metrics the review analyses (F1 and AP for detection; IoU, mIoU, mask AP and F1 for segmentation; OA, kappa and F1 for classification; R², RMSE and MAE for regression and counting). Precision and recall are recorded only when no F1 or AP is given. Mark the authors' main result with `headline = yes` (one result at most; none when the paper presents no single main result, as benchmarks often do).
- When the main table lists more than ten models, record: every transformer-based model; the proposed model; the CNN counterpart of a transformer backbone in the same framework (for example Mask R-CNN with ResNet-50-FPN next to Swin-T-FPN); the baseline the proposed model is built on; and, for each recorded metric, the best CNN (ties included). The remaining rows are referenced in `notes`. Tables with ten models or fewer are recorded in full.
- Record values exactly as printed. The `unit` column of every value says `%`, `fraction` (0 to 1) or the physical unit; conversion to a common scale happens only in the compiled tables.
- Link each result to its model and dataset items, and use `result_setting` for the split or variant in words (for example `test set; mask`).
- Every result also carries four tags that the figures filter on: `result_task`, `level`, `split` and `aggregation`, plus `attribute` when the task is `attribute_estimation`. Results in the paper's standard comparison setting get `primary = yes`. Section 6 says how to set them.

### 4.2 Comparisons (applied in the analysis, not during extraction)

- A transformer-versus-CNN comparison uses the best transformer and the strongest non-transformer deep model (`cnn` or `other`) evaluated by the same study, taken from its `primary` results with the same task, level, metric and dataset. `classical` methods are reported separately and never serve as the baseline.
- Differences are in percentage points for bounded metrics and in relative change for error metrics. Differences within 1 point (2% relative for error metrics) count as ties.
- Comparisons are reported by metric and task, with medians and interquartile ranges, and separately for author-proposed and independent benchmarks.

### 4.3 Transfer

- A transfer pair is the same trained model evaluated on its source test data and on unseen target data with the same metric.

### 4.4 Data efficiency and adaptation

- Labelled data is recorded as an amount with its unit (images, tiles, trees, percent of the training set).
- Parameter-efficient fine-tuning is compared with full fine-tuning only when both are reported in the same paper under the same setting.
- Parameter counts are in millions as reported (unit `M`); `trainable_params` is the number updated during training, or the percentage of all parameters (unit `%`) when the paper gives only that.

### 4.5 Architecture categories

| Category | Rule |
|---|---|
| `pure_vit` | Plain transformer encoder on patch tokens without a convolutional backbone (ViT, DeiT, DETR-style detectors with transformer backbones) |
| `hierarchical_vit` | Multi-scale transformer backbone (Swin, PVT, SegFormer MiT, Twins) |
| `cnn_transformer_hybrid` | Convolutional or other non-transformer backbone (point MLP, state-space) with transformer blocks or self-attention layers added (including DETR and RT-DETR with CNN backbones, YOLO with transformer modules, and ensembles that fuse the outputs of transformer and convolutional models) |
| `foundation_model` | Large pre-trained vision model used for adaptation or zero-shot use (SAM, DINOv2, Prithvi, geospatial foundation models) |
| `vision_language_model` | Model that takes language as input or output (CLIP, Grounding DINO, LLM-based systems) |
| `cnn` | Convolutional model without transformer components, including graph and sparse 3D convolutions (EdgeConv, KPConv, MinkowskiNet) |
| `other` | Non-transformer deep network that is not convolutional: MLP-based point networks (PointNet, PointNet++, PointMLP, PointNeXt), state-space (Mamba) models, recurrent networks |
| `classical` | No deep learning: random forest, SVM, gradient boosting, k-means, watershed, region growing, photogrammetric reconstruction |

Generic attention modules alone (SE, CBAM, coordinate attention) do not make a model transformer-based. Multi-head self-attention blocks do. When a paper does not describe a baseline's architecture, `architecture_category` is `not_reported` with a note.

## 5. Workflow per batch

1. The PDFs of the batch are in `../PDF_Library/` (`Scripts/extraction/corpus.py file <folder>` files new ones), and their text is cached (`corpus.py text`). `corpus.py new B##` creates the batch workbook.
2. Each paper's eligibility is confirmed on the full text. A paper that fails the criteria is set to `Pending` in `papers.csv` with the proposed reason and is not extracted; the author decides.
3. Each paper is extracted by a script, `scripts/B##/P###.py`, built with the helpers in `Scripts/extraction/kit/h.py`; `kit/append.py B## P###` writes its rows into the workbook and can be re-run at any time.
4. Checks, after each paper and again for the whole batch:
   - `corpus.py check B##`: field names, allowed values, required fields per item, cross-references, sources, and the rules of section 6 that a script can test;
   - `kit/verify.py B##`: every number taken from the text appears on its cited page;
   - `kit/linkcheck.py B##`: every number appears in its own evidence quote;
   - `kit/ocrcheck.py B##`: every number read from an image table is found again by OCR on that page;
   - `kit/pagecheck.py B##`: a quote that the text layer holds on several pages, or off its cited page, is visible on the cited page (OCR of the rendered pages);
   - `Scripts/extraction/compile.py`: rebuilds `compiled/` from all batches and fails on any figure-readiness problem.
5. The author cross-checks a random sample of values against the papers (section 7). Fixes go into the paper's script and are re-applied with `append.py`.
6. The final corpus is fixed once all batches are cross-validated: a paper that fails the criteria on closer reading is set to `Exclude` in `papers.csv` with its reason, and PRISMA is recomputed (`Scripts/search/screening.py prisma`).

## 6. Value-selection rules

Every number in a figure or in the text traces to one recorded value. Whenever a paper offers more than one candidate number, the choice is made once, at extraction, by one of the rules below, and recorded where a script can read it. Figure scripts never choose, guess, or convert by magnitude.

### 6.1 Rules

| Rule | Situation | Decision | Recorded as |
|---|---|---|---|
| V1a | The paper prints two different values for the same result | Use the main comparison table (the table on which the paper's comparison with other models rests) over ablation tables, text, abstract and figures | Chosen value in its normal row; the other value in a `cf` item |
| V1b | Two different values for a model's size or cost | Use the whole-model value that goes with the paper's other cost figures for that model (for example the parameter count printed with its FLOPs); a backbone-only count is the alternative | as V1a |
| V1c | Two different dataset or study descriptors (counts, GSD, area) | Use the dataset table or data section over passing mentions; within the data section, the more specific statement (for example the per-split table) | as V1a |
| V2 | A printed aggregate (mean of folds, total of splits) disagrees with the printed parts it summarises | Record the parts; record the aggregate only as a `cf` item whose `cf_target` names it. A figure that needs an aggregate gets it from `compile.py`, computed from the parts | `cf` item |
| V3 | A model has several results | `primary = yes` marks one standard setting per paper and level (6.3). Every model compared in that setting gets `primary` on each recorded metric; at most one primary result per model, task, attribute, level and metric | `primary` |
| V4 | Metric identity | Record the metric the paper computes, by its definition, not its column label (a column labelled AP that averages per-class precision is `precision`). Never derive a performance value from others (no F1 from precision and recall). `level` follows the paper's definition (6.2) | `metric`, `level` |
| V5 | Scale | Values as printed; `compile.py` converts by the `unit` column only (fraction × 100 to %, cm to m), never by magnitude | `unit` |
| V6 | Values read off a chart (`approximate`) | Never primary; figures leave them out unless a figure opts in and marks them | `status` |
| V7 | Derived values | Only unit conversions of non-performance quantities (parameter counts to millions), sums of printed counts, and coded categories (biome from the study area). Performance values are never derived | `status = derived`, formula in `note` |
| V8 | Transfer pairs | Same model, metric, level and aggregation on both sides. If the in-domain value is not given with the same metric, `value_in_domain` is `not_reported` and the pair stays out of paired figures. All pairs of a paper (folds, sites) are recorded; per-paper summaries are computed by `compile.py` | `tr` items |
| V9 | Datasets, copied numbers | Every dataset item has a `dataset_key` from `datasets.csv`. A key records the dataset as the paper itself names or cites it: a public dataset named by several papers has one key; a paper's own data get a key of their own and are never merged with another paper's (each paper is an independent data point, and no relation between papers is inferred or noted). A result the paper says it takes from another work instead of re-running gets `copied_from`, and `compile.py` keeps it out of cross-paper statistics so no number counts twice | `dataset_key`, `copied_from` |
| V10 | Headline and primary differ | `headline` is what the authors present as their main result (for quoting the paper); `primary` is the standard setting (for figures). Both are kept | both tags |

### 6.2 Levels

`level` (results) and `transfer_level` (transfer pairs) say what one scored unit is:

| Level | One scored unit | Typical metrics |
|---|---|---|
| `object` | A detected tree (box or point) matched to a reference tree | AP50, AP50:95, F1, precision, recall |
| `instance_mask` | A predicted crown or trunk mask matched to a reference instance | mask AP, instance F1, mean IoU over trees |
| `pixel` | A pixel of a semantic map | IoU, mIoU, pixel F1 or Dice, pixel OA |
| `instance_3d` | A segmented tree in a point cloud matched to a reference tree | F1, AP, precision, recall, coverage |
| `point` | A point of a 3D semantic map | mIoU, OA |
| `tree` | One prediction per reference tree (species, health, an attribute) | OA, F1, kappa, R², RMSE, MAE |
| `image` | One value per image or tile (counts per image, image labels) | MAE, RMSE, R², OA |
| `plot` | One value per plot or area | MAE, RMSE, R² |
| `not_stated` | The paper does not make the scored unit clear | |

A metric tag says no more than the paper states (rule added on 2026-10-19; `kit/metriccheck.py` checks it):
- `ap50_95` only when the paper states the IoU range 0.5 to 0.95, in any notation, or that it uses the COCO evaluation. A
  mean average precision whose thresholds the paper does not state is `map`, under the name the paper prints (mAP or AP).
- `ap50` only when the paper states an IoU threshold of 0.5.
- A metric the paper names differently but defines identically (detection rate as recall, user's accuracy as precision,
  per-class average IoU as mIoU) takes the standard tag, and the note gives the printed name.

### 6.3 The primary setting (rule V3)

Choose, in this order, and stop at the first rule that decides:

1. The dataset of the paper's main comparison table. If that table covers several datasets: the dataset of the headline result, otherwise the first listed. A paper with results on several levels or tasks (for example detection boxes and crown masks, or detection and health classification) has one primary setting per level and task.
2. The test split before validation, before cross-validation.
3. The mean over repeated runs before a single run. A best-of-runs value is never primary when a mean exists.
4. The configuration the main table compares: no ablation variants, the default input and resolution, every compared model.
5. When the paper reports results only per subset (plot, species, season, class) with no overall value: the subset of the headline result, otherwise the first subset listed.
6. Only standard metric codes are primary; `other` is not. `approximate` values are never primary (V6). A score averaged over classes that are not trees (ground, water, buildings, tracks) is never primary.

When no result of a paper qualifies (for example all exact values are of the `other` metric, or every in-domain value was read off a chart), the paper has no primary results and appears in tables, not in the numeric figures.

`result_task` is the task the score measures: finding diseased, infected or declining trees is `health_classification`, finding dead trees is `mortality_detection`, and a score averaged over species classes (detection or segmentation by species) is `species_classification`; `detection` and `crown_segmentation` are for trees of any kind and condition.

### 6.4 How the figures use this

`Scripts/extraction/compile.py` turns all batch workbooks into `compiled/*.csv` and stops with a list of problems if any check fails. Figures read only those tables:

- performance figures use `primary` results, grouped by `result_task`, `level` and `metric`, on one scale (V5);
- transformer-versus-CNN differences come from `compiled/comparisons.csv` (section 4.2), built from primary results only;
- transfer figures use `compiled/transfer.csv`, with both values present (V8);
- `approximate` values, `copied_from` results and `not_stated` splits are left out unless a figure says otherwise;
- every choice made under V1 or V2 is listed in `compiled/conflicts.csv`, so each plotted number traces to its page and to the rule that picked it.

A study is a paper, or the reports that share a `study_group` in `papers.csv`. A result that two reports of one study both print is marked `duplicate_of` and counted once, and comparisons are made per study.

For figures with one point per study, `canonical_metric` marks one metric per study, task, attribute, level and dataset: the first one the study reports in this order.

| Level | Metric order |
|---|---|
| `object` | AP50, F1, AP50:95, mAP, precision, recall |
| `instance_mask` | AP50, F1, IoU, mIoU, AP50:95, mAP, precision, recall |
| `instance_3d` | F1, AP50, AP50:95, mAP, precision, recall |
| `pixel` | IoU, mIoU, F1, OA, precision, recall |
| `point` | mIoU, IoU, OA, F1 |
| `tree` | OA, F1, kappa, mIoU, precision, recall, RMSE, MAE, R², MAPE |
| `image` | MAE, RMSE, R², OA, F1 |
| `plot` | RMSE, MAE, R² |

`compile.py` also warns when two studies print the same value for the same model, dataset and metric (a copied number without `copied_from`). A warning that has been checked and is fine goes into `compile_review.csv` with the reason, and is then reported as reviewed.

## 7. Author cross-check

After the extraction, the first author checks a random sample of recorded values against the PDFs, independently of the scripts
and of the automatic checks. `Scripts/extraction/crosscheck.py sample` draws, with a fixed seed, one result from every included paper
(a primary result where the paper has one) and 61 further numbers by kind (results, transfer values, model sizes and costs, dataset
sizes, GSD and point density); only values recorded as printed are drawn (values read from images included; derived, approximate and
copied values excluded). The sample sheet (`crosscheck/crosscheck_sample.xlsx`) gives for each value the page, the place, a plain
description and the quote; the author marks it OK, Wrong or Not found. Every value must be correct. A value marked Wrong or Not found is
corrected in its script; the whole paper is then checked again, the same kind of value is checked across its batch, and a new sample is
drawn for that batch. `crosscheck.py summary` records the outcome in `crosscheck/crosscheck_result.json`.

