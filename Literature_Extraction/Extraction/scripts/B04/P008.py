import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T3, T6, T7 = "P008", "Table 3", "Table 6", "Table 7"
# SwinTU-net (counting and detection of olive trees in satellite images): the paper's own totals, split not described. Mask R-CNN (crown segmentation in UAV images): Table 3, test subgroups.
TS = Tags(task="counting", level="not_stated", split="not_stated", agg="single")
TSD = TS(task="detection", level="object")
TM = Tags(task="crown_segmentation", level="instance_mask", split="test", agg="single")
GAP = "SwinTU-net is a U-net-like network that includes encoding, decoding, and skipping links"

ROWS = [
    R("task", "counting;detection;crown_segmentation", 1, "Abstract", "present an actual deep learning method (SwinTU-net) for identifying and counting of olive trees",
      note="Mask R-CNN also segments olive crowns and shadows in UAV images to estimate biovolume (not a transformer)"),
    R("output_unit", "tree_count;crown_mask", 1, "Abstract", "for identifying and counting of olive trees ... performed well in segmenting tree crowns", note="SwinTU-net identifies and counts trees; Mask R-CNN outputs crown and shadow masks"),
    R("tree_setting", "orchard", 4, "Sec. 3.1", "testing site is located within a 50-hectare olive orchard with 11,000 trees established in 2006"),
    R("tree_taxa", "olive", 4, "Sec. 3.1", "testing site is located within a 50-hectare olive orchard with 11,000 trees established in 2006"),
    R("study_area", "Jouf (Al-Jouf), Saudi Arabia", 4, "Sec. 3.1", "research area is positioned in Jouf, Saudi Arabia"),
    R("biome", "mediterranean", 4, "Sec. 3.1", "climate is the Mediterranean, with harsh summer temperatures and mild-wet winters"),
    R("platform", "multiple", 1, "Abstract", "with photos from unmanned aerial vehicles (UAVs) ... using satellite imagery, we present an actual deep learning method (SwinTU-net)",
      note="SwinTU-net: 250 RGB satellite images from Satellites Pro; Mask R-CNN: UAV RGB and multispectral images"),
    R("sensors", "rgb;multispectral", 6, "Sec. 3.6", "A sequoia hyperspectral sensor mounted on a Parrot DiscoPro AG UAV (Al-Jouf, Saudi Arabia) that acquired four spectral features (near-infrared (NIR), red, green, and red-edge)", note="RGB (DJI Phantom 4 camera) and a Sequoia multispectral sensor on a Parrot DiscoPro AG UAV (NDVI and GNDVI)"),
    R("gsd", "3", 6, "Sec. 3.6", "RGB image had a spatial precision of 3 cm/pixels", unit="cm",
      note="UAV RGB 3 cm/pixel, resampled to 13 cm/pixel; multispectral 13 cm/pixel; the GSD of the satellite images is not stated"),
    R("annotation_type", "point;polygon_mask", 2, "Sec. 1", "olive trees were tagged with a center fact to decrease the workload and speed up the annotating process",
      note="satellite images: tree centroids (p.2, p.8); UAV images: polygons around every crown and shadow (p.6)"),
    # SwinTU-net data
    R("dataset_name", "250 RGB satellite images of olive trees, Al-Jouf (Satellites Pro, 512 × 512)", 8, "Sec. 4.1", "collection contains 250 photos collected around Al-Jouf, KSA region, utilizing the Satellites Pro", item="dataset1"),
    R("dataset_key", "own_p008_jouf_satellite", 8, "Sec. 4.1", "collection contains 250 photos collected around Al-Jouf, KSA region, utilizing the Satellites Pro", item="dataset1"),
    R("dataset_role", "train_test", 8, "Sec. 3.10", "suggested method then is developed and validated using an NVIDIAGeForceRTX2060, GPU", item="dataset1",
      note="how the 250 images are split into training and evaluation data is not described; Table 6 evaluates all 250 images"),
    R("dataset_public", "on_request", 12, "Data Availability", "available from the corresponding author upon request", item="dataset1"),
    R("n_images", "250", 11, T6, "Number of images 250", item="dataset1", unit="images", note="resolution 512 × 512 (p.8)"),
    R("n_trees", "73285", 11, T6, "Number of trees 73285", item="dataset1", unit="trees", note="Table 6 also gives 72596 identified trees"),
    # Mask R-CNN data (no image or tree count is recorded: Sec. 3.7 and Table 1 disagree, see the notes)
    R("dataset_name", "OTCS-dataset: UAV olive crown and shadow patches (RGB 3 cm, RGB 13 cm, NDVI 13 cm, GNDVI 13 cm)", 3, "Sec. 1", "OTCS-dataset is a novel documented multispectral of orthoimages database for the olive tree crown identification", item="dataset2"),
    R("dataset_key", "own_p008_jouf_otcs", 3, "Sec. 1", "OTCS-dataset is a novel documented multispectral of orthoimages database for the olive tree crown identification", item="dataset2"),
    R("dataset_role", "train_test", 6, "Sec. 3.7", "with 120 images (80% of a database) utilized for testing its system and 30 photos (20% of the dataset) utilized for validating", item="dataset2",
      note="the text swaps training and testing; Table 3 reports the test subgroups"),
    R("dataset_public", "on_request", 12, "Data Availability", "available from the corresponding author upon request", item="dataset2"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated")
ROWS += model("model1", "SwinTU-net", "proposed", "hierarchical_vit", "Swin Transformer blocks in a U-net-like encoder-decoder with skip connections", "imagenet", None, 1, "Abstract",
              "SwinTU-net is a U-net-like network that includes encoding, decoding, and skipping links ... is the Swin Transformer blocks",
              pre_page=8, pre_loc="Sec. 3.10", pre_ev="scores pretrained model on the ImageNet-1K is utilized to set the set of parameters", pre_note="garbled sentence; weights from an ImageNet-1K pretrained model",
              strat_note="training scheme (SGD, momentum 0.9, weight decay 1e-4) is given but not whether the model is fine-tuned or trained from scratch", cat_note="Swin-based encoder and decoder")
ROWS += model("model2", "Mask regional-CNN (Mask R-CNN)", "cnn_baseline", "cnn", "ResNet50", "other", None, 7, "Sec. 3.8",
              "mask regional-CNN system (regions using convolutional neural networks) ... ResNet50 CNN was used in this research",
              pre_page=7, pre_loc="Sec. 3.8", pre_ev="Transfer learning involves first implementing the model’s parameters using pretrained weights on the COCO-database",
              pre_note="COCO weights", strat_note="transfer learning from COCO weights; whether all layers are retrained is unclear (garbled sentence)",
              cat_note="CNN segmentation model for crown and shadow masks; not compared with SwinTU-net")

res = Counter("res")
# SwinTU-net, Table 7 (the comparison table; the proposed technique is the last column). Four of the five columns are other methods on other databases and are not recorded.
ta, ev_ta = row(PID, 11, r"^\s*TA\s+", 0)
cfr, ev_cfr = row(PID, 11, r"^\s*CFR \(percent\)\s+", 0)
ofr, ev_ofr = row(PID, 11, r"^\s*OFR \(percent\)\s+", 0)
ef7, ev_ef7 = row(PID, 11, r"^\s*EF \(percent\)\s+", 1)
assert (len(ta), len(cfr), len(ofr), len(ef7)) == (4, 5, 5, 5), (ta, cfr, ofr, ef7)
SW = "all 250 satellite images; split not described"
ROWS += result("res1", "other", "model1", "dataset1", ta[-1], 11, T7, ev_ta, unit="%", setting=SW, note="TA, total accuracy: estimated olive trees divided by actual olive trees times 100 (Eq. 10)", **TS)
ROWS += result("res2", "other", "model1", "dataset1", ef7[-1], 11, T7, ev_ef7, unit="%", setting=SW, headline=True,
               note="EF, estimation failure: estimated minus actual olive trees divided by actual olive trees times 100 (Eq. 13)", **TS)
ROWS += result("res3", "other", "model1", "dataset1", cfr[-1], 11, T7, ev_cfr, unit="%", setting=SW, note="CFR, commission failure rate: false olive trees divided by actual olive trees times 100 (Eq. 12)", **TSD)
ROWS += result("res4", "other", "model1", "dataset1", ofr[-1], 11, T7, ev_ofr, unit="%", setting=SW, note="OFR, omission failure rate: omitted olive trees divided by actual olive trees times 100 (Eq. 11)", **TSD)
res.n = 4
# Mask R-CNN crown segmentation, Table 3: F1 (precision and recall are not recorded when F1 is given). Plan (a) with RGB at 3 cm/pixel is the first listed, plain configuration: primary.
LAB = {"RGB-3": r"^\s*Red green blue−3\s+", "RGB-13": r"^\s*Red green blue−13\s+", "NDVI-13": r"^\s*Normalization di\S+erential vegetative indices−13\s+",
       "GNDVI-13": r"^\s*Greenish normalization di\S+erential vegetative indices−13\s+"}
PLANS = [("a", "trained on every subgroup without additional data", [("RGB-3", 0), ("RGB-13", 0), ("NDVI-13", 0), ("GNDVI-13", 0)]),
         ("b", "trained on every subgroup with additional data", [("RGB-3", 1), ("RGB-13", 1), ("NDVI-13", 1), ("GNDVI-13", 1)]),
         ("c", "trained on the merged 13 cm/pixel subgroups with training data", [("RGB-3", 2), ("NDVI-13", 2), ("GNDVI-13", 2)])]
for plan, desc, subs in PLANS:
    for sub, n in subs:
        nums, ev = row(PID, 9, LAB[sub], n)
        assert len(nums) == 6, (plan, sub, nums)  # TP, FP, FN, Accuracy, F1, Recall
        note = "F1 measure column; IoU above 50 percent counts as a true positive"
        if plan == "c" and sub == "RGB-3":
            note += "; the row is printed as Red green blue-3 although plan (c) merges the 13 cm/pixel subgroups"
        ROWS += result(res(), "f1", "model2", "dataset2", nums[4], 9, T3, ev, unit="fraction", setting=f"Table 3 ({plan}): {desc}; test subgroup {sub}", note=note,
                       **TM, primary=(plan == "a" and sub == "RGB-3"))
ROWS += [R("limitations", "Olive trees that stand close to other trees are difficult to distinguish", 12, "Sec. 5", "there are certain disadvantages, including the difficulties in distinguishing olive trees that are near other trees")]
ROWS += avail(data="on_request", data_page=12, data_loc="Data Availability", data_ev="available from the corresponding author upon request")
ROWS += [D("notes", "SwinTU-net is evaluated only through total counts on the 250 satellite images (Tables 6 and 7: TA, EF, CFR and OFR, all metric other). Table 7 also lists four non-deep-learning methods on other databases (QuickBird, SIGPAC viewer), not recorded. Mask R-CNN (CNN, UAV images) is scored by F1 for olive crowns in Table 3 (shadows in Table 4 are not trees and are not recorded). No image or tree count is recorded for the OTCS-dataset: Sec. 3.7 gives 150 patches with 2400 trees per subgroup (120 and 30 images in the 80/20 split) while Table 1 gives 150 training and 40 test images with 145 overall and 650 segments per subgroup. Biovolume accuracies of the six sample trees (Table 5 and p.9) are not recorded.",
           "Sec. 3.7, 4.3 to 4.5 and Tables 1 to 7 (pp. 6 to 11)")]
# Value choices (protocol 6.1): Table 7 (comparison table) against the text and Table 6
ROWS += conflict("cf1", "res2.value", "0.94", 9, "Sec. 4.4", "total prediction error for testing is 0.94 percent after implementing the proposed model", "V1a",
                 "Tables 6 and 7 (the comparison table), Sec. 4.5 and the conclusion give 0.95; the introduction, Sec. 4.4 and p.11 give 0.94", unit="%")
ROWS += conflict("cf2", "res3.value", "0.97", 9, "Sec. 4.4", "around 0.97 percent of nonolive data is misinterpreted as an olive", "V1a",
                 "Table 7, the comparison table, gives 0.98 for the commission failure rate", unit="%")
ROWS += conflict("cf3", "res4.value", "1.2", 9, "Sec. 4.4", "1.2 percent of the olive data were misdiagnosed as nonolive", "V1a",
                 "Table 7, the comparison table, gives 1.3 for the omission failure rate", unit="%")
ROWS += conflict("cf4", "res3.value", "708", 11, T6, "CFG (percent) 708", "V1a",
                 "Table 6 prints 708 under a percent label (CFG), which cannot be a percentage; Table 7, the comparison table, gives 0.98 percent for CFR")
ROWS += conflict("cf5", "res4.value", "913", 11, T6, "OFG (percent) 913", "V1a",
                 "Table 6 prints 913 under a percent label (OFG), which cannot be a percentage; Table 7, the comparison table, gives 1.3 percent for OFR")
