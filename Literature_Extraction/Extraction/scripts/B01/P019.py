import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1, T2, T4 = "Table I", "Table II", "Table IV"
URL = "https://github.com/norlab-ulaval/PercepTreeV1"
ROWS = [
    R("task", "detection;component_segmentation", 1, "Abstract", "pseudo-labels (bounding box and segmentation mask predictions)", note="tree (trunk) boxes and masks in ground-level forest images"),
    R("output_unit", "tree_instance;tree_component", 6, T2, "APbb ... APseg", note="instance boxes and masks"),
    R("tree_setting", "natural_forest", 4, "Sec. III-B", "100 images collected from public, private, and commercial forests in Quebec, Canada", note="plus worldwide Flickr forest photos"),
    NR("tree_taxa", note="forest trees; species not given"),
    R("study_area", "Quebec, Canada (CanaTree100); worldwide (FlickrTree)", 4, T1, "CanaTree100 ... Québec ... FlickrTree100 ... World"),
    D("biome", "multiple", "Quebec forests and worldwide Flickr images; not stated in the paper"),
    R("platform", "ground_camera", 4, "Sec. III-B", "images collected by amateur photographers, encompassing different camera models", note="CanaTree100 frames sampled from ground-level forest videos"),
    R("sensors", "rgb", 4, T1, "CanaTree100 100 1280 × 720", note="RGB photographs and video frames"),
    R("annotation_type", "polygon_mask", 1, "Abstract", "pseudo-labels (bounding box and segmentation mask predictions)", note="human labels in CanaTree100 and FlickrTree100 (instance masks and boxes)"),
    R("dataset_name", "CanaTree100", 4, T1, "CanaTree100 100 1280 × 720 920 Québec", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. IV-A", "The burn-in model is trained on the train set of 60 images from CanaTree100, referred to as CT-Train", item="dataset1"),
    R("dataset_public", "yes", 1, "Abstract", f"Code and dataset links are available at {URL}", item="dataset1"),
    R("n_images", "100", 4, T1, "CanaTree100 100", item="dataset1", unit="images"),
    R("n_trees", "920", 4, T1, "CanaTree100 ... 920", item="dataset1", unit="trees"),
    R("dataset_name", "FlickrTree100", 4, T1, "FlickrTree100 100 ≤ 800 × 800 1463 World", item="dataset2"),
    R("dataset_role", "train_test", 4, "Sec. III-B", "test the generalization capabilities of our approach, for instance when the burn-in is performed strictly on CanaTree100 or strictly on FlickrTree100", item="dataset2"),
    R("dataset_public", "yes", 1, "Abstract", f"Code and dataset links are available at {URL}", item="dataset2"),
    R("n_images", "100", 4, T1, "FlickrTree100 100", item="dataset2", unit="images"),
    R("n_trees", "1463", 4, T1, "FlickrTree100 ... 1463", item="dataset2", unit="trees"),
    R("dataset_name", "FlickrTree110K (unlabelled)", 4, T1, "FlickrTree110K 110 937 ≤ 800 × 800 N/A World", item="dataset3"),
    R("dataset_role", "pretraining", 5, "Sec. III-F", "a ViTDet architecture initialized on COCO is trained on Du with either the set of pseudo-labels", item="dataset3",
      note="unlabelled images trained on with pseudo-labels"),
    R("dataset_public", "partial", 4, "Sec. III-B", "only links to the image URLs will be provided to the research community", item="dataset3"),
    R("n_images", "110937", 4, T1, "FlickrTree110K 110 937", item="dataset3", unit="images"),
]
CFG = [  # item, label in Table II, description, train-domain test set, in-domain bb50:95, seg50:95, other test set, cross bb50:95
    ("model1", "ylct", "ViTDet trained on CanaTree labels only (burn-in)", "CT-Test", "69.6", "64.2", "FT-Test", "46.5"),
    ("model2", "yuct burn", "ViTDet trained on FlickrTree110K pseudo-labels from the CanaTree burn-in model", "CT-Test", "70.4", "64.3", "FT-Test", "54.6"),
    ("model3", "yuct sam", "ViTDet trained on FlickrTree110K pseudo-labels refined by SAM (CanaTree burn-in)", "CT-Test", "69.6", "63.8", "FT-Test", "57.9"),
    ("model4", "ylft", "ViTDet trained on FlickrTree labels only (burn-in)", "FT-Test", "61.8", "56.5", "CT-Test", "65.4"),
    ("model5", "yuft burn", "ViTDet trained on FlickrTree110K pseudo-labels from the FlickrTree burn-in model", "FT-Test", "63.0", "57.1", "CT-Test", "66.0"),
    ("model6", "yuft sam", "ViTDet trained on FlickrTree110K pseudo-labels refined by SAM (FlickrTree burn-in)", "FT-Test", "64.6", "60.6", "CT-Test", "66.0"),
]
res, tr, de = Counter("res"), Counter("tr"), Counter("de")
for item, lab, desc, idt, bb, seg, odt, xbb in CFG:
    ROWS += model(item, desc, "proposed" if lab.endswith("sam") else "transformer_baseline", "pure_vit", "ViT-H (ViTDet)", "other", "full_fine_tuning", 5, "Sec. IV-A",
                  "We chose a ViT-H backbone architecture since it is the largest and highest scoring for object detection available from the checkpoint list",
                  pre_note="COCO checkpoint (Detectron2)", strat_page=4, strat_loc="Sec. III-C", strat_ev="The object detector is initialized from a COCO checkpoint",
                  cat_note="SAM generates pseudo-labels; the trained detector is ViTDet" if lab.endswith("sam") else "")
    ds_in = "dataset1" if idt == "CT-Test" else "dataset2"
    ROWS += result(res(), "ap50_95", item, ds_in, bb, 6, T2, f"{idt} {lab} APbb50:95 {bb}", setting=f"{idt}; detection (box)",
                   headline=(item == "model6"))
    ROWS += result(res(), "ap50_95", item, ds_in, seg, 6, T2, f"{idt} {lab} APseg50:95 {seg}", setting=f"{idt}; segmentation (mask)")
    t = tr()
    ROWS += [R("transfer_source", "CanaTree100 (Quebec forest video frames)" if idt == "CT-Test" else "FlickrTree100 (worldwide Flickr photos)", 4, "Sec. III-B",
               "when the burn-in is performed strictly on CanaTree100 or strictly on FlickrTree100", item=t),
             R("transfer_target", "FlickrTree100 test set" if odt == "FT-Test" else "CanaTree100 test set", 6, T2, f"{odt} {lab} APbb50:95 {xbb}", item=t),
             R("shift_type", "region;sensor", 4, "Sec. III-B", "different camera models, resolutions, geographical regions, and other parameters", item=t),
             R("transfer_model", item, 6, T2, lab, item=t),
             R("transfer_metric", "ap50_95", 6, T2, "APbb50:95", item=t, note="box"),
             R("value_in_domain", bb, 6, T2, f"{idt} {lab} APbb50:95 {bb}", item=t, unit="%"),
             R("value_out_of_domain", xbb, 6, T2, f"{odt} {lab} APbb50:95 {xbb}", item=t, unit="%")]
for size, burn, sam in (("100", "54.6", "57.9"), ("10", "54.2", "57.5"), ("5", "54.6", "57.5"), ("1", "53.3", "55.4")):
    for m, v, lab in (("model2", burn, "yuct burn"), ("model3", sam, "yuct sam")):
        i = de()
        ROWS += [R("label_model", m, 6, T4, f"{size} % {lab}", item=i),
                 R("label_amount", size, 6, T4, f"{size} % {lab} {v}", item=i, note="percent of FlickrTree110K used with pseudo-labels (automatic labels, not human)"),
                 R("label_unit", "percent_of_training_set", 7, "Sec. IV-D", "FlickrTree110K is split into subsets of varying proportions [10, 5, 1] %", item=i),
                 R("label_metric", "ap50_95", 6, T4, "APbb50:95", item=i, note="box; evaluated on FT-Test"),
                 R("label_value", v, 6, T4, f"{size} % {lab} APbb50:95 {v}", item=i, unit="%")]
ROWS += avail(code_url=URL, page=1, loc="Abstract", ev=f"Code and dataset links are available at {URL}", data="partial",
              note_d="labelled sets via links; FlickrTree110K as image URLs only (p.4)")
ROWS += [D("notes", "Ground-level (close-range) images. Table II also reports APbb50, APseg50 and recall; Table III compares SAM prompt types (box, mask and five points best: APbb50:95 57.9 on FT-Test). The CanaTree-trained burn-in model drops from 69.6 to 46.5 APbb50:95 on FT-Test; SAM pseudo-labels reduce the drop (57.9).", "Sec. IV and Tables II to IV (pp. 6 to 7)")]
