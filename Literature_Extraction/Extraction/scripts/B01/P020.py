import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

F6, PAGE = "Fig. 6 (bar labels)", "https://forestvrw.github.io/M2fNet/"
ROWS = [
    R("task", "detection;component_segmentation", 1, "Abstract", "a multimodal transformer-based algorithm for tree detection and instance segmentation",
      note="tree trunk instances in ground-level views (p.2: 'segment the regions of interest of the tree trunks')"),
    R("output_unit", "tree_instance;tree_component", 3, "Sec. 3.2", "we predict the tree masks ... and the smooth L1 loss for box regression"),
    R("tree_setting", "synthetic", 1, "Abstract", "photorealistic virtual forest dataset", note="evaluated on real images from Canadian forests"),
    NR("tree_taxa", note="tree species drawn at random from a model library"),
    R("study_area", "virtual forest scenes (Unreal Engine); real test images from Canadian forests [7]", 4, F6, "Segmentation and detection evaluation on 100 real images collected in Canadian forests [7]"),
    NR("biome", note="virtual scenes; the real test images come from Canadian forests"),
    R("platform", "synthetic", 2, "Sec. 3.1", "the multi-sensor camera was moved along these trajectories", note="rendered ground-level camera views; real test images are ground-level photographs"),
    R("sensors", "rgb", 3, "Sec. 3.2", "two independent encoders ... to extract RGB and depth features", note="RGB plus rendered depth maps"),
    R("annotation_type", "polygon_mask", 2, "Sec. 3.1", "OpenCV was employed to automatically detect object contours and bounding boxes", note="masks rendered automatically by a semantic camera"),
    R("dataset_name", "M2fNet virtual forest dataset (Unreal Engine)", 2, "Sec. 3", "train the model on the newly simulated virtual forest dataset", item="dataset1"),
    R("dataset_role", "pretraining", 4, F6, "The algorithms are greatly improved by incorporating the dataset we created in pre-training", item="dataset1"),
    R("dataset_public", "yes", 2, "Sec. 1", f"Our project page and full dataset will be available at {PAGE}", item="dataset1", note="stated as future availability"),
    R("dataset_url", PAGE, 2, "Sec. 1", f"full dataset will be available at {PAGE}", item="dataset1"),
    NR("n_images", item="dataset1", note="size of the virtual dataset not given"),
    R("dataset_name", "CanaTree100 (real images, Canadian forests) [7]", 4, F6, "100 real images collected in Canadian forests [7]", item="dataset2"),
    R("dataset_role", "test_only", 4, F6, "Segmentation and detection evaluation on 100 real images collected in Canadian forests [7]", item="dataset2"),
    NR("dataset_public", item="dataset2", note="third-party dataset [7]; availability not stated here"),
    R("n_images", "100", 4, F6, "100 real images collected in Canadian forests", item="dataset2", unit="images"),
]
ROWS += model("model1", "M2fNet (RGB and depth)", "proposed", "hierarchical_vit", "two Swin encoders (RGB and depth) with a Mask2Former-style transformer decoder",
              "domain_specific", None, 3, "Sec. 3.2", "two independent encoders, which are hierarchical vision transformers with shifted windows (Swin), to extract RGB and depth features",
              pre_page=4, pre_loc=F6, pre_ev="incorporating the dataset we created in pre-training", pre_note="virtual forest dataset", strat_note="fine-tuning on real data not described")
ROWS += model("model2", "M2fNet (RGB input only)", "proposed", "hierarchical_vit", "Swin encoder with a Mask2Former-style transformer decoder", "domain_specific", None,
              4, F6, "Ours RGB input 87.2 89.7", pre_page=4, pre_loc=F6, pre_ev="incorporating the dataset we created in pre-training", pre_note="virtual forest dataset",
              strat_note="fine-tuning on real data not described")
ROWS += model("model3", "Mask R-CNN based [2]", "cnn_baseline", "cnn", "not stated", None, None, 4, F6, "Mask-RCNN based [2] 83.5 88.2",
              pre_status="not_reported", pre_note="not stated", strat_note="not stated")
res = Counter("res")
for m, lab, mask, box in (("model1", "Ours Multi-modality", "88.9", "91.0"), ("model2", "Ours RGB input", "87.2", "89.7"), ("model3", "Mask-RCNN based [2]", "83.5", "88.2")):
    ROWS += result(res(), "ap50", m, "dataset2", box, 4, F6, f"{lab}: AP50 for bb {box}", setting="100 real CanaTree images; detection (box)", headline=(m == "model1"))
    ROWS += result(res(), "ap50", m, "dataset2", mask, 4, F6, f"{lab}: AP50 for mask {mask}", setting="100 real CanaTree images; segmentation (mask)")
ROWS += avail(data="yes", note_d=f"dataset promised at {PAGE} (p.2)", note_code="No code link; the project page is announced for the dataset")
ROWS += [D("notes", "Four-page workshop paper (IEEE VRW 2024). Results exist only as data labels in Figure 6. The baseline is cited as 'Mask-RCNN based [2]', but reference [2] in the paper is unrelated (Chave 2006, wood density manual). Whether the models were fine-tuned on real images before testing is not described.", "Sec. 3.3, Fig. 6 and references (pp. 3 to 5)")]
