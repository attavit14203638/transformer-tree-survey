import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T2, T4 = "P111", "Table 2", "Table 4"
ROWS = [
    R("task", "species_classification", 1, "Abstract", "we propose an automated, data-driven dual-stream deep learning framework that integrates ALS data with point-cloud metrics to identify individual tree species"),
    R("output_unit", "tree_label", 1, "Abstract", "16,269 labelled individual tree point-clouds of four species"),
    R("tree_setting", "natural_forest", 2, "Sec. 2.1", "The Romeo Malette Forest (RMF; Fig. 1) is a 630,000 ha managed", note="managed boreal mixed-species forest"),
    R("tree_taxa", "black spruce; jack pine; trembling aspen; paper birch", 6, "Table 3", "Black Spruce ... Jack Pine ... Paper Birch ... Trembling Aspen"),
    R("study_area", "Romeo Malette Forest, Ontario, Canada (630,000 ha)", 2, "Sec. 2.1", "The Romeo Malette Forest (RMF; Fig. 1) is a 630,000 ha managed"),
    R("biome", "boreal", 1, "Abstract", "across a 630,000 ha boreal mixed species forest in Ontario, Canada"),
    R("platform", "airborne", 2, "Sec. 2.2", "Flights were conducted at a 3800 m altitude"),
    R("sensors", "lidar", 2, "Sec. 2.2", "which emits a 10 × 10 array of green (λ = 532 nm) beamlets", note="single-photon airborne LiDAR"),
    R("point_density", "40", 2, "Sec. 2.2", "yielding a return density of 40 points/m2 after processing", unit="pts/m2"),
    R("annotation_type", "image_label", 5, "Sec. 3.1", "The process of segmenting and labelling individual tree point-clouds resulted in a total of 16,269 samples being generated",
      note="one species label per segmented tree, from stem-mapped plots and pure-stand pseudo-plots"),
    R("dataset_name", "own ALS individual-tree dataset, Romeo Malette Forest", 5, "Sec. 3.1", "resulted in a total of 16,269 samples being generated (Table 3)", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 3.1", "with 11,388 for training, 2440 for validation, and 2441 for testing", item="dataset1"),
    R("dataset_public", "no", 13, "Data availability", "The authors do not have permission to share data.", item="dataset1"),
    R("n_trees", "16269", 5, "Sec. 3.1", "resulted in a total of 16,269 samples being generated (Table 3)", item="dataset1", unit="trees"),
]
TRN = dict(pre_status="not_reported", pre_note="not stated", strat_page=5, strat_loc="Sec. 2.4", strat_ev="training was conducted for a maximum of 300 epochs")
M = [  # item, Table 4 label regex, Table 2 label, name, role, category, backbone, note
    ("model1", r"Point Transformer \+", "Point Transformer + Metrics Network", "Dual-stream model: Point Transformer extractor with Metrics Network", "proposed", "pure_vit",
     "Point Transformer layer with a TabNet-inspired metrics network", "proposed dual-stream framework"),
    ("model2", r"Multi-Branch \+", "Multi-Branch + Metrics Network", "Dual-stream model: multi-branch (EdgeConv and Point Transformer) extractor with Metrics Network", "proposed",
     "cnn_transformer_hybrid", "parallel EdgeConv and Point Transformer with metrics network", "proposed dual-stream framework"),
    ("model3", r"EdgeConv \+ Metrics", "EdgeConv + Metrics Network", "Dual-stream model: EdgeConv extractor with Metrics Network", "proposed", "cnn",
     "EdgeConv graph convolution with metrics network", "proposed dual-stream framework (graph-convolution extractor)"),
    ("model4", r"Point Transformer\s+(?=\d)", "Point Transformer", "Point Transformer extractor only", "transformer_baseline", "pure_vit", "Point Transformer layer", "ablation without the Metrics Network"),
    ("model5", r"Multi-Branch\s+(?=\d)", "Multi-Branch", "Multi-branch extractor only", "transformer_baseline", "cnn_transformer_hybrid", "parallel EdgeConv and Point Transformer",
     "ablation without the Metrics Network"),
    ("model6", r"EdgeConv\s+(?=\d)", "EdgeConv", "EdgeConv extractor only", "cnn_baseline", "cnn", "EdgeConv graph convolution", "ablation without the Metrics Network"),
]
res = Counter("res")
for item, rx, lab2, name, role, cat, bb, note in M:
    nums, ev = row(PID, 6, rf"^\s*{rx}", 0, cols=(0, 90))
    assert len(nums) == 5, (item, nums, ev)
    p2, e2 = row(PID, 5, rf"^\s*{re.escape(lab2)}\s+(?=\d)", 0, cols=(0, 90))
    ROWS += model(item, name, role, cat, bb, None, None, 6, T4, ev, cat_note=note, **TRN)
    import re
    printed = re.search(r"\d{1,3}(?:,\d{3})+", e2).group(0)
    ROWS += [D("trainable_params", f"{int(printed.replace(',', '')) / 1e6:.6f}".rstrip("0"), f"{printed} trainable parameters in Table 2 (p.5) / 1,000,000", item=item, unit="M")]
    st = "test set of 2441 trees; four species"
    ROWS += result(res(), "f1", item, "dataset1", nums[1], 6, T4, ev, unit="fraction", setting=st, headline=(item == "model3"),
                   note="weighted F1-score" + ("; the paper's best configuration (highest scores, p.5)" if item == "model3" else ""))
    ROWS += result(res(), "oa", item, "dataset1", nums[0], 6, T4, ev, unit="fraction", setting=st)
ROWS += avail(data_page=13, data_loc="Data availability", data_ev="The authors do not have permission to share data.")
ROWS += [D("notes", "Ablation of point extractors (EdgeConv, Point Transformer, both) with and without the metrics stream; the three dual-stream models perform alike (weighted F1 0.69 to 0.70). Weighted F1, precision, recall and ROC AUC are in Table 4.", "Sec. 2.5 to 3.3 and Tables 2 to 4 (pp. 5 to 6)")]
