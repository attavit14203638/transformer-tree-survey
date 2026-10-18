import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2, T3 = "P133", "Table 2", "Table 3"
# Pixel-level detection of pine wilt disease (PWD) in UAV hyperspectral strips by deep one-class classification: the proposed network (3D convolutional layers and transformer blocks, trained with a balanced
# unbiased risk on 300 labelled PWD pixels of strip 1) against OCSVM, BSVM and Deep SVDD on the seven test strips (Table 2, precision, recall and F1 per strip; only F1 is recorded). There is no overall value over the
# strips, so the first listed strip (2) is primary (protocol 6.3 item 5); the other strips are recorded without primary. Table 3 is the same network trained as a binary classifier with labelled negative pixels
# (traditional pixel-level detection) and gives the in-domain value (strip 1) of the transfer pairs. The test pixel sets hold 10,000 PWD and 10,000 other pixels per strip (Sec. 3.4.1).
TP = Tags(task="health_classification", level="pixel", split="test", agg="single")
OBJ_EV = "a novel model consisting of three-dimensional (3D) convolutional layers and transformer blocks is proposed to extract more robust features"
DS_EV = "PWD detection experiments were conducted on eight UAV H2 (high spatial and spectral resolution) image strips"
SPLIT_EV = "strip 1 was used to make the training set, and the remaining strips formed the test set"
ROWS = [
    R("task", "health_classification", 1, "Abstract", DS_EV, note="pixel-level detection of pine trees with middle or late stage pine wilt disease (yellow or reddish-brown needles)"),
    R("output_unit", "crown_mask", 1, "Abstract", DS_EV, note="pixel masks of diseased pine trees, also converted to polygons (Figs. 6 and 7); not separated into instances"),
    R("tree_setting", "natural_forest", 2, "Sec. 2.1", "The study area for the PWD detection was Yantai Hill, Hexi Village, Shandong province, China", note="pine forest on a hill; forest type is not described"),
    R("tree_taxa", "Pinus massoniana, Pinus armandii and Pinus parviflora", 2, "Sec. 2.1", "The main pine species in the study area are Pinus massoniana Lamb, Pinus armandii Franch, and Pinus"),
    R("study_area", "Yantai Hill, Hexi Village, Shandong Province, China", 2, "Sec. 2.1", "The study area for the PWD detection was Yantai Hill, Hexi Village, Shandong province, China"),
    D("biome", "temperate", "Shandong, China (annual rainfall 750 mm, mean temperature 12 C as printed); not stated as a biome in the paper"),
    R("platform", "uav", 2, "Sec. 2.1", "Data acquisition was achieved using a Headwall Nano-Hyperspec imager mounted on a DJI M600 Pro UAV"),
    R("sensors", "hyperspectral", 2, "Sec. 2.1", "Data acquisition was achieved using a Headwall Nano-Hyperspec imager mounted on a DJI M600 Pro UAV", note="274 bands, 400 to 1000 nm"),
    R("gsd", "11", 2, "Sec. 2.1", "and a spatial resolution of 11 cm", unit="cm"),
    R("annotation_type", "polygon_mask", 5, "Sec. 3.4.1", "only 300 diseased pixels", note="300 labelled PWD pixels of strip 1 (ENVI ROI tool) and 400 randomly selected unlabelled pixels for training; 10,000 PWD and 10,000 other reference pixels per test strip"),
    R("dataset_name", "own UAV hyperspectral strips of PWD, Yantai Hill, Hexi Village (eight H2 strips of about 4600 x 700 pixels)", 2, "Sec. 2.1", "when eight UAV H2 image strips were captured", item="dataset1"),
    R("dataset_key", "own_p133_yantai_h2", 2, "Sec. 2.1", "when eight UAV H2 image strips were captured", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 3.4.1", SPLIT_EV, item="dataset1", note="strip 1 for training, strips 2 to 8 for testing (geographic isolation)"),
    R("dataset_public", "on_request", 9, "Data availability", "Data will be made available on request", item="dataset1"),
    D("n_images", "8", "eight UAV H2 image strips (the number is written as a word, Sec. 2.1), each about 4600 x 700 pixels with about 30 percent overlap; the area is 0.312 km2", item="dataset1", unit="strips"),
]
NP = dict(pre_status="not_reported", pre_note="not stated", strat_note="training settings (epochs, optimiser) not stated in the text; trained on 300 labelled PWD pixels and 400 unlabelled pixels of strip 1")
ROWS += model("model1", "Proposed deep OCC model (3D convolutional layers and transformer blocks, balanced unbiased risk)", "proposed", "cnn_transformer_hybrid", "3D convolutional layers as feature extractor followed by two transformer blocks and a fully connected layer; 15 x 15 x 274 input patches",
              None, None, 1, "Abstract", OBJ_EV, cat_note="3D convolution with transformer blocks (hybrid)", **NP)
ROWS += model("model2", "OCSVM", "other_baseline", "classical", "one-class support vector machine", None, None, 5, "Sec. 3.4.2", "one-class support vector machine (OCSVM)", cat_note="classical one-class SVM", pre_status="not_reported", pre_note="not applicable", strat_note="not applicable")
ROWS += model("model3", "BSVM", "other_baseline", "classical", "one-class biased support vector machine (uses unlabelled samples)", None, None, 5, "Sec. 3.4.2", "one-class biased support vector machine (BSVM)", cat_note="classical biased SVM", pre_status="not_reported", pre_note="not applicable", strat_note="not applicable")
ROWS += model("model4", "Deep SVDD", "other_baseline", None, "deep support vector data description (deep one-class model on spatial patches)", None, None, 5, "Sec. 3.4.2", "deep support vector data description (Deep SVDD)",
              cat_note="the paper names Deep SVDD but does not describe its network architecture", **NP)
ROWS += model("model5", "Proposed network trained as a binary classifier (traditional pixel-level detection)", "transformer_baseline", "cnn_transformer_hybrid", "same network with the balanced risk replaced by cross-entropy loss and 100 labelled pixels of each negative object kind",
              None, None, 8, "Sec. 4.4", "the balanced risk estimation was changed to cross-entropy loss", cat_note="the proposed 3D convolution and transformer network used as a binary classifier", **NP)
res = Counter("res")
RID = {}
SS = lambda s: f"Table 2: test strip {s} (10,000 PWD and 10,000 other pixels)"
for s in range(2, 9):
    nums, ev = row(PID, 6, rf"^\s*{s}\s+(?=\d\.\d{{3}}\s)", 0)
    assert len(nums) == 12, (s, nums)
    for item, f1 in (("model2", nums[2]), ("model3", nums[5]), ("model4", nums[8]), ("model1", nums[11])):
        RID[(item, s)] = res()
        ROWS += result(RID[(item, s)], "f1", item, "dataset1", f1, 6, T2, ev, unit="fraction", setting=SS(s), note="F1-score of the PWD class; precision and recall of the same row are not recorded",
                       **TP, primary=(s == 2))
nums3, ev3 = row(PID, 8, r"^\s*F1\s+(?=0\.\d{3}\s)", 0)
assert len(nums3) == 8, nums3
for s in range(2, 9):
    ROWS += result(res(), "f1", "model5", "dataset1", nums3[s - 1], 8, T3, ev3, unit="fraction", setting=f"Table 3: binary classification, test strip {s}", note="F1-score of the PWD class; strip 1 is the training strip", **TP)
# transfer pairs: training strip 1 to the unseen strips 2 to 8 (shift: site). The proposed one-class model has no printed strip 1 value; the binary variant has (Table 3)
tr = Counter("tr")
for item, nm in (("model1", "proposed deep OCC model"), ("model5", "binary classifier variant")):
    for s in range(2, 9):
        t = tr()
        if item == "model1":
            vals, evv, pg, tb = row(PID, 6, rf"^\s*{s}\s+(?=\d\.\d{{3}}\s)", 0)[0][11], row(PID, 6, rf"^\s*{s}\s+(?=\d\.\d{{3}}\s)", 0)[1], 6, T2
        else:
            vals, evv, pg, tb = nums3[s - 1], ev3, 8, T3
        ROWS += [R("transfer_source", "strip 1 (labelled PWD samples and unlabelled pixels)", 5, "Sec. 3.4.1", SPLIT_EV, item=t),
                 R("transfer_target", f"strip {s} (unseen, geographically isolated)", 5, "Sec. 3.4.1", SPLIT_EV, item=t),
                 R("shift_type", "site", 5, "Sec. 3.4.1", SPLIT_EV, item=t, note="strips captured at different times and with different object kinds (Sec. 2.2)"),
                 R("transfer_model", item, 8 if item == "model5" else 6, T3 if item == "model5" else T2, evv, item=t, note=nm),
                 R("transfer_metric", "f1", pg, tb, evv, item=t)]
        if item == "model5":
            ROWS.append(R("value_in_domain", nums3[0], 8, T3, ev3, item=t, unit="fraction", note="F1 on strip 1, the training strip (Table 3)"))
        else:
            ROWS.append(NR("value_in_domain", item=t, note="F1 of the proposed model on strip 1 is not printed (Table 2 starts at strip 2; Fig. 8b shows F1 of all eight strips without values)"))
        ROWS += [R("value_out_of_domain", vals, pg, tb, evv, item=t, unit="fraction"), R("transfer_level", "pixel", pg, tb, evv, item=t)]
ROWS += avail(code="no", weights="not_stated", data="on_request", data_page=9, data_loc="Data availability", data_ev="Data will be made available on request")
ROWS += [D("notes", "Table 2 compares the proposed one-class framework with OCSVM, BSVM and Deep SVDD on the seven test strips (precision, recall and F1 per strip; the paper gives no overall value, so F1 of every strip is recorded and strip 2, the first listed, is primary). The proposed model has the highest recall and F1 in every test strip (F1 0.912 to 0.985). Table 3 is the proposed network trained as a binary classifier with 100 labelled pixels of each negative object kind: F1 0.926 on strip 1 and lower on strips 2 to 8. The seven strip pairs of each of these two models are recorded as transfer pairs (shift site); the one-class model has no printed strip 1 value, so its pairs have no in-domain value. The proportion of PWD estimated for strip 1 is 0.013 against 0.041 printed in Table 1. The sensitivity analyses of the patch size and of the number of labelled PWD pixels (Fig. 8) show F1 values only as dots and are not recorded.",
           "Sec. 3.4 and 4, Tables 1 to 3 (pp. 3 to 8)")]
