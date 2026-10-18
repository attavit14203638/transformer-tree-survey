import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T2, T5 = "P116", "Table 2", "Table 5"
ROWS = [
    R("task", "species_classification", 1, "Abstract", "Deep Learning image-classification networks were used to automatically determine the species of each tree or bush and estimate biodiversity"),
    R("output_unit", "tree_label", 1, "Abstract", "Patches from single trees and bushes were manually extracted from the collected orthomosaics"),
    R("tree_setting", "natural_forest", 5, "Sec. 2.1", "Fir (Abies mariesii) covers about 87% of the area and can be found in both pure and mixed forests"),
    R("tree_taxa", "Abies mariesii, Fagus crenata, Sorbus commixta and other mountain species (9 classes)", 10, "Sec. 2.2.1", "we considered all the data available (with 4 sites and 9 tree species (Table 1))"),
    R("study_area", "Zao mountains, northeastern Japan (four sites)", 5, "Fig. 3", "Location of the four data-collection sites (Zao mountains, northeastern Japan)"),
    D("biome", "temperate", "mid-to-high-altitude forests of the Zao mountains, Japan; not stated in the paper"),
    R("platform", "uav", 6, "Sec. 2.1.2", "The images of the forest were acquired using a Mavic 2 Pro UAV (DJI"),
    R("sensors", "rgb", 1, "Abstract", "we used Unmanned Aerial Vehicles to gather RGB images"),
    R("gsd", "1.5", 6, "Sec. 2.1.2", "1.5–2.1 cm/pixel, the following camera settings were used", unit="cm", note="1.5 to 2.1 cm"),
    R("annotation_type", "image_label", 6, "Sec. 2.1.3", "100 × 100 pixel patches centered at every marked point. This corresponds to ∼ 2 × 2 m", note="one species label per tree or bush patch"),
    R("dataset_name", "own UAV patch dataset, Zao mountains (five-species controlled subset of sites 1 and 2)", 10, "Sec. 2.2.1", "5315 patches from five different classes (Table 1, green cells)", item="dataset1"),
    R("dataset_role", "train_test", 7, "Sec. 2.1.3", "The patches thus built were separated into training, validation, and test sets consisting of 80%, 10%, and 10% of the total patches, respectively", item="dataset1"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    R("n_images", "5315", 10, "Sec. 2.2.1", "5315 patches from five different classes (Table 1, green cells)", item="dataset1", unit="images", note="8583 patches over all sites and species"),
    R("dataset_name", "own UAV patch dataset, Zao mountains (all four sites, nine species; leave-one-site-out folds)", 10, "Sec. 2.2.1",
      "we considered all the data available (with 4 sites and 9 tree species (Table 1)) and broke them down into three sites", item="dataset2"),
    R("dataset_role", "transfer_target", 10, "Sec. 2.2.1", "In the first fold, site 1 was used for testing and sites 2, 3, and 5 for training", item="dataset2"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset2"),
    R("n_images", "8583", 6, "Sec. 2.1.3", "The total number of patches labeled was 8583", item="dataset2", unit="images"),
]
FT = dict(pre_page=10, pre_loc="Sec. 2.2.1", pre_ev="we started from DLNs with ImageNet weights, and we intended to retrain all their layers (unfrozen)",
          strat_page=10, strat_loc="Sec. 2.2.1", strat_ev="we intended to retrain all their layers (unfrozen)")
M = [("model1", "Swin", "Swin Transformer (tiny)", "proposed", "hierarchical_vit", "Swin-T", "best model; used for all later experiments"),
     ("model2", "ConvNext", "ConvNeXt", "cnn_baseline", "cnn", "ConvNeXt (size not stated)", ""),
     ("model3", "ResNet", "ResNet", "cnn_baseline", "cnn", "ResNet (depth not stated)", ""),
     ("model4", "RegNet", "RegNet", "cnn_baseline", "cnn", "RegNet (size not stated)", "")]
acc, ev_a = row(PID, 12, r"^\s*mean\s+(?=\d)", 0)
f1, ev_f = row(PID, 12, r"^\s*mean\s+(?=\d)", 3)
ORDER = ["RegNet", "ConvNext", "ResNet", "Swin"]
res = Counter("res")
for item, lab, name, role, cat, bb, note in M:
    k = ORDER.index(lab)
    ROWS += model(item, name, role, cat, bb, "imagenet", "full_fine_tuning", 12, T2, f"Metric RegNet ConvNext ResNet Swin; {lab}", cat_note=note, **FT)
    st = "test set of the five-species subset; mean of 10 runs"
    ROWS += result(res(), "f1", item, "dataset1", f1[k], 12, T2, "F1 Score mean " + ev_f.split("mean ", 1)[-1], unit="fraction", setting=st,
                   note="column order RegNet, ConvNext, ResNet, Swin")
    ROWS += result(res(), "oa", item, "dataset1", acc[k], 12, T2, "Accuracy mean " + ev_a.split("mean ", 1)[-1], unit="fraction", setting=st)
    best, ev_b = row(PID, 13, rf"^\s*{lab}\s+(?=\d)", 0)
    assert len(best) == 4, (lab, best)
    st_b = "test set of the five-species subset; best of 10 runs"
    ROWS += result(res(), "f1", item, "dataset1", best[3], 13, "Table 3", ev_b, setting=st_b, headline=(item == "model1"),
                   note=("the paper treats the peak result as the most relevant (p.13)" if item == "model1" else ""))
    ROWS += result(res(), "oa", item, "dataset1", best[0], 13, "Table 3", ev_b, setting=st_b)
tr = Counter("tr")
vf, ev_vf = row(PID, 17, r"^\s*F1\s+(?=\d)", 0)
tf, ev_tf = row(PID, 17, r"^\s*F1\s+(?=\d)", 1)
FOLD = [("site 1", "sites 2, 3 and 5"), ("site 2", "sites 1, 3 and 5"), ("sites 3 and 5", "sites 1 and 2")]
for k, (test_site, train_sites) in enumerate(FOLD):
    t = tr()
    ROWS += [R("transfer_source", f"{train_sites} (training); in-domain value on their validation split", 10, "Sec. 2.2.1",
               "In the first fold, site 1 was used for testing and sites 2, 3, and 5 for training. In the second fold, site 2 was used for testing and sites 1, 3, and 5 for training", item=t),
             R("transfer_target", f"held-out {test_site}", 10, "Sec. 2.2.1", "in the third fold, sites 3 and 5 were used for testing and sites 1 and 2 were used for training", item=t),
             R("shift_type", "site", 10, "Sec. 2.2.1", "The class balance between the 3 testing folds and from the training set to the testing set within every fold varied greatly", item=t,
               note="different site and class balance"),
             R("transfer_model", "model1", 10, "Sec. 2.2.1", "For the sake of brevity, we made one single run with the Swin model for every fold", item=t),
             R("transfer_metric", "f1", 17, T5, "F1", item=t, note="nine classes; single run per fold"),
             R("value_in_domain", vf[k], 17, T5, "VALIDATION ... " + ev_vf, item=t, unit="%"),
             R("value_out_of_domain", tf[k], 17, T5, "TEST ... " + ev_tf, item=t, unit="%")]
ROWS += avail()
ROWS += [D("notes", "Leave-one-site-out results are recorded per fold because the printed validation average F1 (94.99) is not the mean of the three folds (91.66); the test average (51.77) is. Data-quantity results (Fig. 8) and biodiversity indices (Tables 4 to 5) are not recorded.", "Sec. 3 and Tables 2, 3 and 5 (pp. 12 to 17)")]
