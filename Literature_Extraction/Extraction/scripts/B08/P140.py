import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2 = "P140", "Table 2"
# Standing deadwood (dead tree crowns and partial canopy dieback) in 1 to 28 cm aerial orthophotos of deadtrees.earth: one SegFormer-B5 encoder with a U-Net decoder, no other model. The scored unit is a deadwood pixel
# (F1 of the deadwood class; precision and recall are not recorded). Results are pooled over a 3-fold spatial block cross-validation (each orthophoto is a block). Table 2 gives F1 per biome group at the biome's best
# resolution; the text gives the mean over the four biome groups (0.61), which is the only overall value, so it is primary and the headline (protocol 6.3 item 5 applies only when no overall value exists). The
# per-biome values are the best resolution bin of each biome (aggregation best) and are not primary. Fig. 4 (F1 by biome and resolution bin) is not recorded.
TF = Tags(task="mortality_detection", level="pixel", split="cross_validation", agg="mean")
MODEL_EV = "we present a transformer-based semantic segmentation model that generalizes across forest biomes"
DS_EV = "consists of 434 sets of orthophotos and labels"
DATA_EV = "are available as orthophoto and tiled dataset on https://deadtrees.earth/"
ROWS = [
    R("task", "mortality_detection", 1, "Abstract", MODEL_EV, note="segmentation of standing deadwood: fully dead tree crowns and partial canopy dieback (leafless branches)"),
    R("output_unit", "crown_mask", 1, "Abstract", MODEL_EV, note="semantic deadwood mask; a predicted polygon is a contiguous deadwood area that can be a branch, partial dieback, a dead tree or a group of dead trees (p.6)"),
    R("tree_setting", "natural_forest", 3, "Sec. 2", "The resulting four biome groups were (1) (sub) tropical, (2) boreal and montane, (3) drylands (including Mediterranean), and (4) temperate forests",
      note="forests of all major biomes; managed forests and plantations are not separated"),
    R("tree_taxa", "not specified (trees of all major forest biomes; broadleaved and coniferous)", 3, "Sec. 2", "The resulting four biome groups were (1) (sub) tropical, (2) boreal and montane, (3) drylands (including Mediterranean), and (4) temperate forests"),
    R("study_area", "global: 434 orthophotos of deadtrees.earth in four biome groups ((sub)tropical, temperate, boreal and montane, drylands)", 3, "Sec. 2", DS_EV),
    R("biome", "multiple", 3, "Sec. 2", "The resulting four biome groups were (1) (sub) tropical, (2) boreal and montane, (3) drylands (including Mediterranean), and (4) temperate forests"),
    R("platform", "multiple", 3, "Sec. 2", "aerial images collected from drones and airplanes"),
    R("sensors", "rgb", 10, "Sec. 6", "aerial imagery with only RGB channels", note="the model takes RGB orthophotos; resolutions 1 to 26.2 cm in the data, rescaled to 2 to 28 cm in 2 cm steps"),
    R("annotation_type", "polygon_mask", 3, "Sec. 2", "High-quality labels precisely delineate the shape of dead trees as detailed polygons", note="medium-quality labels are simple circles; low-quality labels are not used"),
    R("dataset_name", "deadtrees.earth standing deadwood orthophotos (434 labelled orthophotos, 10,778 ha)", 3, "Sec. 2", DS_EV, item="dataset1"),
    R("dataset_key", "deadtrees_earth", 3, "Sec. 2", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 6, "Sec. 4", "we employed a 3-fold spatial block cross-validation strategy, where each orthophoto is considered a spatial block", item="dataset1", note="no separate test set; the results are pooled over the held-out orthophotos of three folds"),
    R("dataset_public", "yes", 11, "Data availability", DATA_EV, item="dataset1"),
    R("dataset_url", "https://deadtrees.earth/", 11, "Data availability", DATA_EV, item="dataset1"),
    R("n_images", "434", 3, "Sec. 2", DS_EV, item="dataset1", unit="orthophotos", note="10,778 ha labelled area; 378,430 tile pairs of 1024 x 1024 pixels after rescaling and tiling (p.6)"),
]
ROWS += model("model1", "SegFormer-B5 with U-Net decoder", "proposed", "hierarchical_vit", "SegFormer-B5 encoder with the standard U-Net decoder (segmentation_models.pytorch)", "imagenet", "full_fine_tuning", 4, "Sec. 3.2",
              "The resulting model consists of a SegFormer-B5 encoder, pretrained on ImageNet", cat_note="SegFormer MiT encoder (protocol 4.5: hierarchical_vit)",
              pre_note="SegFormer-B5 encoder pre-trained on ImageNet", strat_note="all 84.7 M parameters are trainable; AdamW, 105 epochs, learning rate warmed up to 1e-5, Focal Tversky Loss (alpha 0.2, beta 0.8, gamma 2), three A6000 GPUs with batch size 4 each",
              strat_page=5, strat_loc="Sec. 3.3", strat_ev="The model was trained for 105 epochs")
ROWS.append(R("trainable_params", "84.7", 4, "Sec. 3.2", "In total, the model, including the encoder and decoder parts, consists of 84.7 million trainable parameters", item="model1", unit="M", note="encoder and decoder"))
res = Counter("res")
ROWS += result(res(), "f1", "model1", "dataset1", "0.61", 6, "Sec. 4", "This yielded a mean F1-score performance of 0.61, averaged across biomes", unit="fraction", setting="mean over the four biome groups, each at its best resolution; pooled 3-fold spatial block cross-validation",
               note="F1 of the standing deadwood class; mean of the four per-biome values of Table 2 (0.51, 0.71, 0.60, 0.60)", **TF, primary=True, headline=True)
for lab, name in ((r"\(Sub\) Tropical", "(Sub) Tropical"), ("Temperate", "Temperate"), ("Boreal and Montane", "Boreal and Montane"), ("Drylands", "Drylands")):
    nums, _ = row(PID, 6, rf"^\s*{lab}\s+(?=0\.)", 0, cols=(0, 90))
    assert len(nums) == 3, (name, nums)  # F1, precision, recall
    ROWS += result(res(), "f1", "model1", "dataset1", nums[0], 6, T2, f"{name} {nums[0]} {nums[1]} {nums[2]}", unit="fraction", setting=f"Table 2: {name} biome group at its best resolution; pooled 3-fold spatial block cross-validation",
                   note="F1 of the standing deadwood class at the best-performing resolution bin of the biome group; precision and recall of the row are not recorded", **TF(agg="best"))
ROWS += [R("limitations", "Only a SegFormer-based model is reported (no benchmark of other architectures), and the heterogeneous crowd-sourced labels lower the F1 scores", 9, "Sec. 5",
           "we report results only for a SegFormer-based model and did not systematically benchmark alternative architectures")]
ROWS += conflict("cf1", "model1.trainable_params", "81", 4, "Sec. 3.2", "we chose the largest SegFormer variant (B5), which has about 81 million trainable parameters", "V1b",
                 "the paper gives 84.7 M trainable parameters for the whole model (encoder and U-Net decoder); 81 M is the SegFormer-B5 encoder alone", unit="M")
ROWS += avail(code_url="https://github.com/jmoeh/standingdeadwood", weights="yes", data="yes", page=10, loc="Code availability", ev="All code and ready-to-use models are available online",
              data_page=11, data_loc="Data availability", data_ev=DATA_EV)
ROWS += [D("notes", "A single SegFormer-B5 and U-Net decoder model for standing deadwood (dead tree crowns and partial canopy dieback) in 1 to 28 cm orthophotos of deadtrees.earth, evaluated by pooled 3-fold spatial block cross-validation (Table 2: F1 0.51 tropical, 0.71 temperate, 0.60 boreal and montane, 0.60 drylands, each at the best resolution; mean 0.61). The mean of the four biome values is the only overall value and is primary and the headline; the biome rows are the best resolution bin of each biome and are not primary. Only F1 is recorded (precision 0.40 to 0.59, recall 0.69 to 0.90). The scored unit is a deadwood pixel, not a tree: labels include partial canopy dieback and medium-quality labels are circles, and the authors state that low F1 reflects crowd-sourced label quality. There is no model comparison (no baseline is trained). Fig. 4 (F1 by biome and resolution bin) is not recorded.",
           "Sec. 4 and 5, Table 2 (pp. 5 to 9)")]
