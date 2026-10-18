import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2 = "P109", "Table 2"
# Species classification of six urban tree species by ResUnet CNNs (RGBNIR alone, with one LiDAR feature, and a fusion of two encoder-decoders). The only transformer is SAM (ViT-H), used zero-shot to delineate the
# crowns and scored with a boundary F1 (metric other, not primary). The scored unit of the F1 and Kappa is not stated (a semantic segmentation network is evaluated on ITCs not used for training), so level is not_stated.
# Table 2 (text) gives per-species F1 and the Kappa row of the eight configurations; the average F1 of the RGBNIR and fusion models is only in the text of Sec. 3. The Table 2 configurations are primary for Kappa,
# the fusion model (the paper's proposed strategy and the abstract's headline) and the RGBNIR model for the average F1.
TC = Tags(task="species_classification", level="not_stated", split="test", agg="mean")
OBJ_EV = "We found that the optical-LiDAR fusion strategy increased the average F1-score by 12.6 percentage points"
DS_EV = "This study used the ITC dataset of Martins et al. (2021)"
SPLIT_EV = "First, we randomly selected 60% of the ITCs for model training and 40% to assess the classification accuracy"

ROWS = [
    R("task", "species_classification;crown_segmentation", 1, "Abstract", OBJ_EV, note="species classification with CNNs and zero-shot SAM delineation of the crowns"),
    R("output_unit", "tree_label;crown_mask", 1, "Abstract", "SAM outlined ITCs with a boundary F1-score of 98%"),
    R("tree_setting", "urban", 1, "Abstract", "tropical urban area"),
    R("tree_taxa", "Terminalia catappa, Pachira aquatica, Licania tomentosa, Senna siamea, Tamarindus indica, Caesalpinia pluviosa", 2, "Table 1", "The study area comprises the Grajaú neighborhood in Rio de Janeiro, Brazil"),
    R("study_area", "Grajau neighborhood, Rio de Janeiro, Brazil", 2, "Sec. 2.1", "The study area comprises the Grajaú neighborhood in Rio de Janeiro, Brazil"),
    R("biome", "tropical", 1, "Abstract", "tropical urban area"),
    D("platform", "airborne", "aerial images (UltraCam-Eagle Prime) and airborne LiDAR (Trimble Harrier 68i); Sec. 2.2"),
    R("sensors", "multispectral;lidar", 2, "Sec. 2.2", "four bands (Red, Green, Blue, and Nearinfrared", note="RGBNIR aerial camera and airborne LiDAR (Trimble Harrier 68i, 1550 nm)"),
    R("gsd", "15", 2, "Sec. 2.2", "GSD of 15 cm", unit="cm"),
    R("point_density", "12", 2, "Sec. 2.2", "The average density of the point cloud was 12 points/m2", unit="pts/m2"),
    R("annotation_type", "polygon_mask", 2, "Sec. 2.3", "visually delineated in aerial RGB images", note="individual tree crowns delineated visually with species identified from the field inventory"),
    R("dataset_name", "ITC dataset of Martins et al. (2021), Grajau neighbourhood (six species), adjusted to the 2019 aerial images", 2, "Sec. 2.3", DS_EV, item="dataset1"),
    R("dataset_key", "own_p109_grajau_itcs", 2, "Sec. 2.3", DS_EV, item="dataset1"),
    R("dataset_role", "train_test", 4, "Sec. 2.6", SPLIT_EV, item="dataset1", note="five repetitions with different training and testing ITCs"),
    D("dataset_public", "no", "No data link or availability statement in this paper; the ITC dataset is that of Martins et al. (2021)", item="dataset1"),
    D("n_trees", "283", "sum of the ITCs of the six species printed in Table 1 (130, 35, 45, 24, 24 and 25)", item="dataset1", unit="trees"),
]
NPRE = dict(pre_status="not_reported", pre_note="not stated", strat_note="trained for up to 500 epochs with Adam, five repetitions; initialisation not stated")
ROWS += model("model1", "ResUnet fusion of RGBNIR and LiDAR-derived features (two encoder-decoders, proposed)", "proposed", "cnn", "two ResUnet encoder-decoders (RGBNIR and LiDAR features) whose decoder feature maps are concatenated for the classifier", None, None, 4, "Sec. 2.5",
              "We used the ResUnet architecture proposed by Zhang et al. (2018)", cat_note="residual U-Net, convolutional", **NPRE)
ROWS += model("model2", "ResUnet with RGBNIR bands only (baseline)", "cnn_baseline", "cnn", "ResUnet on the RGBNIR bands", None, None, 4, "Sec. 2.6", "The baseline model was trained using the RGBNIR bands only", cat_note="residual U-Net, convolutional", **NPRE)
STK = (("model3", "nx", "RGBNIR_nx"), ("model4", "ny", "RGBNIR_ny"), ("model5", "nz", "RGBNIR_nz"), ("model6", "curvature", "RGBNIR_curvature"), ("model7", "intensity", "RGBNIR_intensity"), ("model8", "CHM", "RGBNIR_CHM"), ("model9", "LAI", "RGBNIR_LAI"))
for item, feat, nm in STK:
    ROWS += model(item, f"ResUnet with RGBNIR stacked with the LiDAR feature {feat} ({nm})", "cnn_baseline", "cnn", f"ResUnet on the RGBNIR bands stacked with {feat}", None, None, 4, "Sec. 2.6",
                  "We trained seven models: RGBNIR_nx, RGBNIR_ny, RGBNIR_nz, RGBNIR_curvature, RGBNIR_intensity, RGBNIR_CHM and RGBNIR_LAI", cat_note="residual U-Net, convolutional", **NPRE)
ROWS += model("model10", "SAM ViT-H (zero-shot ITC delineation with an NDVI filter)", "proposed", "foundation_model", "Segment Anything Model ViT-H (samgeo, default parameters) with a mean NDVI threshold of 0.5 on the segments", None, "zero_shot", 5, "Sec. 2.7.1",
              "We used the ViT-H SAM model", cat_note="SAM, a ViT-H promptable segmentation foundation model", pre_status="not_reported", pre_note="SAM ViT-H weights of Kirillov et al. (2023); the pretraining data is not described in this paper",
              strat_page=5, strat_loc="Sec. 2.7.1", strat_ev="SAM requires no extra training to identify and segment unfamiliar objects accurately", strat_note="used with default parameters")
res = Counter("res")
# Table 2 (p.6): Kappa row of the eight configurations (mean of five runs; the SD is in the note)
kap, ev_k = row(PID, 6, r"^\s*Kappa\s+(?=\d)", 0)
assert len(kap) == 8
sds = ("8.7", "5.7", "5.7", "9.5", "6.5", "9.2", "10.6", "6.4")
for k, item in enumerate(("model2", "model3", "model4", "model5", "model6", "model7", "model8", "model9")):
    ROWS += result(res(), "kappa", item, "dataset1", kap[k], 6, T2, ev_k, unit="%", setting="Table 2: Kappa index (percent) of the five test repetitions, mean", note=f"Kappa index as printed in percent; mean of five runs, SD {sds[k]}", **TC, primary=True)
# Average F1 over the six species (text of Sec. 3, p.7)
AVG_EV = "The average of the F1-score was 71.5% and 84.1% using RGBNIR and the fusion approach, respectively"
ROWS += result(res(), "f1", "model2", "dataset1", "71.5", 7, "Sec. 3", AVG_EV, unit="%", setting="Sec. 3: F1-score averaged over the six species, RGBNIR bands only", note="average F1 of the six species (the six per-species F1 of Table 2 average to 71.5)", **TC, primary=True)
ROWS += result(res(), "f1", "model1", "dataset1", "84.1", 7, "Sec. 3", AVG_EV, unit="%", setting="Sec. 3: F1-score averaged over the six species, optical-LiDAR fusion", note="average F1 of the six species; per-species values are only in Fig. 6", **TC, primary=True, headline=True)
# SAM delineation: boundary F1 (metric other)
ROWS += result(res(), "other", "model10", "dataset1", "98", 7, "Sec. 3", "The BF score reached 98%", unit="%", setting="Sec. 3: ITCs delineated by SAM against the manually delineated ITCs", note="boundary F1 (BF) contour matching score of the SAM-delineated crowns; the evaluation set is not stated",
               **TC(task="crown_segmentation", level="instance_mask", split="not_stated", agg="single"))
ROWS += [R("limitations", "The airborne LiDAR data have a low point density, which limits the surface normals and the species discrimination", 9, "Sec. 4.4", "Our research also encountered limitations, primarily the low point density of our LiDAR data"),
         R("code_url", "https://github.com/felferrari/tree_fusion", 4, "Sec. 2.6", "The codes used in this work are freely available in the GitHub repository: https://github.com/felferrari/tree_fusion"),
         R("code_available", "yes", 4, "Sec. 2.6", "The codes used in this work are freely available in the GitHub repository: https://github.com/felferrari/tree_fusion"),
         D("weights_available", "not_stated", "No statement about trained weights"), D("data_available", "no", "No data link or availability statement in this paper")]
ROWS += [D("notes", "Only the delineation uses a transformer (SAM ViT-H, zero-shot); the species classification is done by ResUnet CNNs. Table 2 gives the F1 of each of the six species and the Kappa index (percent, mean and SD of five runs) of RGBNIR alone and stacked with one LiDAR feature at a time; Kappa is recorded for the eight configurations and the per-species F1 values are not. The fusion model combines the RGBNIR network with a LiDAR network trained on the features that did not reduce Kappa; its F1 is given only as the average over species (84.1 against 71.5 for RGBNIR, abstract: 12.6 points) and in Fig. 6. The boundary F1 of 98 percent for SAM is a contour matching score (metric other). The scored unit and the evaluation of the ITC-level species map (after SAM post-processing) are not given numerically.",
           "Sec. 2 to 3 and Table 2 (pp. 4 to 7)")]
