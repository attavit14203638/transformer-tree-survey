import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

URL = "https://github.com/rneyns/Deep_learning_based_mapping_of-_bee_friendly_trees"
S = "Sec. 4.1.1"
EV = "After training, the SAINT model achieves an overall accuracy on the validation set of 98.8 %, an area under the ROC curve (AUROC) of 0.97, and a F1 score of 0.73"
ROWS = [
    R("task", "species_classification", 4, "Sec. 3.1", "for binary classification (Salix vs. non-Salix)", note="each LiDAR-delineated tree classified as Salix or not"),
    R("output_unit", "tree_label", 4, "Sec. 3.1", "For each tree, the reflectance values in the PlanetScope time series were averaged over the pixels within the tree crown extent obtained from the LiDAR 3D tree model"),
    R("tree_setting", "urban", 3, "Sec. 2", "The municipal tree inventory of the city of Braunschweig contains a total of 415 tree species and 89,362 individual trees on public ground"),
    R("tree_taxa", "Salix (S. alba, S. caprea, S. fragilis) versus all other species", 3, "Sec. 2", "the three most common Salix species with more than a hundred samples were mapped, namely Salix alba, Salix caprea and Salix fragilis"),
    R("study_area", "Braunschweig, Lower Saxony, Germany", 3, "Sec. 2", "Braunschweig (52° 16′08″N, 10° 31′36″E) is a city in Lower Saxony, Germany"),
    D("biome", "temperate", "Braunschweig, Germany; not stated in the paper"),
    R("platform", "multiple", 3, "Sec. 2.1", "PlanetScope images are high-resolution satellite images ... The airborne LiDAR data for the city of Braunschweig", note="satellite time series for classification; airborne LiDAR for crowns"),
    R("sensors", "multispectral;lidar", 3, "Sec. 2.1.2", "Each image contains eight bands: coastal blue, blue, green-1, green, yellow, red, red edge, and near-infrared (NIR)"),
    R("gsd", "3", 3, "Sec. 2.1.2", "multispectral imagery at a spatial resolution of ~3–5 m ... each with a spatial resolution of three meters", unit="m", note="PlanetScope; crowns from LiDAR at 16 pts/m2"),
    R("annotation_type", "image_label", 5, "Sec. 3.1", "This dataset was obtained by randomly splitting the tree dataset obtained from the tree inventory", note="species labels from the municipal tree inventory"),
    R("dataset_name", "Braunschweig tree inventory with PlanetScope time series (40 dates) and LiDAR crowns", 3, "Sec. 2.1.2", "After screening, a time-series of 40 images was selected for the analysis", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. 3.1", "The number of samples in the validation set was determined as 10 % of the total number of trees, resulting in 267 samples for the Salix genus and 11,959 samples for the 'other' class",
      item="dataset1", note="results on this stratified validation set"),
    R("dataset_public", "partial", 13, "Data availability", "Due to data ownership and licensing restrictions, not all datasets used in this study can be shared publicly", item="dataset1",
      note="tree inventory available on request from the city"),
    R("n_trees", "89362", 3, "Sec. 2", "The municipal tree inventory of the city of Braunschweig contains a total of 415 tree species and 89,362 individual trees on public ground", item="dataset1", unit="trees",
      note="inventory size; the validation set has 12,226 trees"),
]
ROWS += model("model1", "SAINT (Self-Attention and Intersample Attention Transformer) on PlanetScope time series", "proposed", "pure_vit", "SAINT tabular transformer", None, None,
              4, "Sec. 3.1", "a tabular deep learning model called the Self-Attention and Intersample Attention Transformer (SAINT)", pre_status="not_reported", pre_note="not stated",
              strat_note="trained on the inventory trees with oversampling of Salix; initialisation not stated", cat_note="transformer on tabular spectral-temporal features, not image patches")
res = Counter("res")
ROWS += result(res(), "f1", "model1", "dataset1", "0.73", 5, S, EV, unit="fraction", setting="validation set; Salix class", headline=True, note="precision 0.69, recall 0.78 for Salix")
ROWS += result(res(), "oa", "model1", "dataset1", "98.8", 5, S, EV, setting="validation set; Salix vs other")
ROWS += avail(code_url=URL, page=13, loc="Data availability", ev="The code used for the deep learning model ... is publicly available at the following GitHub repository", data="partial",
              data_page=13, data_loc="Data availability", data_ev="Due to data ownership and licensing restrictions, not all datasets used in this study can be shared publicly")
ROWS += [D("notes", "Single-model study (no baseline). Overall accuracy is dominated by the non-Salix class (267 Salix vs 11,959 other trees in validation), so F1 for Salix is the informative score.", "Sec. 3.1 and 4.1.1 (pp. 4 to 5)")]
