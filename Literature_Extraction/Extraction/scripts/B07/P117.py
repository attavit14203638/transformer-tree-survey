import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

PID, T2 = "P117", "Table 2 (image)"
# 3DPCT (3D Point Cloud Transformer with dual self-attention) classifies 11 species of single-tree ALS point clouds (1291 trees, German forests). Table 2 is an image with the overall accuracy of six input settings
# (number of points and channels) and the precision and recall per species. The paper has no comparison model and does not describe the split. The best setting (256 points, 8 channels; the abstract's 86.1) is
# the headline and the primary setting, the other five settings are not primary (input ablations without a stated default).
TS = Tags(task="species_classification", level="tree", split="not_stated", agg="single")
OBJ_EV = "There are a total 1291 single tree point clouds of 11 different species from coniferous and deciduous used in this paper"
DS_EV = "The dataset used in this paper was released by 3DGeo Research Group"

ROWS = [
    R("task", "species_classification", 1, "Abstract", "to conduct a tree species classification of Airborne LiDAR data", note="single tree classification of sparse ALS point clouds"),
    R("output_unit", "tree_label", 1, "Abstract", "single tree classification of sparse ALS point cloud"),
    D("tree_setting", "mixed", "managed German forests with 11 conifer and deciduous species; not stated as mixed in the paper"),
    R("tree_taxa", "Abies alba, Acer pseudoplatanus, Carpinus betulus, Fagus sylvatica, Juglans regia, Larix decidua, Picea abies, Pinus sylvestris, Pseudotsuga menziesii, Quercus petraea, Quercus rubra", 2, "Table 1", "We finally retrieved the total of 1291 single tree samples with 11 tree species"),
    R("study_area", "German forests (plots BR01, BR03, BR05, KA09, KA10 and KA11 of the 3DGeo dataset)", 2, "Sec. 2", "which include BR01, BR03, BR05, KA09, KA10, and KA11 of the whole dataset"),
    D("biome", "temperate", "Germany; coded from the study area"),
    R("platform", "airborne", 2, "Sec. 2", "The flight was conducted by a Cessna C207 aircraft equipped with a RIEGL VQ-780i"),
    R("sensors", "lidar", 2, "Sec. 2", "The flight was conducted by a Cessna C207 aircraft equipped with a RIEGL VQ-780i"),
    R("point_density", "72.5", 2, "Sec. 2", "The average density of the whole scanning is 72.5 pts m−2", unit="pts/m2", note="whole scan; standard deviation 31.0 pts/m2"),
    R("annotation_type", "image_label", 2, "Sec. 2", DS_EV, note="species label of each single tree from the field measurements of the 3DGeo dataset"),
    R("dataset_name", "3DGeo individual ALS tree point clouds of six German plots (11 species, 1291 trees; Weiser et al.)", 2, "Sec. 2", DS_EV, item="dataset1"),
    R("dataset_key", "weiser_2022_als_trees", 2, "Sec. 2", DS_EV, item="dataset1", note="the paper cites Weiser et al., 'Individual tree point clouds and tree measurements from multi-platform laser scanning in German forests'"),
    R("dataset_role", "train_test", 2, "Sec. 2", "We finally retrieved the total of 1291 single tree samples with 11 tree species", item="dataset1", note="the split into training and test samples is not described"),
    R("dataset_public", "yes", 2, "Sec. 2", DS_EV, item="dataset1", note="dataset released by the 3DGeo Research Group"),
    R("n_trees", "1291", 2, "Sec. 2", "We finally retrieved the total of 1291 single tree samples with 11 tree species", item="dataset1", unit="trees", note="species with fewer than 19 samples were removed"),
]
ROWS += model("model1", "3DPCT (3D Point Cloud Transformer with dual self-attention)", "proposed", "cnn_transformer_hybrid", "multi-scale kNN local feature aggregation with graph convolution and a global feature learning block of channel-wise and point-wise patch self-attention", None, None, 3, "Sec. 3",
              "two parallel self-attention blocks: the channel-wise self-attention (CSA)", cat_note="graph convolution local feature aggregation followed by a transformer-based global feature block (3DPCT of Lu et al. 2022)",
              pre_status="not_reported", pre_note="not stated", strat_page=3, strat_loc="Sec. 4", strat_ev="We used the Adam optimizer, and the weight decay was set to 0.0001",
              strat_note="Adam, learning rate 0.001, batch size 32, 400, 200 and 200 epochs for 1024, 512 and 256 points; initialisation not stated")
res = Counter("res")
for pts, ch, oa in (("256", "8", "86.1"), ("512", "8", "85.6"), ("1024", "8", "84.8"), ("512", "4", "84.0"), ("1024", "4", "82.1"), ("256", "4", "81.9")):
    best = (pts, ch) == ("256", "8")
    ev = f"No. of points {pts} No. of channels {ch} OA {oa}%"
    ROWS += result(res(), "oa", "model1", "dataset1", oa, 4, T2, ev, unit="%", setting=f"Table 2: {pts} input points and {ch} channels (x, y, z, intensity{' and four more fields' if ch == '8' else ''})",
                   note="overall accuracy of the 11-species classification; the split is not stated", **TS, primary=best, headline=best)
ROWS += [R("limitations", "The Acer pseudoplatanus class (20 samples) is hardest to classify, copying points to fill sparse samples may distort the point distribution, and the use of dense and sparse samples needs more study", 3, "Sec. 3",
           "the AcePse was the hardest class to distinguish that may because only 20 samples of this category"),
         D("code_available", "no", "No code repository or availability statement in the paper"), D("weights_available", "not_stated", "No statement about trained weights"),
         R("data_available", "yes", 2, "Sec. 2", DS_EV)]
ROWS += [D("notes", "Table 2 (an image) gives the overall accuracy and the precision and recall of each of the 11 species for six settings (1024, 512 and 256 points with 4 or 8 channels). Only the overall accuracy of the six settings is recorded; the best setting (256 points, 8 channels, OA 86.1) is the abstract's result and primary. The paper has no comparison model, and the split between training and test trees is not described. The mean number of points per sample is 5516 and 21, 37 and 79 samples have fewer than 256, 256 to 512 and 512 to 1024 points.",
           "Sec. 2 to 4 and Table 2 (pp. 2 to 4)")]
