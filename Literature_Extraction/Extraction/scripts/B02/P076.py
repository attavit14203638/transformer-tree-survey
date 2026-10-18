import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

URL = "https://github.com/ForestryIIP/3DCiLBE"
ROWS = [
    R("task", "species_classification", 5, "Sec. 2.2.1", "(2#) MLiDAR-CLIP: Species identification of single trees",
      note="the individual-tree step of a biomass pipeline; the SAM step segments vegetation areas and biomass is reported per area"),
    R("output_unit", "tree_label", 5, "Sec. 2.2.1", "the segmented vegetation images need to be used for single tree species identification"),
    R("tree_setting", "urban", 1, "Abstract", "Accurately estimating vegetation biomass in urban forested areas"),
    NR("tree_taxa", note="42 common urban species used as control images; species not listed"),
    R("study_area", "Boston, Chicago, Denver, Detroit, Houston, Las Vegas, Los Angeles and Miami, United States", 3, "Sec. 2.1.1",
      "eight American cities (Boston, Chicago, Denver, Detroit, Houston, Las Vegas, Los Angeles, and Miami)"),
    D("biome", "multiple", "eight US cities with 'varying climatic features' (p.3)"),
    R("platform", "airborne", 3, "Table 1", "LiDAR-Urban USGS", note="USGS airborne LiDAR"),
    R("sensors", "lidar", 3, "Sec. 2.1.1", "This includes LiDAR data along with corresponding OSM data"),
    R("gsd", "1", 3, "Sec. 2.1.1", "we have set a uniform image resolution of 1m to ensure the consistency of data", unit="m", note="LiDAR rasterised to 1 m"),
    R("annotation_type", "image_label", 7, "Sec. 2.2.3", "42 commonly found urban vegetation LiDAR images, along with species labels, were chosen as control images"),
    R("dataset_name", "USGS LiDAR and OpenStreetMap data for eight US cities (2012 to 2020)", 3, "Sec. 2.1.1",
      "LiDAR data along with corresponding OSM data for handpicked regions within eight American cities ... from 2012 to 2020", item="dataset1"),
    R("dataset_role", "test_only", 11, "Sec. 3.2.2", "Tests conducted in eight cities show that MLiDAR-CLIP had an average recognition accuracy of 92.72%", item="dataset1"),
    R("dataset_public", "yes", 3, "Table 1", "LiDAR-Urban USGS ... OSM OpenStreetMap", item="dataset1", note="public USGS and OpenStreetMap sources"),
    R("dataset_name", "GRO single-tree LiDAR images [34]", 3, "Sec. 2.1.1", "we collect LiDAR image data for single trees (GRO) [34], and these images will play a role in the MLiDAR-CLIP", item="dataset2"),
    R("dataset_role", "train_test", 7, "Sec. 2.2.3", "42 commonly found urban vegetation LiDAR images, along with species labels, were chosen as control images", item="dataset2",
      note="labelled control images for the CLIP matching"),
    NR("dataset_public", item="dataset2", note="availability not stated"),
]
ROWS += model("model1", "MLiDAR-CLIP (multi-view LiDAR CLIP)", "proposed", "vision_language_model", "CLIP text and image encoders", "clip", "few_shot", 7, "Sec. 2.2.3",
              "we propose an efficient and accurate vegetation species identification method called MLiDAR-CLIP", pre_page=5, pre_loc="Sec. 2.2.1",
              pre_ev="we propose the (2#) MLiDAR-CLIP method based on CLIP models", strat_page=7, strat_loc="Sec. 2.2.3",
              strat_ev="42 commonly found urban vegetation LiDAR images, along with species labels, were chosen as control images",
              strat_note="species assigned by similarity to the labelled control images; no weight training described")
res = Counter("res")
ROWS += result(res(), "oa", "model1", "dataset1", "92.72", 11, "Sec. 3.2.2", "Tests conducted in eight cities show that MLiDAR-CLIP had an average recognition accuracy of 92.72%",
               setting="single trees from eight cities; average over cities", headline=True, note="species recognition accuracy (Φ); 11.5% above SVM and 4.8% above PointCLIP")
ROWS += avail(code_url=URL, page=5, loc="Sec. 2.2.1", ev=f"The implementation is available at: {URL}")
ROWS += [D("notes", "LiDAR-SAM (Table 3: mIoU 0.94 vs SAM 0.91 and PointNet++ 0.85) is a binary vegetation-area segmentation ('goal areas' vs 'other areas', Table 2), so it is not an individual-tree result and is not recorded; biomass is reported per city in kg/m2 (Tables 4 to 5). Baseline species accuracies are only in Figure 5.", "Sec. 2.2 and 3.2 (pp. 5 to 12)")]
