import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

T1 = "Table 1"
ROWS = [
    R("task", "detection;counting", 1, "Abstract", "detecting mangrove seedlings ... In the first approach, seedling density was predicted ... In the second approach ... object detection"),
    R("output_unit", "tree_instance;tree_count", 3, "Methods", "integral over any region corresponds to the count of seedlings in that area", note="density map with DoG peak detections; DETR boxes"),
    R("tree_setting", "mangrove", 2, "Methods", "the mangrove communities are entirely dominated by the species Avicennia marina", note="aerial seeding restoration sites"),
    R("tree_taxa", "Avicennia marina seedlings", 3, "Methods", "A model was trained to detect seedlings of the mangrove species A. marina"),
    R("study_area", "Ghanadah and Al Taweelah, Emirate of Abu Dhabi, United Arab Emirates; 22 seeding sites, 0.41 km2", 2, "Methods", "A total of 22 sites, covering an area of 0.41 km²"),
    R("biome", "arid", 2, "Methods", "The area is characterised by extreme aridity and hypersaline conditions"),
    R("platform", "uav", 2, "Methods", "a Phase One iXM-100 RGB camera with a 35 mm lens mounted on a DJI M300 UAV"),
    R("sensors", "rgb", 2, "Methods", "Phase One iXM-100 RGB camera"),
    R("gsd", "0.85", 2, "Methods", "orthomosaics at 0.6 cm spatial resolution, subsequently upsampled to 0.85 cm", unit="cm"),
    R("annotation_type", "point", 3, "Methods", "point labels were rasterized into the same spatial resolution (0.85 cm) as the UAV imagery and then blurred",
      note="points converted to buffered squares for DETR training"),
    R("dataset_name", "own UAV mangrove seedling dataset (Abu Dhabi, 2.2 m image chips)", 3, "Methods", "A total of 11,088 image chips of 2.2 m × 2.2 m in size were randomly extracted from UAV orthomosaics", item="dataset1"),
    R("dataset_role", "train_test", 4, "Methods", "stratified into training (90%, n = 9,979) and validation (10%, n = 1,109) datasets", item="dataset1",
      note="results are on the validation split; no separate test set"),
    R("dataset_public", "on_request", 12, "Data availability", "not publicly available due to data sharing agreements ... may be made available by the corresponding author upon reasonable request", item="dataset1"),
    R("n_images", "11088", 3, "Methods", "A total of 11,088 image chips", item="dataset1", unit="images"),
    NR("n_trees", item="dataset1", note="number of labelled seedlings not given"),
]
ROWS += model("model1", "MaxViT-UNet density estimation", "proposed", "hierarchical_vit", "MaxViT-tiny encoder with UNet decoder", "imagenet", "full_fine_tuning",
              4, "Methods", "A density estimation model was developed using a MaxViT encoder pre-trained on ImageNet-1K and a UNet decoder",
              strat_ev="The model was fine-tuned with multiple augmentations", cat_note="MaxViT: multi-axis attention with MBConv blocks")
ROWS += [R("total_params", "31.8", 4, "Methods", "This resulted in a final model with 31.8 million parameters", item="model1", unit="M")]
ROWS += model("model2", "ResNet-DETR", "transformer_baseline", "cnn_transformer_hybrid", "ResNet-50", "other", "full_fine_tuning",
              5, "Methods", "a DETR model with a ResNet-50 backbone (ResNet-DETR) pre-trained on COCO dataset using 300 object queries",
              pre_note="COCO", strat_ev="fine-tuned using the same number of epochs, data augmentations, batch size, weight decay, and learning rate scheduler")
ROWS += [R("total_params", "42", 5, "Methods", "The final ResNet-DETR model contained 42 million parameters", item="model2", unit="M")]
TAB = [  # encoder/decoder label, model, resolution, kernel, threshold, F1
    ("ResNet-50 DETR", "model2", "0.85", "14.5", "0.2", "0.62"), ("MaxViT-tiny UNet", "model1", "0.85", "14.5", "1.5", "0.68"),
    ("ResNet-50 DETR", "model2", "0.85", "16.2", "0.2", "0.63"), ("MaxViT-tiny UNet", "model1", "0.85", "16.2", "1.3", "0.69"),
    ("ResNet-50 DETR", "model2", "0.85", "19.6", "0.2", "0.64"), ("MaxViT-tiny UNet", "model1", "0.85", "19.6", "1.3", "0.70"),
    ("ResNet-50 DETR", "model2", "1.7", "19.6", "0.25", "0.60"), ("MaxViT-tiny UNet", "model1", "1.7", "19.6", "1.3", "0.62"),
    ("ResNet-50 DETR", "model2", "3.4", "19.6", "0.2", "0.49"), ("MaxViT-tiny UNet", "model1", "3.4", "19.6", "1.3", "0.46"),
]
res = Counter("res")
for lab, m, resn, k, thr, f1 in TAB:
    ROWS += result(res(), "f1", m, "dataset1", f1, 7, T1, f"{lab} {resn} {k} {thr} ... F1-score {f1}", unit="fraction",
                   setting=f"validation split; {resn} cm imagery; kernel {k} cm", headline=(m == "model1" and resn == "0.85" and k == "19.6"))
ROWS += avail(data="on_request", data_page=12, data_loc="Data availability", data_ev="may be made available by the corresponding author upon reasonable request")
ROWS += [R("limitations", "Evaluated on the validation split rather than a held-out test set, and on two adjacent areas only", 11, "Discussion",
           "Although the comparisons were performed on the validation set rather than a held-out test set"),
         D("notes", "Both models are transformer-based, so the paper gives no transformer versus CNN comparison. Models at 1.7 cm and 3.4 cm were run on downsampled imagery (Table 1); a twofold reduction lowered F1 by 0.04 to 0.08 and a fourfold reduction by 0.15 to 0.24.", "Results and Table 1 (pp. 6 to 7)")]
