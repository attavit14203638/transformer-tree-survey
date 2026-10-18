import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter  # noqa: E402

ROWS = [
    R("task", "crown_segmentation;species_classification", 5, "Sec. III-A", "each image annotated to identify eight potential categories like: lake, obstacle, ground, track, tree, spruce, birch, and pine",
      note="masks of individual trees (spruce, birch, pine) among other forest classes"),
    R("output_unit", "crown_mask;tree_label", 5, "Fig. 1", "Each color represents separate mask"),
    R("tree_setting", "natural_forest", 5, "Sec. III-A", "It contains diverse forest conditions"),
    R("tree_taxa", "spruce; birch; pine", 5, "Sec. III-A", "lake, obstacle, ground, track, tree, spruce, birch, and pine"),
    NR("study_area", note="FinnForest 3.0 dataset; location not stated in the paper"),
    NR("biome", note="not stated"),
    D("platform", "ground_camera", "Figure 1 (p.5) shows ground-level forest views with vehicle tracks; the text refers to UAV imagery in general terms"),
    R("sensors", "rgb", 5, "Sec. III-A", "There are 300 images in 1280×720 resolution", note="colour images"),
    R("annotation_type", "polygon_mask", 5, "Fig. 1", "Example of the annotated images from the FinnForest 3.0 dataset visualized in CVAT. Each color represents separate mask"),
    R("dataset_name", "FinnForest 3.0", 5, "Sec. III-A", "FinnForest 3.0 is a dataset designed for advanced research in semantic forestry segmentation", item="dataset1"),
    R("dataset_role", "train_test", 5, "Sec. III-A", "split into training (250 images) and validation (50 images) folders", item="dataset1", note="results on the validation images"),
    NR("dataset_public", item="dataset1", note="availability not stated"),
    R("n_images", "300", 5, "Sec. III-A", "There are 300 images in 1280×720 resolution", item="dataset1", unit="images"),
]
ROWS += model("model1", "SAM masks labelled by a fine-tuned YOLOv8 detector (SAM-YOLOv8 pipeline)", "proposed", "foundation_model", "SAM ViT-H with YOLOv8", "sa1b", "zero_shot",
              14, "Sec. IV-F", "SAM was used at the start of the pipeline, followed by the YOLOv8 model trained on the FinnForest dataset as the detector",
              pre_page=12, pre_loc="Sec. IV-E", pre_ev="specifically VIT_B, VIT_L, and VIT_H ... The VIT_H configuration generates the most accurate masks",
              pre_note="SAM weights used as released (SA-1B)", strat_note="SAM not fine-tuned; YOLOv8 trained on FinnForest labels the SAM masks")
ROWS += model("model2", "YOLOv8x-seg", "cnn_baseline", "cnn", "YOLOv8x-seg", None, "full_fine_tuning", 9, "Sec. IV-D", "we used a YOLOv8x-seg model as the baseline",
              pre_status="not_reported", pre_note="not stated", strat_page=8, strat_loc="Table 1 notes", strat_ev="The table metrics are derived from fine-tuning YOLOv8 models on the FinnForest 3.0 dataset")
ROWS += model("model3", "Zero-shot detectors (CLIP, Grounding DINO) with SAM", "transformer_baseline", "vision_language_model", "CLIP; Grounding DINO; SAM", "other", "zero_shot",
              13, "Sec. IV-F", "The first trial involved using Zero-shot detector models (CLIP, GroundingDINO) before introducing SAM", pre_note="released pre-trained weights")
ROWS += model("model4", "SAM masks labelled by BLIP2 image-to-text descriptions", "transformer_baseline", "vision_language_model", "SAM; BLIP2", "other", "zero_shot",
              14, "Sec. IV-F", "SAM was used to generate image segments, which were then converted into text descriptions", pre_note="released pre-trained weights")
res = Counter("res")
ROWS += result(res(), "ap50", "model1", "dataset1", "0.763", 14, "Sec. IV-F", "with an overall mAPmask 50 of 0.763 and mAPmask 50-95 of 0.731 for high-quality images",
               unit="fraction", setting="validation; all classes; masks; 'high-quality images'", headline=True)
ROWS += result(res(), "ap50_95", "model1", "dataset1", "0.731", 14, "Sec. IV-F", "with an overall mAPmask 50 of 0.763 and mAPmask 50-95 of 0.731 for high-quality images",
               unit="fraction", setting="validation; all classes; masks; 'high-quality images'")
ROWS += result(res(), "ap50", "model1", "dataset1", "0.907", 14, "Sec. IV-F", "achieving mAPmask 50 scores of 0.763 for all elements and 0.907 specifically for spruce",
               unit="fraction", setting="validation; spruce class; masks")
ROWS += result(res(), "ap50", "model2", "dataset1", "0.778", 9, "Sec. IV-D and Table 3 (image)", "On this unmodified FinnForest dataset, the model achieved the following performance metrics mAPmask 50: 0.778, mAPmask 50-95: 0.472",
               unit="fraction", setting="validation; all classes; masks; unmodified images", note="Table 1 (p.8) gives mAPmask 50-95 of 0.434 for YOLOv8x-seg in the model-size comparison")
ROWS += result(res(), "ap50_95", "model2", "dataset1", "0.472", 9, "Sec. IV-D and Table 3 (image)", "mAPmask 50: 0.778, mAPmask 50-95: 0.472",
               unit="fraction", setting="validation; all classes; masks; unmodified images")
ROWS += result(res(), "ap50", "model3", "dataset1", "0.549", 13, "Sec. IV-F", "with the overall mAPbox 50 at 0.549 and mAPbox 50-95 at 0.522", unit="fraction", setting="validation; all classes; boxes")
ROWS += result(res(), "ap50_95", "model3", "dataset1", "0.522", 13, "Sec. IV-F", "with the overall mAPbox 50 at 0.549 and mAPbox 50-95 at 0.522", unit="fraction", setting="validation; all classes; boxes")
ROWS += result(res(), "ap50", "model4", "dataset1", "0.752", 14, "Sec. IV-F", "while for spruce trees, mAPmask 50 was only 0.752 and mAPmask 50-95 was 0.394", unit="fraction", setting="validation; spruce class; masks")
ROWS += result(res(), "ap50_95", "model4", "dataset1", "0.394", 14, "Sec. IV-F", "while for spruce trees, mAPmask 50 was only 0.752 and mAPmask 50-95 was 0.394", unit="fraction", setting="validation; spruce class; masks")
ROWS += avail()
ROWS += [D("notes", "The SAM-YOLOv8 values are reported for 'high-quality images', so they are not on the same evaluation set as the YOLOv8x-seg baseline. Tables 1 to 3 are images.", "Sec. IV-D to IV-F (pp. 8 to 14)")]
