import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_h import R, NR, D, avail, model, result, Counter, row  # noqa: E402

PID, T1, T4 = "P121", "Table 1", "Table 4"
ROWS = [
    R("task", "species_classification", 1, "Abstract", "we propose UAV4Tree a Deep Learning based system that automatically classifies RGB optical images obtained by the UAV"),
    R("output_unit", "tree_label", 3, "Sec. 3.2", "The class IDs present in the datasets are as follows: 0 for the Acer opalus class, 1 for the Castanea sativa class, 2 for the Olea europaea class, and 3 for the Quercus pubescens class",
      note="one species label per image of a tree"),
    R("tree_setting", "mixed", 3, "Sec. 3.1", "In the Ascoli Piceno area ... the cultivation of the Olea europaea species is widespread", note="woods and olive cultivation"),
    R("tree_taxa", "Acer opalus; Castanea sativa; Olea europaea; Quercus pubescens", 3, "Sec. 3.2",
      "0 for the Acer opalus class, 1 for the Castanea sativa class, 2 for the Olea europaea class, and 3 for the Quercus pubescens class"),
    R("study_area", "province of Ascoli Piceno, Marche Region, Italy (UAV test images); training images from iNaturalist", 3, "Sec. 3.1",
      "The study area is located in the Marche Region, in the center of Italy, in particular, some species present in the province of Ascoli Piceno"),
    NR("biome", note="central Italy; biome not stated"),
    R("platform", "multiple", 3, "Sec. 3.2", "obtained through the iNaturalist website ... captured from the ground ... The test dataset contains images obtained through frame grabs from films made using UAVs",
      note="ground-level training images, UAV test images"),
    R("sensors", "rgb", 3, "Sec. 3.2", "The RGB images, in JPG format with various resolutions"),
    R("annotation_type", "image_label", 4, "Sec. 3.2", "already accurately annotated by participants in the iNaturalist project"),
    R("dataset_name", "UAV4Tree data: iNaturalist ground images (training, validation) and UAV frames from the Marche Region (test)", 3, "Sec. 3.2",
      "The images were divided as follows: 2690 images in the training dataset (Figure 2), 560 in the validation dataset, and 2650 in the testing dataset", item="dataset1"),
    R("dataset_role", "train_test", 3, "Sec. 3.2", "a significantly different set of images from the training and validation dataset is used", item="dataset1",
      note="test images come from a different platform (UAV) than training images (ground)"),
    D("dataset_public", "no", "No dataset link or availability statement in the paper", item="dataset1"),
    D("n_images", "5900", "2690 training + 560 validation + 2650 test images (p.3, Sec. 3.2)", item="dataset1", unit="images"),
]
FT = dict(pre_page=4, pre_loc="Sec. 3.2", pre_ev="The various models used are pre-trained (Imagenet1K)", strat_page=4, strat_loc="Sec. 3.2",
          strat_ev="afterwards, fine-tuning (running adjustment of drop-out, learning rate, weight decay) where necessary")
M = [("model1", r"Vision Transformer B16", "Vision Transformer B16", "transformer_baseline", "pure_vit", "ViT-B/16"),
     ("model2", r"Densenet161", "DenseNet161", "cnn_baseline", "cnn", "DenseNet-161"),
     ("model3", r"Resnet152", "ResNet152", "cnn_baseline", "cnn", "ResNet-152"),
     ("model4", r"VGG19", "VGG19", "cnn_baseline", "cnn", "VGG-19"),
     ("model5", r"InceptionV3", "InceptionV3", "cnn_baseline", "cnn", "Inception-v3")]
res, tr = Counter("res"), Counter("tr")


def transfer(item, val, test, ev, loc, setting):
    t = tr()
    return [R("transfer_source", "iNaturalist ground-level photographs (training and validation)", 3, "Sec. 3.2",
              "obtained through the iNaturalist website, depicting the specimens of the four species in different phenological states, captured from the ground", item=t),
            R("transfer_target", "UAV video frames from the Marche Region", 4, "Sec. 3.2", "The test dataset contains images obtained through frame grabs from films made using UAVs", item=t),
            R("shift_type", "platform;region", 4, "Sec. 3.2", "These models were used to recognize images captured by UAV, which show trees from a different perspective", item=t),
            R("transfer_model", item, 6, loc, ev, item=t),
            R("transfer_metric", "oa", 6, loc, "Accuracy (validation) ... Accuracy (test)", item=t, note=setting),
            R("value_in_domain", val, 6, loc, ev, item=t, unit="fraction", note="validation accuracy on ground-level images"),
            R("value_out_of_domain", test, 6, loc, ev, item=t, unit="fraction")]


for item, lab, name, role, cat, bb in M:
    nums, ev = row(PID, 6, rf"^\s*{lab}\s+(?=\d)", 0)
    ROWS += model(item, name, role, cat, bb, "imagenet", "full_fine_tuning", 6, T1, ev, cat_note=("best on the augmented data" if item == "model1" else ""), **FT)
    st = "UAV test images (2650); original training set"
    ROWS += result(res(), "oa", item, "dataset1", nums[1], 6, T1, ev, unit="fraction", setting=st)
    ROWS += result(res(), "f1", item, "dataset1", nums[2], 6, T1, ev, unit="fraction", setting=st)
    ROWS += transfer(item, nums[0], nums[1], ev, T1, "original (non-augmented) training set")
T4ROWS = {"model2": row(PID, 6, r"^\s*Densenet161\s+50\s+(?=\d)", 0, cols=(0, 75)),   # epoch column (50) skipped by the regex
          "model1": row(PID, 6, r"^\s+30\s+(?=\d)", 0, cols=(0, 75))}                  # ViT-B16 row: label split over two lines around this one
for item in ("model1", "model2"):
    nums, ev = T4ROWS[item]
    ev = ("Vision Transformer B16 " if item == "model1" else "") + ev
    assert len(nums) == 3, (item, nums, ev)
    st = "UAV test images (2640); augmented training set (63,000 images)"
    ROWS += result(res(), "oa", item, "dataset1", nums[1], 6, T4, ev, unit="fraction", setting=st, headline=(item == "model1"))
    ROWS += result(res(), "f1", item, "dataset1", nums[2], 6, T4, ev, unit="fraction", setting=st)
    ROWS += transfer(item, nums[0], nums[1], ev, T4, "augmented training set")
ROWS += avail()
ROWS += [D("notes", "Models are trained on ground-level iNaturalist photographs and tested on UAV frames, so every test score is cross-platform. Table 1 also gives results on SR-GAN-enhanced test images (ViT-B16 0.75 accuracy); training on UAV and testing on ground images gives 0.48 for ViT-B16 (Table 5).", "Sec. 3 to 5 and Tables 1 to 5 (pp. 3 to 6)")]
