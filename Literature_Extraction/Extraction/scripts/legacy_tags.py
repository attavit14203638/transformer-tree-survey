"""Protocol section 6 tags for the papers of B01 to B03, which were extracted before the tags existed.

`Scripts/extraction/kit/append.py` calls `apply(pid, rows)` after running a B01 to B03 paper script, so re-running that
script always gives the paper its tags back. Papers of B04 onward set their tags in their own scripts.

Per paper:
  default  task, level, split, agg (and attribute) for every result
  res      [(match, tags)]: the first rule whose match fits a result overrides the defaults
  primary  [match]: a result is primary when any match fits (approximate values and metric 'other' never are)
  tr       transfer level; 'note' takes it from the transfer_metric note (box -> object, mask -> instance_mask)
  ds       dataset item -> dataset_key (datasets.csv)
  copied   result items copied from another paper -> source (rule V9)
  cf       conflict items (rule V1, V2): target, alternative, unit, page, location, evidence, rule, reason

match keys: set (substring of result_setting), set_eq (exact), loc (substring of the value's location), metric, items,
model (sets), ds (dataset item). All given keys must fit; an empty match fits every result.
"""

ALL = {}


def _p(pid, task, level, split="test", agg="single", attribute=None, res=(), primary=(ALL,), tr=None, ds=None, copied=None, cf=()):
    SPEC[pid] = dict(default=dict(task=task, level=level, split=split, agg=agg, attribute=attribute), res=list(res),
                     primary=list(primary), tr=tr, ds=ds or {}, copied=copied or {}, cf=list(cf))


SPEC = {}
MASK = ({"set": "(mask)"}, dict(task="crown_segmentation", level="instance_mask"))
TRUNK_MASK = ({"set": "(mask)"}, dict(task="component_segmentation", level="instance_mask"))

# ---------------------------------------------------------------------------------------------------------------- B01
_p("P001", "detection", "object", res=[MASK], ds={"dataset1": "own_p001_ghaf"})
_p("P002", "detection", "object", res=[({"metric": {"r2"}}, dict(task="counting", level="image"))],
   primary=[{"set": "ID-zone test"}, {"set": "ID test"}], tr="object", ds={"dataset1": "rajasthan_agroforestry_shift"})
_p("P010", "counting", "image", agg="mean", res=[({"metric": {"f1"}}, dict(task="detection", level="object"))],
   primary=[{"ds": "dataset1"}], ds={"dataset1": "denmark_aerial_trees", "dataset2": "rwanda_aerial_trees"})
_p("P011", "detection", "object", res=[MASK], tr="note", ds={"dataset1": "own_p011_ajman_uav", "dataset2": "al_ain_date_palm"})
_p("P012", "detection", "object", split="validation", primary=[{"set_eq": "validation split; 0.85 cm imagery; kernel 19.6 cm"}],
   ds={"dataset1": "own_p012_mangrove"})
_p("P014", "counting", "image", primary=[{"ds": "dataset1"}], ds={"dataset1": "kcl_london", "dataset2": "yosemite_trees"},
   copied={f"res{i}": "P030" for i in range(1, 31)})
_p("P015", "detection", "object", primary=[{"loc": "Table 5"}], ds={"dataset1": "own_p015_liriodendron"},
   cf=[("dataset1.n_trees", "7824", "trees", 4, "Sec. 2", "a total of 1239 images including 7824 individual tree samples were obtained",
        "V1c", "The per-split counts printed in the same sentence (5548, 686 and 878) add up to 7112, which is recorded")])
_p("P016", "detection", "object", res=[({"ds": "dataset2"}, dict(split="not_stated"))], primary=[{"ds": "dataset1"}],
   ds={"dataset1": "campo_grande_urban_trees", "dataset2": "own_p016_guangzhou"})
_p("P017", "detection", "object", split="not_stated", ds={"dataset1": "own_p017_bulianta"},
   cf=[("gsd", "5", "m", 4, "Table 1", "Average ground sampling distance 5m", "V1c",
        "Table 1 also prints 'Spatial resolution 0.01 m', which matches the UAV flights and is recorded")])
_p("P018", "detection", "object", res=[MASK, ({"items": {"res41"}}, dict(task="health_classification", level="tree"))], tr="note",
   ds={"dataset1": "own_p018_ajman_uav", "dataset2": "al_ain_date_palm", "dataset3": "own_p018_date_palm_health"})
_p("P019", "detection", "object", res=[TRUNK_MASK], primary=[{"ds": "dataset2"}], tr="object",
   ds={"dataset1": "canatree100", "dataset2": "flickrtree100", "dataset3": "flickrtree110k"})
_p("P020", "detection", "object", res=[TRUNK_MASK], ds={"dataset1": "m2fnet_virtual_forest", "dataset2": "canatree100"})
_p("P021", "detection", "object", split="validation", primary=[{"set": "six bands"}], ds={"dataset1": "own_p021_huangkeng"})
_p("P022", "detection", "object", primary=[{"ds": "dataset1", "loc": "Table 1"}], tr="object",
   ds={"dataset1": "own_p022_dataset_a", "dataset2": "own_p022_dataset_b"},
   cf=[("res31.value", "45.31", "%", 12, "Sec. 3", "On Dataset_B, MTCDNet achieved mAP, AP50, and AP75 scores of 45.31%, 93.22%, and 31.79%",
        "V1a", "Table 2 gives 48.41 for MTCDNet; the values in the text are those of Early Fusion in Table 2"),
       ("res32.value", "93.22", "%", 12, "Sec. 3", "On Dataset_B, MTCDNet achieved mAP, AP50, and AP75 scores of 45.31%, 93.22%, and 31.79%",
        "V1a", "Table 2 gives 94.58 for MTCDNet; the values in the text are those of Early Fusion in Table 2")])
_p("P024", "detection", "object", agg="mean", primary=[{"set": "test plot #2"}], ds={"dataset1": "own_p024_kroof_uav"})
_p("P025", "detection", "object", ds={"dataset1": "quebec_trees_sbl_2021"})
_p("P027", "health_classification", "object", ds={"dataset1": "own_p027_longyou"},
   cf=[("res2.value", "89.5", "%", 11, "Sec. 4.6", "but still much higher than the 89.5% of the original model", "V1a",
        "Table 2, the main comparison, gives 91.5 for RT-DETR; 89.5 is the baseline of the ablation study")])
_p("P030", "counting", "image", primary=[{"ds": "dataset1"}],
   ds={"dataset1": "kcl_london", "dataset2": "jiangsu_gaofen2", "dataset3": "yosemite_trees"})
_p("P032", "crown_segmentation", "instance_mask", res=[({"set": "9 species classes"}, dict(task="species_classification"))],
   primary=[{"set": "single class 'tree'"}], ds={"dataset1": "quebec_plantations"})

# ---------------------------------------------------------------------------------------------------------------- B02
_p("P033", "crown_segmentation", "pixel", primary=[{"set": "BARE configuration"}], ds={"dataset1": "oam_tcd"})
_p("P035", "crown_segmentation", "instance_mask", res=[({"set": "multi-class"}, dict(task="species_classification"))],
   primary=[{"ds": "dataset1", "set": "single class"}],
   ds={"dataset1": "quebec_plantations", "dataset2": "quebec_trees_sbl_2021", "dataset3": "bci_crowns_2022"})
_p("P040", "crown_segmentation", "pixel", ds={"dataset1": "own_p040_acacia"})
_p("P042", "species_classification", "instance_mask", split="validation", res=[({"set": "boxes"}, dict(level="object"))], primary=[],
   ds={"dataset1": "finnforest3"},
   cf=[("res5.value", "0.434", "fraction", 8, "Table 1 (image)", "YOLOv8x-seg mAPmask 50-95 0.434", "V1a",
        "Sec. IV-D and Table 3 give 0.472 for the model carried forward; Table 1 is the model-size comparison")])
_p("P044", "crown_segmentation", "instance_mask", tr="instance_mask", ds={"dataset1": "isprs_itc_contest"})
_p("P051", "crown_segmentation", "pixel", primary=[], tr="pixel",
   ds={"dataset1": "own_p051_ajman_uav", "dataset2": "ajman_date_palm_multisource", "dataset3": "al_ain_date_palm"})
_p("P056", "crown_segmentation", "pixel", primary=[{"loc": "Table 3", "set_eq": "test set; input: eight multispectral bands"}], tr="pixel",
   ds={"dataset1": "own_p056_worldview3", "dataset2": "own_p056_dibba"})
_p("P057", "crown_segmentation", "instance_mask", primary=[{"items": {"res3"}}], ds={"dataset1": "own_p057_kroof_uav"})
_p("P058", "species_classification", "instance_mask", split="validation", ds={"dataset1": "tree_species_is_p058"})
_p("P060", "crown_segmentation", "instance_mask", ds={"dataset1": "own_p060_black_forest"})
_p("P062", "detection", "object", primary=[{"ds": "dataset2"}], tr="object",
   ds={"dataset1": "neon_treeevaluation", "dataset2": "detectree2_malaysia"})
_p("P063", "component_segmentation", "point", primary=[{"loc": "Table 5"}], ds={"dataset1": "forestsemantic_ms"})
_p("P072", "attribute_estimation", "tree", attribute="dbh", primary=[{"set": "sample plot 1"}], ds={"dataset1": "own_p072_beijing_plots"})
_p("P074", "component_segmentation", "pixel", split="validation",
   res=[({"items": {"res7"}}, dict(task="attribute_estimation", level="tree", attribute="stem_volume"))],
   primary=[{"loc": "Sec. IV-A"}], ds={"dataset1": "own_p074_beijing_trunks"})
_p("P076", "species_classification", "tree", ds={"dataset1": "usgs_lidar_osm_8cities", "dataset2": "gro_single_tree_lidar"})
_p("P078", "tree_segmentation_3d", "instance_3d", res=[({"metric": {"miou"}}, dict(task="component_segmentation", level="point"))],
   primary=[{"ds": "dataset1"}], tr="instance_3d",
   ds={"dataset1": "for_instance_v2", "dataset2": "wytham_woods_tls", "dataset3": "lautx"})
_p("P082", "detection", "object", res=[({"set": "rate of correct segmentation"}, dict(task="crown_segmentation", level="instance_mask"))],
   primary=[], ds={"dataset1": "own_p082_payette"})
_p("P083", "detection", "instance_3d", split="not_stated",
   res=[({"metric": {"r2", "rmse"}}, dict(task="attribute_estimation", level="tree", attribute="trunk_radius"))],
   primary=[{"items": {"res1", "res2", "res3", "res4"}}], ds={"dataset1": "own_p083_danzhou_rubber"})

# ---------------------------------------------------------------------------------------------------------------- B03
_p("P090", "crown_segmentation", "pixel", res=[({"loc": "Table 5"}, dict(task="tree_segmentation_3d", level="instance_3d"))],
   primary=[{"loc": "Table 3", "set": "Forest 1"}, {"loc": "Table 2", "set": "Forest 1", "model": {"model1"}}],
   ds={"dataset1": "own_p090_ganzhou"},
   cf=[("dataset1.n_images", "7239", "images", 6, "Sec. 2.3", "a total of 7239 image patches were generated, covering 5824 distinct tree instances",
        "V1c", "The per-split counts printed next to it (5548, 686 and 878) add up to 7112, which is recorded")])
_p("P091", "component_segmentation", "pixel",
   res=[({"set": "stem volume"}, dict(task="attribute_estimation", level="tree", attribute="stem_volume")),
        ({"set": "tree height"}, dict(task="attribute_estimation", level="tree", attribute="height"))],
   primary=[{"items": {"res7", "res8", "res9", "res17"}}], ds={"dataset1": "own_p091_beijing_trunks"})
_p("P101", "species_classification", "tree",
   res=[({"loc": "Table 5"}, dict(task="tree_segmentation_3d", level="instance_3d")), ({"loc": "Table 6"}, dict(level="instance_3d"))],
   primary=[{"loc": "Table 4"}, {"loc": "Table 5", "ds": "dataset1"}], tr="instance_3d",
   ds={"dataset1": "whu_stree_nj", "dataset2": "whu_stree_sy"})
_p("P105", "species_classification", "tree", ds={"dataset1": "own_p105_nefu_uls"})
_p("P106", "species_classification", "tree", split="validation", ds={"dataset1": "own_p106_braunschweig"})
_p("P107", "species_classification", "tree", agg="mean", primary=[{"ds": "dataset1"}], ds={"dataset1": "own_p107_umtpc", "dataset2": "stpctls"})
_p("P111", "species_classification", "tree", ds={"dataset1": "own_p111_romeo_malette"})
_p("P112", "species_classification", "object", primary=[{"set": "test set of the multi-source dataset"}], ds={"dataset1": "own_p112_multisource"})
_p("P113", "species_classification", "tree", split="validation", ds={"dataset1": "own_p113_langya"})
_p("P116", "species_classification", "tree", agg="mean", res=[({"loc": "Table 3"}, dict(agg="best"))], primary=[{"loc": "Table 2"}], tr="tree",
   ds={"dataset1": "own_p116_zao", "dataset2": "own_p116_zao"},
   cf=[("Table 5 average validation F1", "94.99", "%", 17, "Table 5", "F1 92.02 85.95 97.01 94.99", "V2",
        "The printed average is not the mean of the three folds (91.66); the three folds are recorded as transfer pairs")])
_p("P118", "species_classification", "tree", primary=[{"set": "summer test images"}], tr="tree", ds={"dataset1": "own_p118_martell"})
_p("P121", "species_classification", "tree", primary=[{"loc": "Table 1"}], tr="tree", ds={"dataset1": "own_p121_uav4tree"})
_p("P122", "species_classification", "not_stated", primary=[{"set": "visible, spectral and LiDAR"}], ds={"dataset1": "own_p122_hunan_ohtl"})
_p("P127", "health_classification", "object", primary=[{"loc": "Table 3"}, {"loc": "Table 4", "model": {"model1", "model3"}}],
   ds={"dataset1": "own_p127_huangshan_pwd"})
_p("P128", "crown_segmentation", "instance_mask", tr="instance_mask", ds={"dataset1": "own_p128_nfu_poplar", "dataset2": "own_p128_site2"})
_p("P132", "health_classification", "object", split="validation", primary=[{"loc": "Table 6"}], ds={"dataset1": "own_p132_qingdao"})
_p("P135", "mortality_detection", "instance_mask", agg="mean", res=[({"set": "pixel IoU"}, dict(level="pixel"))],
   ds={"dataset1": "treemort_nls_finland"},
   cf=[("gsd", "0.5", "m", 5, "Sec. 3.4", "were processed into orthophotos at a spatial resolution of 0.5 m", "V1c",
        "Table 1 (p.6) gives 0.25 m/pixel and Sec. 3.4 (p.6) describes the annotated images as 25 cm, which is recorded")])
_p("P139", "health_classification", "object", ds={"dataset1": "own_p139_huangshan_pwd"})
_p("P148", "health_classification", "object", agg="not_stated", primary=[{"loc": "Table 2"}], ds={"dataset1": "own_p148_boluo_citrus"},
   cf=[("model1.total_params", "29.0", "M", 11, "Table 2", "Swin-T YOLOX(proposed) 29.0 M 93.3% 73.6%", "V1b",
        "Table 3 gives 45.89 M for the unpruned model together with its 209.70 GFLOPs; 29.0 M matches the Swin-T backbone alone")])

TAG_FIELDS = ("result_task", "attribute", "level", "split", "aggregation", "primary", "copied_from", "dataset_key", "transfer_level")


def _fits(match, f, loc):
    g = lambda k: f.get(k, ("",))[0]  # noqa: E731
    if "set" in match and match["set"] not in g("result_setting"):
        return False
    if "set_eq" in match and match["set_eq"] != g("result_setting"):
        return False
    if "loc" in match and match["loc"] not in loc:
        return False
    if "metric" in match and g("metric") not in match["metric"]:
        return False
    if "items" in match and f["_item"] not in match["items"]:
        return False
    if "model" in match and g("result_model") not in match["model"]:
        return False
    if "ds" in match and g("result_dataset") != match["ds"]:
        return False
    return True


def apply(pid, rows):
    """rows: (item, field, value, unit, page, location, evidence, status, note) of one paper, without tags."""
    spec = SPEC.get(pid)
    if spec is None:
        return rows
    rows = [r for r in rows if r[1] not in TAG_FIELDS and not r[0].startswith("cf")]
    items = {}
    for r in rows:
        if r[0]:
            items.setdefault(r[0], {})[r[1]] = (r[2], r)
    last = {r[0]: i for i, r in enumerate(rows) if r[0]}
    extra = {}
    for item, f in items.items():
        f["_item"] = item
        add = []
        if item.startswith("res"):
            v = f["value"][1]
            src = (v[4], v[5], v[6])
            t = dict(spec["default"])
            for match, over in spec["res"]:
                if _fits(match, f, v[5]):
                    t.update(over)
                    break
            prim = (v[7] != "approximate" and f["metric"][0] != "other"
                    and any(_fits(m, f, v[5]) for m in spec["primary"]))
            add.append(("result_task", t["task"]))
            if t["task"] == "attribute_estimation":
                add.append(("attribute", t["attribute"]))
            add += [("level", t["level"]), ("split", t["split"]), ("aggregation", t["agg"])]
            if prim:
                add.append(("primary", "yes"))
            if item in spec["copied"]:
                add.append(("copied_from", spec["copied"][item]))
            extra[last[item]] = [(item, k, val, "", *src, "reported", "") for k, val in add]
        elif item.startswith("dataset"):
            key = spec["ds"][item]
            n = f["dataset_name"][1]
            if n[7] == "reported":
                extra[last[item]] = [(item, "dataset_key", key, "", n[4], n[5], n[6], "reported", "")]
            else:
                extra[last[item]] = [(item, "dataset_key", key, "", "", "", "", "derived", "key from datasets.csv (rule V9)")]
        elif item.startswith("tr"):
            m = f["transfer_metric"][1]
            lvl = spec["tr"]
            if lvl == "note":
                lvl = "instance_mask" if "mask" in (m[8] or "") else "object"
            extra[last[item]] = [(item, "transfer_level", lvl, "", m[4], m[5], m[6], "reported", "")]
    out = []
    for i, r in enumerate(rows):
        out.append(r)
        out += extra.get(i, [])
    for n, (target, alt, unit, page, loc, ev, rule, reason) in enumerate(spec["cf"], start=1):
        it = f"cf{n}"
        out += [(it, "cf_target", target, "", page, loc, ev, "reported", ""),
                (it, "cf_alternative", alt, unit, page, loc, ev, "reported", ""),
                (it, "cf_rule", rule, "", page, loc, ev, "reported", reason)]
    return out
