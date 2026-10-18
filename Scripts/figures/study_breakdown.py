"""Table 3 of the manuscript: a study-by-study breakdown of a representative subset of the corpus (and, internally, of every
study).

Selection of the subset (to be stated in the caption): studies are added one at a time, each time the one that represents the
most category values not yet in the table (architecture family, task, platform, sensor, biome, tree setting, annotation); ties
go to a study with a transformer-versus-baseline comparison, then to more primary results, then to the more recent study.
Studies are then added until every combination of architecture family with task, and of architecture family with sensor, that
occurs in the corpus is represented.

Writes generated/study_breakdown.tex (Table 3), and figure_data/study_breakdown.csv and study_breakdown_all.csv (every study;
internal, with references to the extraction).
"""
import re

import pandas as pd

import geography
import rq1_comparisons
import rq2_transfer
import taxonomy
from common import METRIC, TASK_GROUP, load, paths, set_style, studies, study_of, study_values, write

FAMILY_ABBR = {"pure_vit": "Pure ViT", "hierarchical_vit": "Hier. ViT", "cnn_transformer_hybrid": "Hybrid", "foundation_model": "FM",
               "vision_language_model": "VLM"}
TASK_ABBR = {"Detection and counting": "Det.", "Crown segmentation": "Crown seg.", "3D structure and attributes": "3D/attr.",
             "Species classification": "Species", "Health and mortality": "Health"}
PLATFORM = {"uav": "UAV", "airborne": "Airborne", "satellite": "Satellite", "ground_camera": "Ground", "mobile_lidar": "MLS",
            "terrestrial_lidar": "TLS", "multiple": "Multiple", "synthetic": "Synthetic"}
SENSOR = {"rgb": "RGB", "multispectral": "MS", "hyperspectral": "HS", "lidar": "LiDAR", "dsm": "DSM"}
ANNOT = {"polygon_mask": "Mask", "bounding_box": "Box", "point_label": "Point labels", "image_label": "Image label",
         "regression_target": "Attribute", "point": "Dot"}
SETTING = {"natural_forest": "Natural", "mixed": "Mixed", "plantation": "Plantation", "urban": "Urban", "orchard": "Orchard",
           "synthetic": "Synthetic", "agroforestry": "Agroforestry", "mangrove": "Mangrove"}
BIOME = {"tropical": "Tropical", "subtropical": "Subtropical", "arid": "Arid", "mediterranean": "Mediterranean", "temperate": "Temperate",
         "boreal": "Boreal", "multiple": "Multiple"}
TRAINING = {"full_fine_tuning": "FT", "peft_lora": "PEFT", "peft_adapter": "PEFT", "peft_prompt": "PEFT", "peft_other": "PEFT",
            "head_only": "Head", "zero_shot": "ZS", "few_shot": "FS", "from_scratch": "Scratch"}
SHIFT_ABBR = {"Site": "Site", "Site + sensor or platform": "Site+acq.", "Region": "Region", "Region + sensor or platform": "Region+acq.",
              "Biome": "Biome", "Sensor, platform or resolution only": "Acq.", "Species only": "Species"}
COUNTRY_ABBR = {"United States of America": "USA", "United Kingdom": "UK", "United Arab Emirates": "UAE"}
DIMENSIONS = ["task", "platform", "sensors", "biome", "tree_setting", "annotation_type"]


def tex(s):
    return re.sub(r"([&%_#$])", r"\\\1", str(s)).replace("R²", "R$^2$")


def joined(values, table, order=None):
    vals = [v for v in values if v not in ("not_reported", "nan", "")]
    vals = sorted(vals, key=lambda v: list(table).index(v) if v in table else 99)
    return ", ".join(table.get(v, v) for v in vals) or "NR"


def short_model(name):
    name = re.sub(r"^Proposed\s+", "", taxonomy.display(name))
    name = name[0].upper() + name[1:]
    if len(name) > 24 and " (" in name:
        name = name.split(" (")[0]
    if len(name) > 28:
        name = re.split(r" on | trained ", name)[0]
    return name


ATTRIBUTE = {"dbh": "DBH", "height": "height", "crown_diameter": "crown diameter", "crown_area": "crown area", "biomass": "biomass",
             "volume": "volume", "count": "count"}


def item_order(df):
    return df.assign(_n=df["item"].str.extract(r"(\d+)", expand=False).astype(float)).sort_values(["paper_id", "_n"])


def metric_label(row):
    if row.metric != "other":
        return METRIC.get(row.metric, row.metric)
    name = re.split(r"[,;:(]", str(row.metric_note))[0].strip()
    name = re.sub(r"^the |\s+row$", "", name)
    return name if 0 < len(name) <= 20 else "study metric"


def headline(r, pids, principal):
    """The result a study marks as its main one; otherwise the first primary result of its principal transformer on the study's
    preferred metric (then any result of that model), in the order the paper reports them."""
    x = item_order(r[r.paper_id.isin(pids) & r.value_std.notna()])
    pick = x[x.headline == True]  # noqa: E712
    if pick.empty:
        m = x[x.model_name == principal]
        pick = m.sort_values(["primary", "canonical_metric"], ascending=False, kind="stable")
    if pick.empty:
        return "", None
    row = pick.iloc[0]
    unit = "" if row.unit_std in ("%", "fraction") or pd.isna(row.unit_std) else f" {row.unit_std}"
    same = str(row.unit) == str(row.unit_std) and pd.notna(row.value)  # no conversion: show the value as printed
    val = str(row.value) if same else (f"{row.value_std:.1f}" if row.unit_std == "%" else f"{row.value_std:.3g}")
    attr = f" ({ATTRIBUTE.get(row.attribute, str(row.attribute).replace('_', ' '))})" if pd.notna(row.attribute) and row.attribute else ""
    return f"{metric_label(row)} {val}{unit}{attr}", f"{row.paper_id}:{row['item']}, p.{row.page}"


def table():
    st = studies()
    tx = taxonomy.assign().set_index("study")
    geo = geography.table()[0].set_index("study")
    vals = {col: study_values(col) for col in DIMENSIONS}
    s = load("studies")
    s = s[s.extracted == True]  # noqa: E712
    s["study"] = s.paper_id.map(study_of())
    code = s.groupby("study").code_available.agg(lambda v: (v == "yes").any())
    r = load("results")
    c = rq1_comparisons.data()[0]
    diff = c.groupby("study").diff_pp.median()
    t = rq2_transfer.data()[0]
    shifts = t.groupby("study").shift_group.agg(lambda g: sorted(set(g), key=list(SHIFT_ABBR).index))
    rows = []
    for x in st.itertuples():
        g = geo.loc[x.study]
        countries = [COUNTRY_ABBR.get(c_, c_) for c_ in str(g.countries).split(";") if c_ and c_ != "nan"]
        area = str(g.study_area).lower()
        place = (", ".join(countries) if len(countries) <= 2 else f"{len(countries)} countries") if countries else \
            ("Global" if "global" in area or "worldwide" in area else "Synthetic" if "virtual" in area or "synthetic" in area else "NR")
        tasks = sorted({TASK_GROUP.get(v, v) for v in vals["task"][x.study]}, key=list(TASK_ABBR).index)
        res, ref = headline(r, x.papers.split(";"), x.principal_model)
        rows.append(dict(
            study=x.study, label=x.label, bib_key=x.bib_key, country=place, biome=joined(vals["biome"][x.study], BIOME),
            setting=joined(vals["tree_setting"][x.study], SETTING), platform=joined(vals["platform"][x.study], PLATFORM),
            sensor=joined(vals["sensors"][x.study], SENSOR), annotation=joined(vals["annotation_type"][x.study], ANNOT),
            task=", ".join(TASK_ABBR[t_] for t_ in tasks), family=FAMILY_ABBR[x.family], model=short_model(x.principal_model),
            base=tx.loc[x.study, "model_group"], training=TRAINING.get(x.principal_strategy, "NR"), result=res, result_ref=ref,
            diff=f"{diff[x.study]:+.1f}" if x.study in diff.index else "", transfer=", ".join(SHIFT_ABBR[k] for k in shifts.get(x.study, [])),
            code="yes" if code.get(x.study, False) else "", year=x.year, family_key=x.family, tasks=tasks,
            sensors=sorted(v for v in vals["sensors"][x.study] if v not in ("not_reported", "nan"))))
    return pd.DataFrame(rows)


def select(df):
    """Greedy coverage of every category value, then of every family-task combination (rule in the module docstring)."""
    dims = {col: study_values(col) for col in DIMENSIONS}
    dims["task"] = {k: {TASK_GROUP.get(v, v) for v in vs} for k, vs in dims["task"].items()}
    cover = {x.study: {("family", x.family_key)} | {(d, v) for d in DIMENSIONS for v in dims[d][x.study] if v not in ("not_reported", "nan")}
             for x in df.itertuples()}
    r = load("results")
    r["study"] = r.paper_id.map(study_of())
    rich = r[r.primary == True].groupby("study").size()  # noqa: E712
    key = {x.study: (x.diff != "", rich.get(x.study, 0), x.year) for x in df.itertuples()}
    universe = set().union(*cover.values())
    chosen, covered = [], set()
    while covered != universe:
        best = max(cover, key=lambda s: (len(cover[s] - covered),) + key[s])
        chosen.append(best)
        covered |= cover[best]
    by = df.set_index("study")
    for part in ("tasks", "sensors"):  # every family-task, then every family-sensor combination of the corpus
        cells = {(x.family_key, v) for x in df.itertuples() for v in getattr(x, part)}
        have = {(by.loc[s_, "family_key"], v) for s_ in chosen for v in by.loc[s_, part]}
        for cell in sorted(cells - have):
            if cell in have:
                continue
            cands = df[(df.family_key == cell[0]) & df[part].map(lambda vs: cell[1] in vs)]
            best = max(cands.study, key=lambda s_: key[s_])
            chosen.append(best)
            have |= {(cell[0], v) for v in by.loc[best, part]}
    out = df[df.study.isin(chosen)].copy()
    out["family_order"] = out.family_key.map({k: i for i, k in enumerate(FAMILY_ABBR)})
    return out.sort_values(["family_order", "year", "label"])


COLS = [  # (header, column type, content); the difference from the baseline and the transfer tests stay in the figure data
    ("Study", "L{2.3cm}", lambda x: f"\\citet{{{x.bib_key}}}"),
    ("Study area (country; biome; setting)", "L{2.9cm}", lambda x: "; ".join([x.country, low(x.biome), low(x.setting)])),
    ("Data (platform; sensor; annotation)", "L{2.9cm}", lambda x: "; ".join([x.platform, x.sensor, low(x.annotation)])),
    ("Task", "L{1.5cm}", lambda x: x.task),
    ("Family: principal model", "Y", lambda x: f"{x.family}: {x.model}"),
    ("Training", "L{1.15cm}", lambda x: x.training),
    ("Main result", "L{2.0cm}", lambda x: x.result),
    ("Code", "c", lambda x: "\\checkmark" if x.code else ""),
]
RAW = {"Study", "Code"}  # cells written as LaTeX, not escaped


def low(s):
    return ", ".join(v if v == "NR" else v.lower() for v in s.split(", "))


def body(df):
    lines = []
    for x in df.itertuples():
        cells = []
        for head, _, get in COLS:
            v = get(x)
            cells.append(v if head in RAW else tex(v) if v else "")
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines)


def tabular(df, env):
    head = " & ".join(h for h, _, _ in COLS) + " \\\\"
    spec = "@{}" + "".join(t for _, t, _ in COLS) + "@{}"
    spec = re.sub(r"L\{([^}]*)\}", r">{\\raggedright\\arraybackslash}p{\1}", spec).replace("Y", ">{\\raggedright\\arraybackslash}X")
    if env == "tabularx":
        return f"\\begin{{tabularx}}{{\\textwidth}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n{body(df)}\n\\bottomrule\n\\end{{tabularx}}%\n"  # % : no stray space at the end of the input file
    spec = spec.replace("}X", "}p{3.4cm}")
    return (f"\\begin{{longtable}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n\\endhead\n{body(df)}\n\\bottomrule\n"
            "\\end{longtable}\n")


NOTE = ("% Generated by Scripts/figures/study_breakdown.py; do not edit by hand. Abbreviations for the caption: Hier. ViT, hierarchical\n"
        "% ViT; FM, foundation model; VLM, vision-language model; MS, multispectral; HS, hyperspectral; MLS and TLS, mobile and\n"
        "% terrestrial laser scanning; FT, full fine-tuning; PEFT, parameter-efficient tuning; ZS and FS, zero- and few-shot; Head,\n"
        "% frozen backbone; NR, not reported. Main result: bounded metrics in percent.\n")


def main():
    df = table()
    sub = select(df)
    keep = ["label", "bib_key", "study", "country", "biome", "setting", "platform", "sensor", "annotation", "task", "family", "model", "base",
            "training", "result", "result_ref", "diff", "transfer", "code"]
    write(sub[keep], "study_breakdown")
    full = df.assign(family_order=df.family_key.map({k: i for i, k in enumerate(FAMILY_ABBR)})).sort_values(["family_order", "year", "label"])
    write(full[keep], "study_breakdown_all")
    paths.PAPER_GENERATED.mkdir(parents=True, exist_ok=True)
    metrics = sorted({r_.split(" ")[0] for r_ in sub.result if r_})
    note = NOTE + f"% Metrics in the main table: {', '.join(metrics)}; DBH, diameter at breast height.\n"
    (paths.PAPER_GENERATED / "study_breakdown.tex").write_text(note + tabular(sub, "tabularx"), encoding="utf-8")
    print(f"  {len(sub)} studies in the breakdown table (generated/study_breakdown.tex); all {len(df)} in figure_data/study_breakdown_all.csv")


if __name__ == "__main__":
    set_style()
    main()
