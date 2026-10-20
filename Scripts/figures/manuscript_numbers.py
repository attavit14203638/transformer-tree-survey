"""Numbers quoted in the manuscript text, computed with the same data and code as the figures.

  Scripts/.venv/bin/python Scripts/figures/manuscript_numbers.py      (after compile.py; run make_figures.py too when figures change)

Writes generated/numbers.tex in the manuscript folder (paths.PAPER_GENERATED), one LaTeX macro per number (for example \\NStudies, \\PctNorthern), and
Literature_Extraction/Analysis/figure_data/numbers.csv (macro, value, text, definition, source). The manuscript uses the macros,
so no number is typed by hand. Percentages have one decimal unless stated; differences and changes are in percentage points and
signed in math mode. Supporting lists: figure_data/shared_public_datasets.csv and figure_data/peft_same_setting.csv.
"""
import csv
import json
import re
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import geography
import overview
import rq1_comparisons
import rq2_transfer
import rq3_training
import taxonomy
from common import FAMILIES, FAMILY, TIE_PP, TUKEY_K, bib_key, load, outcome, paths, per_study, studies, study_of, write

sys.path.insert(0, str(paths.SCRIPTS / "search"))
import query_terms  # noqa: E402

CONFIDENCE = 0.95  # of the upper bound on the extraction error rate (crosscheck.py uses 1 - 0.05^(1/n))

ROWS = []
CLAIMS = []  # statements the text makes in words, checked against the data (description, holds)
FAM_KEY = {"pure_vit": "PureViT", "hierarchical_vit": "HierViT", "cnn_transformer_hybrid": "Hybrid", "foundation_model": "Foundation",
           "vision_language_model": "VLM", "non_transformer": "NonTransformer"}
ROMAN = {2022: "MMXXII", 2023: "MMXXIII", 2024: "MMXXIV", 2025: "MMXXV"}


def camel(label):
    label = str(label).replace("3D", "Three D ").replace("+", " plus ").replace("R²", "R squared")
    return "".join(w[:1].upper() + w[1:] for w in re.findall(r"[A-Za-z]+", label))


def put(macro, value, text, definition, source):
    assert re.fullmatch(r"[A-Za-z]+", macro), macro
    assert macro not in {r["macro"] for r in ROWS}, macro
    ROWS.append(dict(macro=macro, value=value, text=text, definition=definition, source=source))


WORDS = "zero one two three four five six seven eight nine ten".split()


def count(macro, v, definition, source):
    put(macro, int(v), f"{int(v):,}".replace(",", "{,}"), definition, source)
    if 0 <= int(v) <= 10:  # the same count in words, for prose such as "only one study" (\<macro>Word)
        put(macro + "Word", int(v), WORDS[int(v)], definition + " (in words)", source)


def percent(macro, a, b, definition, source, digits=1):
    v = round(100 * a / b, digits) if b else 0.0
    put(macro, v, f"{v:.{digits}f}", definition + f" ({a} of {b})", source)


def signed(macro, v, definition, source):
    v = round(float(v), 1)
    put(macro, v, f"${v:+.1f}$".replace("+-", "-"), definition, source)


def plain(macro, v, text, definition, source):
    put(macro, v, text, definition, source)


def corpus():
    src = "Literature_Extraction/Search/screening/prisma_counts.json"
    c = json.loads((paths.SEARCH / "screening" / "prisma_counts.json").read_text())
    idf, scr, om, inc = c["identification"], c["screening"], c["other_methods"], c["included"]
    count("NRecords", idf["records_total"], "records identified from the databases", src)
    for k, v in idf["databases"].items():
        count("N" + camel(k), v, f"records from {k}", src)
    count("NDuplicates", idf["duplicates_removed"], "duplicate records removed before screening", src)
    count("NScreened", scr["records_screened"], "records screened on title and abstract", src)
    count("NExcludedScreening", scr["records_excluded"], "records excluded on title and abstract", src)
    count("NSought", scr["reports_sought"], "database reports sought for retrieval", src)
    count("NNotRetrieved", scr["reports_not_retrieved"], "database reports not retrieved", src)
    count("NAssessed", scr["reports_assessed"], "database reports assessed in full text", src)
    count("NExcludedFullText", sum(scr["reports_excluded"].values()), "database reports excluded in full text", src)
    count("NOtherRecords", sum(om["records_by_method"].values()), "records identified by other methods", src)
    count("NOtherAssessed", om["reports_assessed"], "other-method reports assessed in full text", src)
    count("NOtherExcluded", sum(om["reports_excluded"].values()), "other-method reports excluded in full text", src)
    reasons = pd.Series(scr["reports_excluded"]).add(pd.Series(om["reports_excluded"]), fill_value=0)
    short = {"Not a tree task": "NotTree", "Area-level mapping, not individual trees": "AreaLevel",
             "No transformer-based model trained or evaluated": "NoTransformer", "Not remote sensing or close-range tree imagery": "NotImagery",
             "Outside date range": "DateRange", "No individual-tree output evaluated": "NoTreeOutput", "No quantitative evaluation": "NoEvaluation"}
    for k, v in reasons.items():
        count("NExcl" + short[k], v, f"full-text exclusions, both routes: {k}", src)
    count("NReports", inc["reports"], "included reports", src)
    count("NStudies", inc["studies"], "included studies (reports of one study count once)", src)
    count("NReportsDatabases", inc["via_databases"], "included reports found through the databases", src)
    count("NReportsOther", inc["via_other_methods"], "included reports found through other methods", src)
    a = json.loads((paths.SEARCH / "screening" / "screening_agreement.json").read_text())
    src = "Literature_Extraction/Search/screening/screening_agreement.json"
    count("NDoubleScreened", a["n_double_screened"], "records screened independently by the author (blind sample)", src)
    plain("PctAgreement", a["percent_agreement"], f"{a['percent_agreement']:.1f}", "agreement between the two screenings (percent)", src)
    plain("KappaScreening", a["cohen_kappa"], f"{a['cohen_kappa']:.2f}", "Cohen's kappa of the two screenings", src)
    count("NScreeningDisagreements", len(a["disagreements"]), "records of the blind sample on which the two screenings disagreed", src)
    percent("PctDoubleScreened", a["n_double_screened"], c["screening"]["records_screened"],
            "share of the screened records in the author's blind sample", src)
    x = json.loads((paths.EXTRACTION / "crosscheck" / "crosscheck_result.json").read_text())
    src = "Literature_Extraction/Extraction/crosscheck/crosscheck_result.json"
    cur = x["current_corpus"]
    count("NCrossChecked", x["checked"], "extracted values checked by the author against the papers (random sample)", src)
    count("NCrossCheckedCorrect", x["ok"], "sampled values found correct", src)
    count("NCrossCheckedFinal", cur["rows"], "sampled values that belong to the final corpus", src)
    assert abs(cur["upper_95_error_rate"] - round(1 - (1 - CONFIDENCE) ** (1 / cur["rows"]), 4)) < 1e-9, "bound and CONFIDENCE disagree"
    plain("PctErrorBound", round(100 * cur["upper_95_error_rate"], 1), f"{100 * cur['upper_95_error_rate']:.1f}",
          "one-sided 95% upper bound of the error rate, final corpus, 1 - 0.05^(1/n), in percent", src)
    src = "Literature_Extraction/Extraction/compiled/"
    r, m = load("results"), load("models")
    count("NResults", len(r), "extracted results (values with model, dataset, metric and page)", src + "results.csv")
    count("NPrimaryResults", int(r.primary.sum()), "results on the main test setting of each paper", src + "results.csv")
    count("NModels", len(m), "extracted models", src + "models.csv")
    count("NTransformerModels", int(m.transformer.sum()), "extracted transformer models", src + "models.csv")
    count("NValueChoices", len(load("conflicts")), "recorded choices between two values a paper prints for one result", src + "conflicts.csv")
    count("NComparisonsAll", len(load("comparisons")), "transformer versus non-transformer comparisons recorded, all metrics", src + "comparisons.csv")
    count("NTransferPairsAll", len(load("transfer")), "transfer pairs recorded, complete or not, all metrics", src + "transfer.csv")
    t = m[m.transformer == True]  # noqa: E712
    sized = int((t.total_params_M.notna() | t.trainable_params_M.notna() | t.trainable_params_pct.notna()).sum())
    count("NTransformerModelsSized", sized, "transformer models with a reported parameter count (total or trainable)", src + "models.csv")
    percent("PctTransformerModelsSized", sized, len(t), "share of transformer models with a reported parameter count", src + "models.csv")


def corpus_shape():
    st = studies()
    n = len(st)
    src = "figure_data/studies.csv (common.studies)"
    years = st.year.value_counts().sort_index()
    for y, k in ROMAN.items():
        count("NStudies" + k, years.get(y, 0), f"studies published in {y}", src)
    for y0, y1 in zip(list(ROMAN)[:-1], list(ROMAN)[1:]):
        percent("PctGrowth" + ROMAN[y1], years[y1] - years[y0], years[y0], f"increase in studies from {y0} to {y1}, percent", src, 0)
    percent("PctGrowthTotal", years[2025] - years[2022], years[2022], "increase in studies from 2022 to 2025, percent", src, 0)
    percent("PctStudiesMMXXV", years[2025], n, "share of studies published in 2025", src)
    for k in FAMILIES:
        v = int((st.family == k).sum())
        count("NFam" + FAM_KEY[k], v, f"studies whose architecture family is {FAMILY[k]}", src)
        percent("PctFam" + FAM_KEY[k], v, n, f"share of studies in {FAMILY[k]}", src)
        for y, ky in ROMAN.items():  # yearly counts per family, as drawn in the timeline
            count("NFam" + FAM_KEY[k] + "In" + ky, int(((st.family == k) & (st.year == y)).sum()),
                  f"studies of family {FAMILY[k]} published in {y}", src)
    for basis, key in (("proposed transformer", "Proposed"), ("best-scoring transformer (baseline only)", "Baseline"),
                       ("transformer used, no comparable score", "Used")):
        v = int((st.basis == basis).sum())
        count("NBasis" + key, v, f"studies whose family comes from: {basis}", src)
        percent("PctBasis" + key, v, n, f"share of studies whose family comes from: {basis}", src)
    tx = taxonomy.assign()
    src = "figure_data/taxonomy.csv (taxonomy.assign)"
    for node, key in ((taxonomy.SPECIFIC, "StudySpecific"), (taxonomy.OTHER_NAMED, "OtherNamed")):
        v = int((tx.node == node).sum())
        count("N" + key, v, f"studies in the residual group '{node}'", src)
        percent("Pct" + key, v, n, f"share of studies in '{node}'", src)
        for k in FAMILIES:  # composition of the group by architecture family, given in the text
            vk = int(((tx.node == node) & (tx.family == k)).sum())
            if vk:
                count("N" + key + FAM_KEY[k], vk, f"studies in '{node}' whose family is {FAMILY[k]}", src)
    v = int(tx.node.isin([taxonomy.SPECIFIC, taxonomy.OTHER_NAMED]).sum())
    count("NResidual", v, "studies in the two residual groups together", src)
    percent("PctResidual", v, n, "share of studies in the two residual groups", src)
    groups = {"SAM": ["SAM", "SAM 2"], "Swin": ["Swin"], "SegFormer": ["SegFormer"], "PointTransformer": ["Point transformer"],
              "DETR": ["DETR family", "RT-DETR", "Deformable DETR"], "YOLO": ["YOLO-based"], "MaskFormer": ["Mask2Former family"], "ViT": ["ViT"]}
    for key, gs in groups.items():
        count("NGroup" + key, int(tx.model_group.isin(gs).sum()), f"studies whose principal transformer is in: {', '.join(gs)}", src)
    count("NGroupPointTransformerPureViT", int(((tx.model_group == "Point transformer") & (tx.family == "pure_vit")).sum()),
          "pure-ViT studies whose principal transformer is a point transformer", src)
    tab, n = overview.counts()
    prefix = {"task": "Task", "output_unit": "Output", "annotation_type": "Annot", "platform": "Platform", "sensors": "Sensor", "tree_setting": "Setting"}
    for row in tab.itertuples():
        count("N" + prefix[row.field] + camel(row.value), row.studies, f"studies with {row.field} = {row.value}", "figure_data/overview.csv")
        percent("Pct" + prefix[row.field] + camel(row.value), row.studies, n, f"share of studies with {row.field} = {row.value}", "figure_data/overview.csv")
    ev = overview.evidence()
    src = "figure_data/overview_evidence_map.csv (overview.evidence)"
    for p in overview.PLATFORMS:
        for q in overview.SENSORS:
            v = int(ev.acquisition.map(lambda vs: (p, q) in vs).sum())
            if v:
                count("NAcq" + camel(p) + camel(q), v, f"studies with platform {p} and sensor {q}", src)
                percent("PctAcq" + camel(p) + camel(q), v, n, f"share of studies with platform {p} and sensor {q}", src)
    health = ev[ev.tasks.map(lambda ts: "Health and mortality" in ts)]
    count("NHealthNaturalForest", int(health.settings.map(lambda vs: "Natural forest" in vs).sum()),
          "health and mortality studies set in natural forests", src)
    geo, per_country, _, biome = geography.table()
    src = "figure_data/geography.csv (geography.table)"
    located = geo[geo.countries != ""]
    count("NStudiesWithCountry", len(located), "studies that name their study area by country", src)
    percent("PctStudiesWithCountry", len(located), n, "share of studies that name a country", src)
    count("NStudiesNoCountry", n - len(located), "studies with a global, synthetic or unstated study area", src)
    count("NCountries", len(per_country), "countries named by at least one study", src)
    count("NNorthern", int(located.northern.sum()), "studies wholly in the Northern Hemisphere (by country)", src)
    percent("PctNorthern", int(located.northern.sum()), len(located), "share of country-located studies wholly in the Northern Hemisphere", src)
    for c, v in per_country.most_common(5):
        count("NCountry" + camel(c), v, f"studies with a study area in {c}", src)
        percent("PctCountry" + camel(c), v, n, f"share of studies with a study area in {c}", src)
    b = pd.Series([v for vals in biome.values() for v in vals]).value_counts()
    for k, v in b.items():
        count("NBiome" + camel(k), v, f"studies with biome = {k}", src)
        percent("PctBiome" + camel(k), v, n, f"share of studies with biome = {k}", src)
    s = load("studies")
    s = s[s.extracted == True]  # noqa: E712
    s["study"] = s.paper_id.map(study_of())
    src = "compiled/studies.csv (a study counts when any of its reports states it)"
    for col, val, key in (("code_available", "yes", "Code"), ("weights_available", "yes", "Weights"), ("data_available", "yes", "Data"),
                          ("data_available", "on_request", "DataOnRequest"), ("data_available", "partial", "DataPartial")):
        v = s[s[col] == val].study.nunique()
        count("N" + key + "Available", v, f"studies with {col} = {val}", src)
        percent("Pct" + key + "Available", v, n, f"share of studies with {col} = {val}", src)
    d = load("datasets")
    d["study"] = d.paper_id.map(study_of())
    d["kind"] = d.dataset_key.map(pd.read_csv(paths.DATASETS_CSV).set_index("dataset_key").kind)  # registry: own, public or other
    src = "compiled/datasets.csv with the kind of each dataset key in Extraction/datasets.csv"
    pub = d[d.kind == "public"]
    shared = pub.groupby("dataset_key").agg(studies=("study", "nunique"), name=("dataset_name", "first")).reset_index()
    shared = shared[shared.studies >= 2].sort_values("studies", ascending=False)
    write(shared, "shared_public_datasets")
    count("NSharedPublicDatasets", len(shared), "public datasets used by two or more studies", "figure_data/shared_public_datasets.csv")
    shared_table(pub, shared)
    included_citations()
    kinds = d.groupby("study").kind.agg(set)
    for key, test, text in (("OwnDataOnly", lambda k: k == {"own"}, "studies that use only data they collected or labelled themselves"),
                            ("PublicData", lambda k: "public" in k, "studies that use at least one public dataset"),
                            ("UnidentifiedData", lambda k: "other" in k and "public" not in k, "studies whose data source is not identified")):
        v = int(kinds.map(test).sum())
        count("NStudies" + key, v, text, src)
        percent("PctStudies" + key, v, n, "share of " + text, src)
    v = pub[pub.dataset_key.isin(shared.dataset_key)].study.nunique()
    count("NStudiesSharedBenchmark", v, "studies that use a public dataset also used by another study", "figure_data/shared_public_datasets.csv")
    percent("PctStudiesSharedBenchmark", v, n, "share of studies that use a public dataset also used by another study",
            "figure_data/shared_public_datasets.csv")


def rq1():
    c, (lo, hi) = rq1_comparisons.data()
    n = len(studies())
    src = "figure_data/rq1_comparisons.csv (rq1_comparisons.data)"
    count("NComparisons", len(c), "transformer versus non-transformer comparisons (preferred metric, percent)", src)
    count("NComparisonStudies", c.study.nunique(), "studies with at least one comparison", src)
    percent("PctComparisonStudies", c.study.nunique(), n, "share of studies with a comparison", src)
    per = c.groupby("study").diff_pp.median()
    q1, md, q3 = np.percentile(per, [25, 50, 75])
    signed("MedianDiffAll", md, "median of the per-study differences, all studies (pp)", src)
    signed("QOneDiffAll", q1, "first quartile of the per-study differences (pp)", src)
    signed("QThreeDiffAll", q3, "third quartile of the per-study differences (pp)", src)
    o = per.map(outcome)
    for k, key in (("up", "Up"), ("tie", "Tie"), ("down", "Down")):
        count("N" + key + "All", int((o == k).sum()), f"studies whose per-study difference is '{k}' (tie: within 1 pp)", src)
        percent("Pct" + key + "All", int((o == k).sum()), len(per), f"share of the comparison studies whose per-study difference is '{k}'", src)
    for col, keys, pre in (("transformer_category", {k: FAM_KEY[k] for k in FAMILIES}, "Fam"),
                           ("task_group", None, "Task"), ("transformer_role", {"Proposed by the study": "Proposed", "Baseline only": "BaselineOnly"}, "Role")):
        for r in per_study(c, col, "diff_pp").itertuples():
            key = pre + (keys[r.group] if keys else camel(r.group))
            count("NCmpStudies" + key, r.studies, f"studies in RQ1 group {r.group}", "figure_data/rq1_summary.csv")
            signed("MedianDiff" + key, r.median, f"median per-study difference in group {r.group} (pp)", "figure_data/rq1_summary.csv")
            signed("QOneDiff" + key, r.q1, f"first quartile in group {r.group} (pp)", "figure_data/rq1_summary.csv")
            signed("QThreeDiff" + key, r.q3, f"third quartile in group {r.group} (pp)", "figure_data/rq1_summary.csv")
            for k in ("up", "tie", "down"):
                count("N" + k.capitalize() + key, getattr(r, k), f"studies in group {r.group} whose difference is '{k}'", "figure_data/rq1_summary.csv")
    fam = per_study(c, "transformer_category", "diff_pp").set_index("group")
    main = fam.loc[["cnn_transformer_hybrid", "pure_vit", "hierarchical_vit"]]
    signed("MedianDiffMaxMainFam", main["median"].max(), "largest median difference among hybrids, pure ViTs and hierarchical ViTs (pp)",
           "figure_data/rq1_summary.csv")
    CLAIMS.append(("the interquartile ranges of hybrids, pure ViTs and hierarchical ViTs reach zero",
                   bool(((main.q1 <= 0) & (main.q3 >= 0)).all())))
    keys = bib_key()
    out = c[c.outlier].assign(key=lambda d: d.paper_id.map(keys) if "paper_id" in d else d.study.map(keys))
    CLAIMS.append(("RQ1 outliers below the fence are the four studies the text names",
                   set(out[out.diff_pp < 0].key) == {"speckenwirth_treesegtoolbox_2024", "chen_zero-shot_2025", "kaviani_tree_2025",
                                                     "lungo_vaschetti_treepseco_2025"}))
    CLAIMS.append(("RQ1 outliers above the fence are the five studies the text names",
                   set(out[out.diff_pp > 0].key) == {"jiang_detection_2025", "dersch_novel_2022", "chen_two-stage_2025",
                                                     "valicharla_precision_2023", "zhang_parameter-efficient_2025"}))
    count("NOutliersRQOne", int(c.outlier.sum()), "RQ1 comparisons outside Tukey's fences", "figure_data/rq1_outliers.csv")
    signed("FenceLowRQOne", lo, "lower Tukey fence of the RQ1 differences (pp)", src)
    signed("FenceHighRQOne", hi, "upper Tukey fence of the RQ1 differences (pp)", src)


def rq2():
    t, (lo, hi) = rq2_transfer.data()
    n = len(studies())
    src = "figure_data/rq2_transfer.csv (rq2_transfer.data)"
    count("NTransferPairs", len(t), "complete in-domain and out-of-domain pairs (percent metrics)", src)
    count("NTransferStudies", t.study.nunique(), "studies with a complete transfer pair", src)
    percent("PctTransferStudies", t.study.nunique(), n, "share of studies with a complete transfer pair", src)
    per = t.groupby("study").change_pp.median()
    q1, md, q3 = np.percentile(per, [25, 50, 75])
    signed("MedianChangeAll", md, "median of the per-study changes under domain shift (pp)", src)
    signed("QOneChangeAll", q1, "first quartile of the per-study changes (pp)", src)
    signed("QThreeChangeAll", q3, "third quartile of the per-study changes (pp)", src)
    o = per.map(outcome)
    for k, key in (("up", "Up"), ("tie", "Tie"), ("down", "Down")):
        count("N" + key + "Transfer", int((o == k).sum()), f"studies whose per-study change is '{k}'", src)
    shift_key = {"Site": "Site", "Site + sensor or platform": "SiteAcq", "Region": "Region", "Region + sensor or platform": "RegionAcq",
                 "Biome": "Biome", "Sensor, platform or resolution only": "AcqOnly", "Species only": "SpeciesOnly"}
    fam_key = {FAMILY[k]: FAM_KEY[k] for k in FAMILIES} | {"Non-transformer": "NonTransformer"}
    for col, keys in (("shift_group", shift_key), ("family_group", fam_key)):
        for r in per_study(t, col, "change_pp").itertuples():
            key = keys[r.group]
            count("NTrStudies" + key, r.studies, f"studies in RQ2 group {r.group}", "figure_data/rq2_summary.csv")
            count("NTrPairs" + key, r.points, f"pairs in RQ2 group {r.group}", "figure_data/rq2_summary.csv")
            signed("MedianChange" + key, r.median, f"median per-study change in group {r.group} (pp)", "figure_data/rq2_summary.csv")
    geo = geography.table()[0].set_index("study")
    located = geo.loc[geo.index.intersection(t.study.unique())]
    located = located[located.countries != ""]
    count("NTransferStudiesWithCountry", len(located), "studies with a complete transfer pair that name a country",
          "figure_data/geography.csv and rq2_transfer.csv")
    count("NTransferNorthern", int(located.northern.sum()), "of those, studies wholly in the Northern Hemisphere (by country)",
          "figure_data/geography.csv and rq2_transfer.csv")
    count("NShiftKinds", t.shift_group.nunique(), "kinds of shift among the complete transfer pairs", src)
    CLAIMS.append(("one RQ2 outlier lies above the upper fence ('the one large positive change')", int((t.outlier & (t.change_pp > 0)).sum()) == 1))
    CLAIMS.append(("one study tests a biome shift ('the single biome shift')", t[t.shift_group == "Biome"].study.nunique() == 1))
    count("NOutliersRQTwo", int(t.outlier.sum()), "RQ2 pairs outside Tukey's fences", "figure_data/rq2_outliers.csv")
    count("NOutlierStudiesRQTwo", t[t.outlier].study.nunique(), "studies with an RQ2 outlier", "figure_data/rq2_outliers.csv")
    signed("FenceLowRQTwo", lo, "lower Tukey fence of the RQ2 changes (pp)", src)
    signed("FenceHighRQTwo", hi, "upper Tukey fence of the RQ2 changes (pp)", src)
    diffs = []
    for (study, *_), g in t.groupby(["study", "source", "target", "metric", "level"]):
        if g.transformer.any() and (~g.transformer).any():
            diffs.append((study, g[g.transformer].change_pp.median() - g[~g.transformer].change_pp.median()))
    d = pd.DataFrame(diffs, columns=["study", "delta"]).groupby("study")["delta"].median()
    count("NMatchedStudies", len(d), "studies that test transformer and non-transformer models under the same shift", src)
    count("NMatchedTransformerLess", int((d > 0).sum()), "of those, studies where the transformer changes less or gains more (median)", src)


def rq3():
    st, fm, fm_st = rq3_training.tables()
    n = len(st)
    src = "figure_data/rq3_strategies.csv (rq3_training.tables)"
    stated = int((st.strategy != "Not reported").sum())
    count("NTrainingStated", stated, "studies that state how their principal transformer is trained", src)
    percent("PctTrainingStated", stated, n, "share of studies that state the training of the principal transformer", src)
    count("NTrainingNotReported", n - stated, "studies that do not state it", src)
    percent("PctTrainingNotReported", n - stated, n, "share of studies that do not state it", src)
    for k, v in st.strategy.value_counts().items():
        count("NStrat" + camel(k), v, f"studies whose principal transformer is: {k}", src)
    for k in FAMILIES:
        sub = st[st.family == k]
        v = int((sub.strategy != "Not reported").sum())
        count("NTrainingStated" + FAM_KEY[k], v, f"{FAMILY[k]} studies that state how their principal transformer is trained", src)
        percent("PctTrainingStated" + FAM_KEY[k], v, len(sub), f"share of {FAMILY[k]} studies that state it", src)
    src = "figure_data/rq3_foundation_models.csv (rq3_training.tables)"
    count("NFMStudies", len(fm_st), "studies that use a foundation or vision-language model", src)
    percent("PctFMStudies", len(fm_st), n, "share of studies that use a foundation or vision-language model", src)
    for k in rq3_training.STRATEGIES:
        v = int(fm_st.strategy.map(lambda s: k in s).sum())
        count("NFM" + camel(k), v, f"of those, studies that use one in the mode: {k}", src)
        percent("PctFM" + camel(k), v, len(fm_st), f"share of foundation or vision-language studies using the mode: {k}", src)
    de = load("data_efficiency")
    de["study"] = de.paper_id.map(study_of())
    count("NLabelEfficiencyStudies", de.study.nunique(), "studies that report results for several amounts of training labels",
          "compiled/data_efficiency.csv")
    peft, pairs, t = rq3_training.same_setting()
    count("NPEFTStudies", peft.study.nunique(), "studies with a parameter-efficiently tuned model", "compiled/models.csv")
    pairs = write(pairs, "peft_same_setting")
    count("NPEFTSameSetting", pairs.study.nunique() if len(pairs) else 0,
          "studies that report a parameter-efficiently tuned model and the same base model fully fine-tuned, on the same dataset and metric",
          "figure_data/peft_same_setting.csv")
    peft_table(t)


def included_citations():
    """generated/included_studies.tex: \\CiteIncludedStudies cites every included report, so every study is in the reference list
    (and can carry its number in the figures)."""
    keys = sorted(set(bib_key().values()))
    tex = ("% Generated by Scripts/figures/manuscript_numbers.py; do not edit by hand.\n"
           f"% {len(keys)} included reports ({len(set(study_of().values()))} studies).\n"
           "\\newcommand{\\CiteIncludedStudies}{\\citep{" + ",".join(keys) + "}}\n")
    (paths.PAPER_GENERATED / "included_studies.tex").write_text(tex, encoding="utf-8")


# Short name and data of each shared public dataset, as the studies that use it describe it (compiled/datasets.csv,
# dataset_name). The registry names carry extraction notes in brackets and are not shown in the manuscript.
SHARED_DISPLAY = {  # display name, data and, where the dataset has its own paper in the bib, that paper
    "neon_treeevaluation": ("NEON TreeEvaluation", "Airborne RGB and hyperspectral images", "weinstein_benchmark_2021"),
    "yosemite_trees": ("Yosemite tree counting", "Aerial RGB images"),
    "al_ain_date_palm": ("Al Ain date palm", "UAV RGB images"),
    "kcl_london": ("KCL-London tree counting", "RGB images, 0.2 m GSD"),
    "mopad": ("MOPAD oil palm", "UAV RGB images"),
    "stpctls": ("STPCTLS", "TLS point clouds of single trees"),
    "canatree100": ("CanaTree100", "Forest video frames"),
    "for_instance_v1": ("FOR-instance", "UAV laser scanning", "puliti_for-instance_2023"),
    "forestsemantic_evo": ("ForestSemantic", "TLS point clouds"),
    "jiangsu_gaofen2": ("Jiangsu tree counting", "Gaofen-2 satellite images, 0.8 m GSD"),
    "lewos": ("LeWoS", "TLS point clouds of tropical trees"),
    "quebec_trees_sbl_2021": ("Quebec Trees", "UAV RGB images", "cloutier_influence_2023"),
    "whu_stree_nj": ("WHU-STree Nanjing", "Street-tree images and point clouds"),
}


def shared_table(pub, shared):
    """The shared-datasets table of the manuscript: every public dataset used by two or more studies, its data, and the studies
    that use it."""
    key_of = {}
    for pid, key in bib_key().items():
        key_of.setdefault(study_of()[pid], key)  # one citation per study (its first report)
    missing = sorted(set(shared.dataset_key) - set(SHARED_DISPLAY))
    if missing:
        raise SystemExit(f"shared datasets without a display name in SHARED_DISPLAY: {missing}")
    rows = []
    for x in shared.itertuples():
        keys = sorted({key_of[s_] for s_ in pub[pub.dataset_key == x.dataset_key].study})
        name, data, *src = SHARED_DISPLAY[x.dataset_key]
        name += f" \\citep{{{src[0]}}}" if src else ""
        rows.append(f"{name} & {data} & {x.studies} & \\citep{{{','.join(keys)}}} \\\\")
    tex = ("% Generated by Scripts/figures/manuscript_numbers.py; do not edit by hand. Names and data: SHARED_DISPLAY.\n"
           "% Width \\linewidth: a one-column table.\n"
           "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}>{\\raggedright\\arraybackslash}Xc"
           ">{\\raggedright\\arraybackslash}p{0.22\\linewidth}@{}}\n"
           "\\toprule\nPublic dataset & Data & Studies & Used by \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}%\n")
    (paths.PAPER_GENERATED / "shared_datasets.tex").write_text(tex, encoding="utf-8")


def peft_table(t):
    """The same-setting comparison of adaptation strategies (manuscript Table 6) and the numbers the text quotes from it."""
    pid, ds = t.paper_id.iloc[0], t.dataset_key.iloc[0]
    write(t[["method", "model_name", "training_strategy", "trainable_params_pct", "ap50", "ap50_95", "page", "location"]], "peft_table")
    full = t[t.training_strategy == "full_fine_tuning"].iloc[0]
    pe = t[t.training_strategy.astype(str).str.startswith("peft")]
    best = pe.sort_values("ap50", ascending=False).iloc[0]
    head = t[t.training_strategy == "head_only"]
    src = "figure_data/peft_table.csv (" + full.location + ", p." + str(int(full.page)) + ")"
    plain("PEFTFullAPFifty", full.ap50, f"{full.ap50:.1f}", "AP50 of full fine-tuning in the same-setting study", src)
    plain("PEFTBestAPFifty", best.ap50, f"{best.ap50:.1f}", f"best AP50 of a parameter-efficient method ({best.method})", src)
    plain("PEFTBestName", best.method, best.method.split(" (")[0], "name of that method as printed", src)
    plain("PEFTBestTrainable", best.trainable_params_pct, f"{best.trainable_params_pct:.2f}", "its share of trainable parameters (percent)", src)
    percent("PctPEFTBestOfFull", best.ap50, full.ap50, "its AP50 as a share of the full fine-tuning AP50", src)
    plain("PEFTFullAPStrict", full.ap50_95, f"{full.ap50_95:.1f}", "AP50:95 of full fine-tuning in the same-setting study", src)
    plain("PEFTBestAPStrict", best.ap50_95, f"{best.ap50_95:.1f}", f"AP50:95 of the method with the best AP50 ({best.method})", src)
    plain("PEFTGapAPFifty", round(full.ap50 - best.ap50, 1), f"{full.ap50 - best.ap50:.1f}",
          "AP50 of full fine-tuning minus the best parameter-efficient AP50 (points)", src)
    count("NPEFTVariants", len(pe), "parameter-efficient methods compared in that study", src)
    plain("PEFTLowAPFifty", pe.ap50.min(), f"{pe.ap50.min():.1f}", "lowest AP50 among them", src)
    plain("PEFTMinTrainable", pe.trainable_params_pct.min(), f"{pe.trainable_params_pct.min():.2f}", "smallest share of trainable parameters among them (percent)", src)
    plain("PEFTMaxTrainable", pe.trainable_params_pct.max(), f"{pe.trainable_params_pct.max():.2f}", "largest share of trainable parameters among them (percent)", src)
    if len(head):
        h = head.iloc[0]
        plain("PEFTHeadOnlyAPFifty", h.ap50, f"{h.ap50:.1f}", f"AP50 with the backbone frozen ({h.method})", src)
        percent("PctHeadOnlyOfFull", h.ap50, full.ap50, "frozen-backbone AP50 as a share of the full fine-tuning AP50", src)
    n_img = load("datasets").query("paper_id == @pid and dataset_key == @ds").n_images.iloc[0]
    count("NPEFTImages", n_img, "images in the dataset of the same-setting study (80 percent for training, 20 percent for testing)",
          "compiled/datasets.csv")
    rows = [f"{x.method.replace('_', ' ')} & {x.trainable_params_pct:.2f} & {x.ap50:.1f} & {x.ap50_95:.1f} \\\\" for x in t.itertuples()]
    tex = ("% Generated by Scripts/figures/manuscript_numbers.py; do not edit by hand. Width \\linewidth: a one-column table.\n"
           "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrr@{}}\n\\toprule\n"
           "Adaptation & Trainable (\\%) & AP50 & AP50:95 \\\\\n\\midrule\n" + "\n".join(rows)
           + "\n\\bottomrule\n\\end{tabular*}%\n")  # % : no stray space at the end of the input file
    (paths.PAPER_GENERATED / "peft_same_setting.tex").write_text(tex, encoding="utf-8")


def fmt(v):
    """A recorded value as printed: no trailing zeros, no float artefacts."""
    return f"{round(float(v), 4):g}"


def listing(values, last="and"):
    v = [fmt(x) for x in values]
    return v[0] if len(v) == 1 else ", ".join(v[:-1]) + f" {last} " + v[-1]


def paper(key):
    pid = {v: k for k, v in bib_key().items()}.get(key)
    assert pid, f"no included paper with bib key {key}"
    return pid


def search_facts():
    """The search window, the Scopus loading date and the dates the searches were run (query_terms.py, search_log.json)."""
    q, src = query_terms, "Scripts/search/query_terms.py"
    plain("SearchYearFrom", q.YEAR_FROM, str(q.YEAR_FROM), "first publication year searched", src)
    plain("SearchYearTo", q.YEAR_TO, str(q.YEAR_TO), "last publication year searched", src)
    plain("SearchYearExcluded", q.YEAR_TO + 1, str(q.YEAR_TO + 1), "first year of versions of record outside the date rule", src)
    d = datetime.strptime(q.INDEX_CUTOFF, "%Y%m%d")
    plain("SearchIndexCutoff", q.INDEX_CUTOFF, f"{d.day} {d:%B %Y}", "Scopus records first loaded before this date", src)
    log = json.loads((paths.SEARCH / "search_log.json").read_text())
    tz, src = ZoneInfo(log["timezone"]), "Literature_Extraction/Search/search_log.json"
    when = {}
    for key, name in (("Scopus", "Scopus"), ("WoS", "Web of Science"), ("ArXiv", "arXiv")):
        t = datetime.fromisoformat(log["sources"][name]["run_at_utc"].replace("Z", "+00:00")).astimezone(tz)
        when[key] = t
        plain("SearchDate" + key, t.date().isoformat(), f"{t.day} {t:%B}", f"day the {name} search was run ({log['timezone']})", src)
        plain("SearchDate" + key + "Full", t.date().isoformat(), f"{t.day} {t:%B %Y}", f"date the {name} search was run", src)
    months = {f"{t:%B %Y}" for t in when.values()}
    assert len(months) == 1, months
    plain("SearchMonth", months.pop(), f"{when['Scopus']:%B %Y}", "month in which all three searches were run", src)
    strings = {"SearchStringScopus": q.render_scopus(), "SearchStringWoS": q.render_wos(),
               "SearchStringArXiv": json.loads((paths.SEARCH / "results" / "arxiv_run.json").read_text())["api_query"],
               "SearchStringThirdBlock": "(" + " OR ".join(q._q(t) for t in q.BLOCK_C) + ")"}
    lines = ["% Generated by Scripts/figures/manuscript_numbers.py from Scripts/search/query_terms.py and results/arxiv_run.json."]
    for k, v in strings.items():
        assert not re.search(r"[\\{}$&#^_%~]", v), f"{k} holds a character that LaTeX would need escaped"
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (paths.PAPER_GENERATED / "search_strings.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def constants():
    """Settings of the analysis, taken from the code that applies them."""
    src = "Scripts/figures/common.py"
    plain("ConstTieZone", TIE_PP, fmt(TIE_PP), "half-width of the tie zone (pp)", src)
    unit = lambda u: u if TIE_PP == 1 else u + "s"  # noqa: E731
    word = WORDS[int(TIE_PP)] if float(TIE_PP).is_integer() and TIE_PP <= 10 else fmt(TIE_PP)
    plain("ConstTieZonePP", TIE_PP, f"{word} {unit('percentage point')}", "the tie zone in words", src)
    plain("ConstTieZonePoint", TIE_PP, f"{word} {unit('point')}", "the tie zone in words (short)", src)
    plain("ConstTukeyK", TUKEY_K, fmt(TUKEY_K), "Tukey's fences: interquartile ranges beyond the quartiles", src)
    plain("ConstPctConfidence", 100 * CONFIDENCE, fmt(100 * CONFIDENCE), "confidence of the upper bound on the error rate (percent)",
          "Scripts/figures/manuscript_numbers.py (CONFIDENCE)")


def years():
    """Years that are findings: from the publication years of the included studies."""
    st, src = studies(), "figure_data/studies.csv (common.studies)"
    yr = st.year.astype(int)
    first, last = int(yr.min()), int(yr.max())
    plain("YearFirstStudy", first, str(first), "publication year of the earliest included study", src)
    plain("YearLastStudy", last, str(last), "publication year of the latest included study", src)
    by = st.assign(year=yr).groupby(["year", "family"]).size().unstack(fill_value=0)
    lead = [y for y in by.index if by.loc[y, "hierarchical_vit"] > by.loc[y].drop("hierarchical_vit").max()]
    assert len(lead) == 1, lead
    plain("YearHierViTLead", lead[0], str(lead[0]), "the year in which hierarchical ViTs are the most frequent family", src)
    for fam, key in (("foundation_model", "FM"), ("vision_language_model", "VLM")):
        y = int(yr[st.family == fam].min())
        plain("YearFirst" + key, y, str(y), f"first year with a study of family {FAMILY[fam]}", src)
    # the text pairs these years with the per-year counts named after them (\NStudiesMMXXII, \NFamHierViTInMMXXIII, ...)
    CLAIMS.append(("the year macros agree with the per-year counts the text pairs them with",
                   (first, last, lead[0], int(yr[st.family == "foundation_model"].min()), int(yr[st.family == "vision_language_model"].min()))
                   == (2022, 2025, 2023, 2024, 2024)))


def single_paper_values():
    """Values the text quotes from one study, read from its rows in the compiled tables (paper, item, page)."""
    tr, res, de, comp = load("transfer"), load("results"), load("data_efficiency"), load("studies")
    def src(key, page, loc):
        return f"{key}, p. {page}, {loc}"
    # A DETR palm detector on the storm-hit estate (RQ2 outliers)
    k = tr[(tr.paper_id == paper("kwong_enhancing_2024")) & tr.target.str.contains("storm") & (tr.location_in == "Table 1")]
    base, gan = k[k.model == "model1"], k[k.model == "model2"]
    assert len(base) == len(gan) == 1
    base, gan = base.iloc[0], gan.iloc[0]
    s = src("kwong_enhancing_2024", base.page_in, base.location_in)
    plain("ValStormBaselineIn", base.value_in_std, fmt(base.value_in_std), "DETR baseline AP50 in domain", s)
    plain("ValStormBaselineOut", base.value_out_std, fmt(base.value_out_std), "DETR baseline AP50 on the storm-hit estate", s)
    plain("ValStormGANOut", gan.value_out_std, fmt(gan.value_out_std), "GAN-trained variant AP50 on the storm-hit estate", s)
    # Date-palm segmenters: another city and UAV system, and the own test set resampled (RQ2, where models hold up)
    g = tr[tr.paper_id == paper("gibril_large-scale_2023")]
    cross = g[(g.shift_type == "site;sensor;resolution") & (g.transformer == True)]  # noqa: E712
    resampled = g[g.shift_type == "resolution"]
    s = src("gibril_large-scale_2023", cross.page_in.iloc[0], cross.location_in.iloc[0])
    gsd = comp[comp.paper_id == paper("gibril_large-scale_2023")].gsd_cm.iloc[0]
    plain("ValPalmGSD", gsd, fmt(gsd), "ground sampling distance of the training imagery (cm)", "compiled/studies.csv (gibril_large-scale_2023)")
    for key, v in (("ValPalmCrossCityLow", cross.change_pp.min()), ("ValPalmCrossCityHigh", cross.change_pp.max())):
        plain(key, v, f"${v:+g}$".replace("+-", "-"), "change of mean IoU of the transformer models in another city and UAV system (pp)", s)
    gsds = set(resampled.target.str.extract(r"(\d+(?:\.\d+)?) cm", expand=False))
    assert len(gsds) == 1, gsds
    v = gsds.pop()
    plain("ValPalmResampledGSD", v, fmt(v), "resolution the own test set was resampled to (cm), from the recorded target", s)
    loss = sorted(-resampled.change_pp)
    plain("ValPalmResampleLossLow", loss[0], fmt(loss[0]), "smallest loss of mean IoU after resampling (points)", s)
    plain("ValPalmResampleLossHigh", loss[-1], fmt(loss[-1]), "largest loss of mean IoU after resampling (points)", s)
    y = tr[tr.paper_id == paper("yan_tm-wsnet_2025")]
    assert len(y) == 1
    plain("ValRubberLoss", -y.change_pp.iloc[0], fmt(-y.change_pp.iloc[0]), "loss of mean IoU on new point clouds of the same plantation",
          src("yan_tm-wsnet_2025", y.page_in.iloc[0], y.location_in.iloc[0]))
    z = tr[tr.paper_id == paper("zhang_mtcdnet_2025")]
    loss = sorted(-z.change_pp)
    assert len(loss) == 2
    s = src("zhang_mtcdnet_2025", "12-13", "Tables 1 and 2")
    plain("ValSwapLossLow", loss[0], fmt(loss[0]), "AP50:95 lost when training and test datasets are swapped, smaller direction", s)
    plain("ValSwapLossHigh", loss[1], fmt(loss[1]), "AP50:95 lost when training and test datasets are swapped, larger direction", s)
    # The same-setting PEFT study: its data split
    split = {m for m in res[res.paper_id == paper("zhang_parameter-efficient_2025")].setting.str.extract(r"test split \((\d+)%\)", expand=False)}
    assert len(split) == 1, split
    test = int(split.pop())
    s = "zhang_parameter-efficient_2025, p. 11, Sec. 4.1 (results: setting)"
    plain("ValPEFTTestSplit", test, str(test), "share of the images used for testing (percent)", s)
    plain("ValPEFTTrainSplit", 100 - test, str(100 - test), "share of the images used for training (percent)", s)
    # The two reports of one study: the same wmAP
    a = res[(res.paper_id == paper("teng_assessing_2025")) & (res["item"] == "res36")]
    b = res[(res.paper_id == paper("teng_bringing_2025")) & (res["item"] == "res48")]
    assert len(a) == len(b) == 1 and fmt(a.value.iloc[0]) == fmt(b.value.iloc[0]) and "wmAP" in a.metric_note.iloc[0]
    plain("ValDSMPrompterWmAP", a.value.iloc[0], fmt(a.value.iloc[0]), "weighted mean average precision printed in both reports (percent)",
          "teng_assessing_2025, p. 4, Table 1; teng_bringing_2025, p. 6, Table 1")
    # Label amounts of the data-efficiency studies
    def amounts(key, unit):
        d = de[de.paper_id == paper(key)]
        assert set(d.label_unit) == {unit}, (key, set(d.label_unit))
        return sorted(set(d.label_amount.astype(float))), f"{key}, p. {d.page.iloc[0]}, {d.location.iloc[0]}"
    v, s = amounts("sachdeva_distribution_2024", "images")
    plain("ValSachdevaImages", v, listing(v), "training images compared (list)", s)
    CLAIMS.append(("the smallest amount in Sachdeva et al. is a single image", v[0] == 1))
    v, s = amounts("amirkolaee_adatreeformer_2024", "images")
    plain("ValAdaTreeFormerImages", v, listing(v, "or"), "labelled target-domain images compared (list)", s)
    for key, mac, unit in (("amirkolaee_treeformer_2023", "TreeFormer", "percent_of_training_set"), ("hua_gdpgo-sam_2025", "GDPGO", "images"),
                           ("grondin_leveraging_2024", "Grondin", "percent_of_training_set"),
                           ("liu_multifeature_2025", "LiuLabels", "percent_of_training_set")):
        v, s = amounts(key, unit)
        plain(f"Val{mac}Low", v[0], fmt(v[0]), f"smallest label amount ({unit})", s)
        plain(f"Val{mac}High", v[-1], fmt(v[-1]), f"largest label amount ({unit})", s)


def other_reviews():
    """Figures quoted from earlier reviews (Related work), recorded with page and quote in Extraction/related_reviews.csv."""
    for r in csv.DictReader(open(paths.EXTRACTION / "related_reviews.csv", encoding="utf-8")):
        name = "Review" + camel(r["bib_key"].split("_")[0]) + camel(r["quantity"])
        text = f"{date.fromisoformat(r['value']):%B %Y}" if r["quantity"].startswith("period") else r["value"]
        plain(name, r["value"], text, f"{r['quantity']} of {r['bib_key']}", f"Extraction/related_reviews.csv ({r['page']}: \"{r['quote']}\")")


def check_claims():
    failed = [d for d, ok in CLAIMS if not ok]
    if failed:
        raise SystemExit("statements of the text no longer hold:\n  " + "\n  ".join(failed))
    print(f"  {len(CLAIMS)} statements of the text checked against the data")


def main():
    for part in (corpus, corpus_shape, rq1, rq2, rq3, search_facts, constants, years, single_paper_values, other_reviews, check_claims):
        part()
    df = write(pd.DataFrame(ROWS), "numbers")
    lines = ["% Generated by Scripts/figures/manuscript_numbers.py from the compiled extraction tables; do not edit by hand.",
             "% Definitions and sources: Literature_Extraction/Analysis/figure_data/numbers.csv"]
    lines += [f"\\newcommand{{\\{r.macro}}}{{{r.text}}}  % {r.definition}" for r in df.itertuples()]
    paths.PAPER_GENERATED.mkdir(parents=True, exist_ok=True)
    (paths.PAPER_GENERATED / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {len(df)} numbers written to generated/numbers.tex and figure_data/numbers.csv")


if __name__ == "__main__":
    main()
