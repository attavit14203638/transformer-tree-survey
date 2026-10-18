"""RQ2: performance of the same trained model in its own domain and under a reported domain shift (complete transfer pairs).

Change = out-of-domain minus in-domain value (percentage points, bounded metrics in percent). Shift groups follow the shift types
the papers state: the largest geographic shift (biome, region or site), with '+ sensor or platform' when the sensor, platform or
resolution also changes. Panels: (a) kind of shift, (b) model family. Outliers are named in panel (a) and listed in
figure_data/rq2_outliers.csv.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D

from common import (COLOR, FAMILIES, FAMILY, FAMILY_SHORT, GREY, INK, METRIC, box_handles, family_handles, labels, legend_row, load, outcome,
                    panel_titles, points_panel, save, set_style, study_of, tukey, write)

SHIFTS = ["Site", "Site + sensor or platform", "Region", "Region + sensor or platform", "Biome", "Sensor, platform or resolution only", "Species only"]
SHIFT_TICK = {"Site": "Site", "Site + sensor or platform": "Site +\nsensor/platform", "Region": "Region",
              "Region + sensor or platform": "Region +\nsensor/platform", "Biome": "Biome",
              "Sensor, platform or resolution only": "Sensor, platform\nor resolution", "Species only": "Species only"}
SHIFT_MARKER = {"Site": "o", "Site + sensor or platform": "s", "Region": "^", "Region + sensor or platform": "D", "Biome": "P",
                "Sensor, platform or resolution only": "X", "Species only": "v"}
YLIM = (-98, 62)


def shift_group(s):
    s = set(str(s).split(";"))
    geo = "Biome" if "biome" in s else "Region" if "region" in s else "Site" if "site" in s else None
    acq = bool(s & {"sensor", "platform", "resolution"})
    if geo:
        return f"{geo} + sensor or platform" if acq else geo
    return "Sensor, platform or resolution only" if acq else "Species only"


def data():
    t = load("transfer")
    t = t[(t.complete == True) & (t.unit_std == "%") & t.change_pp.notna()].copy()  # noqa: E712
    t["study"] = t.paper_id.map(study_of())
    t["shift_group"] = t.shift_type.map(shift_group)
    t["metric_label"] = t.metric.map(METRIC)
    t["outcome"] = t.change_pp.map(outcome)
    t["label"] = t.paper_id.map(labels())
    t["colour_key"] = t.architecture_category.where(t.transformer == True, "non_transformer")  # noqa: E712
    t["family_group"] = t.colour_key.map(lambda k: FAMILY.get(k, "Non-transformer"))
    lo, hi = tukey(t.change_pp)
    t["outlier"] = (t.change_pp < lo) | (t.change_pp > hi)
    return t, (lo, hi)


def main():
    t, (lo, hi) = data()
    rng = np.random.default_rng(2026)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.4), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.06})
    sa = points_panel(ax1, t, "shift_group", [g for g in SHIFTS if (t.shift_group == g).any()], SHIFT_TICK, "change_pp", YLIM,
                      label_outliers=True, rng=rng, marker_of=lambda p: "o")
    ax1.set_ylabel("Out-of-domain minus in-domain value\n(percentage points)")
    fam_order = [FAMILY[k] for k in FAMILIES if (t.colour_key == k).any()] + ["Non-transformer"]
    ticks = {FAMILY[k]: FAMILY_SHORT[k] for k in FAMILIES} | {"Non-transformer": "Non-\ntransformer"}
    fam_colour = {FAMILY[k]: COLOR[k] for k in FAMILIES} | {"Non-transformer": GREY}
    sb = points_panel(ax2, t, "family_group", fam_order, ticks, "change_pp", YLIM, box_colour=lambda g: fam_colour[g], rng=rng,
                      marker_of=lambda p: SHIFT_MARKER[p["shift_group"]])
    ax2.tick_params(labelleft=False)
    bottom = panel_titles(fig, [(ax1, "(a) By kind of shift"), (ax2, "(b) By model family")])
    shifts = [Line2D([], [], ls="", marker=SHIFT_MARKER[g], color=INK, label=g) for g in SHIFTS if (t.shift_group == g).any()]
    legend_row(fig, [(family_handles([k for k in FAMILIES if (t.colour_key == k).any()], "Non-transformer"), "Model family (colour)", 1),
                     (shifts, "Kind of shift (shape in panel b)", 1),
                     (box_handles(("is higher out of domain (> +1 pp)", "changes within ±1 pp", "is lower out of domain (< −1 pp)")),
                      "Boxes and counts (one value per study)", 1)], y=bottom - 0.03)
    save(fig, "rq2_transfer")
    cols = ["label", "study", "paper_id", "item", "model_name", "architecture_category", "family_group", "transformer", "metric_label", "level",
            "shift_type", "shift_group", "source", "target", "value_in_std", "page_in", "location_in", "value_out_std", "page_out", "location_out",
            "change_pp", "change_rel_pct", "outcome", "outlier", "approximate"]
    write(t[cols].sort_values("change_pp"), "rq2_transfer")
    write(t[t.outlier][cols].sort_values("change_pp"), "rq2_outliers")
    write(pd.concat([sa.assign(panel="a"), sb.assign(panel="b")]), "rq2_summary")
    print(f"  {len(t)} complete pairs from {t.study.nunique()} studies; outliers outside [{lo:.1f}, {hi:.1f}] pp: {int(t.outlier.sum())}")


if __name__ == "__main__":
    set_style()
    main()
