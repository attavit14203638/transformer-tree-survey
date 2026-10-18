"""RQ1: the best transformer against the strongest non-transformer deep baseline of the same study and setting.

One comparison per study, task, level and dataset, on the study's preferred metric (METRIC_ORDER in compile.py); bounded metrics
in percent; difference in percentage points. Panels: (a) architecture family of the best transformer, (b) task, (c) whether the
best transformer is the model the study proposes or only a baseline. Outliers are named in panel (a) and listed, with their
context, in figure_data/rq1_outliers.csv.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D

from common import (COLOR, FAMILIES, FAMILY_SHORT, INK, MARKER, METRIC, TASK_GROUP, TASKS, box_handles, context, family_handles, labels,
                    legend_row, load, outcome, panel_titles, points_panel, save, set_style, tukey, write)

ROLES = ["Proposed by the study", "Baseline only"]
YLIM = (-48, 60)


def data():
    c = load("comparisons")
    c = c[(c.canonical_metric == True) & (c.unit_std == "%") & c.diff_pp.notna()].copy()  # noqa: E712
    c["task_group"] = c.task.map(TASK_GROUP)
    c["metric_label"] = c.metric.map(METRIC)
    c["outcome"] = c.diff_pp.map(outcome)
    c["colour_key"] = c.transformer_category
    t, b = context(c, "transformer_ref"), context(c, "baseline_ref")
    c["transformer_role"] = np.where(t.model_role.to_numpy() == "proposed", ROLES[0], ROLES[1])
    c["transformer_strategy"], c["transformer_pretraining"], c["transformer_setting"] = t.training_strategy.to_numpy(), t.pretraining.to_numpy(), t.setting.to_numpy()
    c["baseline_strategy"], c["baseline_pretraining"] = b.training_strategy.to_numpy(), b.pretraining.to_numpy()
    c["label"] = c.transformer_ref.str.split(":").str[0].map(labels())
    lo, hi = tukey(c.diff_pp)
    c["outlier"] = (c.diff_pp < lo) | (c.diff_pp > hi)
    return c, (lo, hi)


def main():
    c, (lo, hi) = data()
    rng = np.random.default_rng(2026)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), gridspec_kw={"wspace": 0.1, "width_ratios": [5, 5, 2.3]})
    fams = [k for k in FAMILIES if (c.transformer_category == k).any()]
    s1 = points_panel(axes[0], c, "transformer_category", fams, FAMILY_SHORT, "diff_pp", YLIM, box_colour=lambda g: COLOR[g],
                      label_outliers=True, rng=rng)
    task_ticks = {t: t.replace(" and ", " and\n").replace(" segmentation", "\nsegmentation").replace(" classification", "\nclassification") for t in TASKS}
    s2 = points_panel(axes[1], c, "task_group", [t for t in TASKS if (c.task_group == t).any()], task_ticks, "diff_pp", YLIM, rng=rng)
    s3 = points_panel(axes[2], c, "transformer_role", ROLES, {ROLES[0]: "Proposed by\nthe study", ROLES[1]: "Baseline\nonly"}, "diff_pp", YLIM, rng=rng)
    axes[0].set_ylabel("Best transformer minus strongest non-transformer\ndeep baseline (percentage points)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    bottom = panel_titles(fig, [(axes[0], "(a) By architecture family"), (axes[1], "(b) By task"), (axes[2], "(c) By role of the transformer")])
    shapes = [Line2D([], [], ls="", marker=MARKER[m], color=INK, label=m) for m in MARKER if (c.metric_label == m).any()]
    legend_row(fig, [(family_handles(fams), "Architecture family (colour)", 1), (shapes, "Metric (shape)", 2),
                     (box_handles(("favours the transformer (> +1 pp)", "is a tie (within ±1 pp)", "favours the baseline (< −1 pp)")),
                      "Boxes and counts (one value per study)", 1)], y=bottom - 0.03)
    save(fig, "rq1_comparisons")
    cols = ["label", "study", "papers", "task", "task_group", "level", "metric_label", "dataset_key", "split", "best_transformer", "transformer_category",
            "transformer_role", "transformer_strategy", "transformer_pretraining", "transformer_value", "transformer_ref", "transformer_page",
            "best_baseline", "baseline_category", "baseline_strategy", "baseline_pretraining", "baseline_value", "baseline_ref", "baseline_page",
            "diff_pp", "outcome", "outlier", "transformer_setting"]
    write(c[cols].sort_values("diff_pp"), "rq1_comparisons")
    write(c[c.outlier][cols].sort_values("diff_pp"), "rq1_outliers")
    write(pd.concat([s1.assign(panel="a"), s2.assign(panel="b"), s3.assign(panel="c")]), "rq1_summary")
    print(f"  {len(c)} comparisons from {c.study.nunique()} studies; outliers outside [{lo:.1f}, {hi:.1f}] pp: {int(c.outlier.sum())}")


if __name__ == "__main__":
    set_style()
    main()
