"""RQ3: how the principal transformer of each study is trained, one square per study (a), and what the one study that compares
adaptation strategies of the same model on the same data reports (b).

(a) Studies are grouped by architecture family and ordered by training strategy within each family; 'not reported' means the
study does not state how its principal transformer was trained. (b) One row per adaptation strategy, ordered by the share of
parameters trained (printed on the right): the strategy's value as a share of full fine-tuning, for AP50 (filled) and the
study's stricter metric (open): AP50:95, or mAP as printed when the study does not state its IoU thresholds. same_setting()
also feeds the text macros and the PEFT table (manuscript_numbers.py).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from common import (COLOR, FAMILIES, FAMILY, GREY, GRID, INK, LIGHT, labels, legend_row, load, panel_titles, save, set_style, studies,
                    study_of, write)

STRICT_LABEL = {"ap50_95": "AP50:95", "map": "mAP"}
STRATEGY = {"from_scratch": "Trained from scratch", "full_fine_tuning": "Full fine-tuning", "peft_lora": "Parameter-efficient tuning",
            "peft_adapter": "Parameter-efficient tuning", "peft_prompt": "Parameter-efficient tuning", "peft_other": "Parameter-efficient tuning",
            "head_only": "Head only (frozen backbone)", "zero_shot": "Zero-shot or few-shot", "few_shot": "Zero-shot or few-shot",
            "not_reported": "Not reported"}
STRATEGIES = ["Trained from scratch", "Full fine-tuning", "Parameter-efficient tuning", "Head only (frozen backbone)", "Zero-shot or few-shot",
              "Not reported"]
SHADE = dict(zip(STRATEGIES, ["#2F3E56", "#5B7BA6", "#9DB7D5", "#7E6B5A", "#C4B39A", LIGHT]))  # kept apart from the family hues
PER_LINE = 26


def tables():
    """Training of each study's principal transformer; adaptation of every foundation or vision-language model, per study."""
    st = studies()
    st["strategy"] = st.principal_strategy.map(STRATEGY).fillna("Not reported")
    m = load("models")
    m["study"] = m.paper_id.map(study_of())
    fm = m[m.architecture_category.isin(["foundation_model", "vision_language_model"])].copy()
    fm["strategy"] = fm.training_strategy.map(STRATEGY).fillna("Not reported")
    return st, fm, fm.groupby("study").strategy.agg(lambda x: sorted(set(x))).reset_index()


def same_setting():
    """Parameter-efficiently tuned models compared with the same base model fully fine-tuned on the same dataset and metric:
    the PEFT models, the matching (study, dataset, metric) pairs, and the full comparison table of the one such study."""
    m, r = load("models"), load("results")
    m["study"] = m.paper_id.map(study_of())
    peft = m[m.training_strategy.astype(str).str.startswith("peft")]
    pairs = []
    for pid, g in m[m.paper_id.isin(peft.paper_id)].groupby("paper_id"):
        pe = g[g.training_strategy.astype(str).str.startswith("peft")]
        full = g[(g.training_strategy == "full_fine_tuning") & g.backbone.isin(pe.backbone)]  # the same base model, fully fine-tuned
        rr = r[(r.paper_id == pid) & r.value_std.notna()]
        for (ds, met), x in rr.groupby(["dataset_key", "metric"]):
            a, b = x[x.model.isin(pe["item"])], x[x.model.isin(full["item"])]
            if len(a) and len(b):
                pairs.append(dict(paper_id=pid, study=study_of()[pid], dataset_key=ds, metric=met, peft=";".join(a.model_name.unique()),
                                  peft_best=a.value_std.max(), full=";".join(b.model_name.unique()), full_best=b.value_std.max()))
    pairs = pd.DataFrame(pairs)
    assert pairs.study.nunique() == 1, "more than one same-setting study: extend Table 6, the figure and the macros"
    pid, ds = pairs.paper_id.iloc[0], pairs.dataset_key.iloc[0]
    g = m[(m.paper_id == pid) & (m.training_strategy.astype(str).str.startswith("peft") | m.training_strategy.isin(["full_fine_tuning", "head_only"]))]
    rr = r[(r.paper_id == pid) & (r.dataset_key == ds) & r.metric.isin(["ap50", "ap50_95", "map"]) & r.model.isin(g["item"])]
    t = rr.pivot_table(index="model", columns="metric", values="value_std", aggfunc="first").reset_index()
    strict = "ap50_95" if "ap50_95" in t.columns else "map"  # the stricter metric, under the name the study gives it
    t["strict"], t["strict_metric"] = t[strict], STRICT_LABEL[strict]
    t = t.merge(g.rename(columns={"item": "model"})[["model", "model_name", "training_strategy", "trainable_params_pct"]], on="model")
    t["method"] = t.model_name.str.split(" with ", n=1).str[-1].str.replace(" fine-tuning", "", regex=False)
    first = rr.drop_duplicates("model").set_index("model")
    t["page"], t["location"] = t.model.map(first.page), t.model.map(first.location)
    t["paper_id"], t["dataset_key"] = pid, ds
    return peft, pairs, t.sort_values(["trainable_params_pct", "ap50"], ascending=[False, False])


def units(ax, st):
    y = 0.0
    for k in FAMILIES:
        sub = st[st.family == k].assign(o=lambda d: d.strategy.map(STRATEGIES.index)).sort_values(["o", "label"])
        if sub.empty:
            continue
        lines = -(-len(sub) // PER_LINE)
        for j, s_ in enumerate(sub.strategy):
            row, col = divmod(j, PER_LINE)
            ax.add_patch(Rectangle((col, -(y + row)), 0.84, 0.84, fc=SHADE[s_], ec="white", lw=0.4))
        mid = -(y + (lines - 1) / 2) + 0.42
        ax.text(-0.5, mid, f"{FAMILY[k]} ({len(sub)})", ha="right", va="center", fontsize=9.5, color=COLOR[k], fontweight="bold")
        stated = int((sub.strategy != "Not reported").sum())
        ax.text(PER_LINE + 0.3, mid, f"stated: {stated} of {len(sub)}", ha="left", va="center", fontsize=9, color=INK, style="italic")
        y += lines + 0.55
    ax.set_xlim(-0.3, PER_LINE + 4.6)
    ax.set_ylim(-y + 0.4, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")


def tradeoff(ax, t):
    """One row per adaptation strategy, ordered by the share of parameters trained; each row joins the strategy's stricter
    metric (open) and AP50 (filled) as a share of full fine-tuning, and prints the share trained on the right."""
    full = t[t.training_strategy == "full_fine_tuning"].iloc[0]
    rest = t[t.training_strategy != "full_fine_tuning"].copy()
    rest["r50"], rest["r95"] = 100 * rest.ap50 / full.ap50, 100 * rest.strict / full.strict
    rest = rest.sort_values(["trainable_params_pct", "ap50"], ascending=[False, False]).reset_index(drop=True)
    lo = np.floor(min(rest.r50.min(), rest.r95.min()) - 1)
    ax.axvline(100, color=SHADE["Full fine-tuning"], ls="--", lw=1.2, zorder=1)
    ax.text(99.85, 0.75, f"full fine-tuning = 100%\n(AP50 {full.ap50:.1f}, {full.strict_metric} {full.strict:.1f})", ha="right", va="bottom", fontsize=8.5,
            color=SHADE["Full fine-tuning"])
    names = []
    for i, p in enumerate(rest.itertuples()):
        y = -i
        col = SHADE["Head only (frozen backbone)"] if p.training_strategy == "head_only" else "#5E86B5"
        ax.axhline(y, color="#F2F2F2", lw=0.8, zorder=0)
        ax.plot([p.r95, p.r50], [y, y], color=col, lw=2.2, alpha=0.55, zorder=2, solid_capstyle="round")
        ax.scatter(p.r50, y, s=58, color=col, edgecolor=INK, linewidth=0.5, zorder=3)
        ax.scatter(p.r95, y, s=58, facecolor="white", edgecolor=col, linewidth=1.4, zorder=3)
        ax.text(101.15, y, f"{p.trainable_params_pct:.2f}", ha="right", va="center", fontsize=8.5, color=INK)
        name = p.method.replace(" (proposed PEFT)", " (proposed)")
        names.append(name + " (frozen backbone)" if p.training_strategy == "head_only" else name)
    ax.text(101.15, 0.75, "trained\n(%)", ha="right", va="bottom", fontsize=8.5, color=INK, style="italic")
    ax.set_yticks([-i for i in range(len(rest))])
    ax.set_yticklabels(names, fontsize=8.8)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-len(rest) + 0.4, 1.7)
    ax.set_xlim(lo, 101.3)
    ax.xaxis.grid(True, color="#F0F0F0", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Value as a share of full fine-tuning (%)")


def main():
    st, fm, fm_st = tables()
    write(st[["study", "label", "family", "basis", "principal_model", "principal_strategy", "strategy"]], "rq3_strategies")
    write(fm[["study", "paper_id", "item", "model_name", "architecture_category", "training_strategy", "strategy"]], "rq3_foundation_models")
    _, _, t = same_setting()
    write(t, "rq3_adaptation")
    d = load("datasets").query("paper_id == @t.paper_id.iloc[0] and dataset_key == @t.dataset_key.iloc[0]").iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.0), gridspec_kw={"width_ratios": [1.75, 1], "wspace": 0.12})
    units(ax1, st)
    stated = int((st.strategy != "Not reported").sum())
    ax1.text(0.0, 1.0, f"Training of the principal transformer is stated for {stated} of {len(st)} studies ({100 * stated / len(st):.0f}%)",
             transform=ax1.transAxes, fontsize=9.5, fontweight="bold", color=INK, va="bottom")
    tradeoff(ax2, t)
    ax2.set_title(f"Single study: {labels()[t.paper_id.iloc[0]]}, Grounding DINO (Swin-T), {int(d.n_images)} UAV images", fontsize=9.5, loc="left",
                  color=INK)
    bottom = panel_titles(fig, [(ax1, "(a) Training of each study's principal transformer\n(one square per study)"),
                                (ax2, "(b) Adaptation strategies in the one study that\ncompares them on the same model and data")])
    marks = [Line2D([], [], ls="", marker="o", color="#5E86B5", markeredgecolor=INK, label="AP50"),
             Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor="#5E86B5", markeredgewidth=1.3,
                    label=t.strict_metric.iloc[0])]
    legend_row(fig, [([Patch(color=SHADE[s_], label=s_) for s_ in STRATEGIES], "Training or adaptation (colour)", 3),
                     (marks, "Metric in panel (b)", 1)], y=bottom - 0.03)
    save(fig, "rq3_training")
    print(f"  principal transformer training stated for {stated} of {len(st)} studies; same-setting comparison: {len(t)} strategies "
          f"in {labels()[t.paper_id.iloc[0]]}")


if __name__ == "__main__":
    set_style()
    main()
