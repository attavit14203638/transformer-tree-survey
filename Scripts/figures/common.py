"""Shared look and data of the manuscript figures. Every figure script imports this module.

Data come only from Literature_Extraction/Extraction/compiled/*.csv. Each figure writes the values it draws, with the paper,
result item and PDF page of every value, to Literature_Extraction/Analysis/figure_data (internal tables), and the figure itself
to Literature_Extraction/Analysis/figures as a 300-dpi JPEG preview (make_figures.py --submission adds the journal JPEGs). Figures never show project identifiers: papers are named
"First author et al. (year)" from the manuscript bibliography (a, b, c when two studies share first author and year).

Counting rules shared by the figures (to be stated in the captions):
- The unit is the study; reports of one study (study_group in papers.csv) count once. Comparisons and transfer pairs are
  points within studies, and every summary (median, interquartile range, counts of outcomes) uses one value per study and group.
- Architecture family of a study: the family of the transformer the study proposes; if it proposes none, the family of its
  best-scoring transformer on its preferred metric; if no transformer has a comparable score, the most frequent family among
  the transformers it uses.
- Values from an unstated data split, values a paper quotes from earlier work and duplicates are excluded by compile.py.
- Differences and changes are in percentage points and use bounded metrics reported in percent.
- Outliers are values outside Tukey's fences (more than 1.5 interquartile ranges beyond the quartiles) of the pooled points.
"""
import csv
import io
import os
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", str(paths.CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(paths.CACHE / "fontconfig"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ------------------------------------------------------------------------------------------------------------------ style
FAMILIES = ["pure_vit", "hierarchical_vit", "cnn_transformer_hybrid", "foundation_model", "vision_language_model"]
FAMILY = {"pure_vit": "Pure ViT", "hierarchical_vit": "Hierarchical ViT", "cnn_transformer_hybrid": "CNN-transformer hybrid",
          "foundation_model": "Foundation model", "vision_language_model": "Vision-language model"}
FAMILY_SHORT = {"pure_vit": "Pure\nViT", "hierarchical_vit": "Hierarchical\nViT", "cnn_transformer_hybrid": "CNN-transformer\nhybrid",
                "foundation_model": "Foundation\nmodel", "vision_language_model": "Vision-language\nmodel"}
# Hues of the original submission (blue, green, orange, red, purple) on the colour-blind-safe Okabe-Ito palette
COLOR = {"pure_vit": "#0072B2", "hierarchical_vit": "#009E73", "cnn_transformer_hybrid": "#E69F00", "foundation_model": "#D55E00",
         "vision_language_model": "#CC79A7"}
GREY, LIGHT, INK, GRID = "#8C8C8C", "#D0D0D0", "#333333", "#E8E8E8"
METRIC = {"ap50": "AP50", "ap50_95": "AP50:95", "map": "mAP", "f1": "F1", "iou": "IoU", "miou": "mIoU", "oa": "OA", "kappa": "kappa",
          "precision": "precision", "recall": "recall", "r2": "R²", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE", "other": "score"}
MARKER = {"AP50": "o", "AP50:95": "H", "mAP": "P", "F1": "^", "IoU": "s", "mIoU": "p", "OA": "D", "kappa": "d", "precision": "X", "recall": "*"}
TIE_PP = 1.0
TASK_GROUP = {"detection": "Detection and counting", "counting": "Detection and counting", "crown_segmentation": "Crown segmentation",
              "tree_segmentation_3d": "3D structure and attributes", "component_segmentation": "3D structure and attributes",
              "attribute_estimation": "3D structure and attributes", "species_classification": "Species classification",
              "health_classification": "Health and mortality", "mortality_detection": "Health and mortality"}
TASKS = ["Detection and counting", "Crown segmentation", "3D structure and attributes", "Species classification", "Health and mortality"]


def set_style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 10,
                         "axes.labelsize": 10.5, "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
                         "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK, "axes.linewidth": 0.8, "figure.facecolor": "white"})


def panel_titles(fig, pairs, pad=0.025, size=11.5, x=None):
    """Panel titles on one level under a multi-panel figure: below the lowest element of any panel, each centred under its
    panel, or left-aligned at figure position x (for stacked panels, so that their titles line up). pairs: list of
    (axes, title). Returns the figure y of the bottom of the titles, for the legends underneath."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    y = min(ax.get_tightbbox(renderer).transformed(to_fig).y0 for ax, _ in pairs) - pad
    texts = []
    for ax, text in pairs:
        pos = ax.get_position()
        tx, ha = ((pos.x0 + pos.x1) / 2, "center") if x is None else (x, "left")
        texts.append(fig.text(tx, y, text, ha=ha, va="top", fontsize=size, fontweight="bold"))
    fig.canvas.draw()
    return min(t.get_window_extent(renderer).transformed(to_fig).y0 for t in texts)


def legend(where, handles, title, **kw):
    kw = {"frameon": True, "edgecolor": "#CCCCCC", "fontsize": 8.5, "title_fontsize": 9, **kw}
    lg = where.legend(handles=handles, title=title, **kw)
    lg.get_title().set_fontweight("bold")
    return lg


PREVIEW_DPI = 300  # JPEG previews for viewing and drafting
SUBMISSION = False  # set by make_figures.py --submission
MIN_WIDTH_PX = 7480  # journal minimum for line art at full page width (1000 dpi at 190 mm); data figures are made 7600 px wide
PAD_FRACTION = 0.011  # uniform white margin on all four sides, as a share of the content width (about 2 mm at full page width)
JPEG_KW = {"quality": 90, "subsampling": 0, "optimize": True}  # high quality, full colour resolution for crisp text, lean files


def submission_dir():
    out = paths.ANALYSIS_FIGURES / "submission"
    out.mkdir(parents=True, exist_ok=True)
    return out


def finish_jpeg(img, path, dpi=None):
    """Trim the white border of an image, add a uniform white margin, and save it as a high-quality JPEG."""
    from PIL import Image
    img = img.convert("RGB")
    a = np.asarray(img)
    ys, xs = np.where((a < 245).any(axis=2))
    img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    pad = max(12, round(PAD_FRACTION * img.width))
    out = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), "white")
    out.paste(img, (pad, pad))
    out.save(path, "JPEG", dpi=(dpi, dpi) if dpi else (300, 300), **JPEG_KW)
    return out.size


def render(fig, dpi):
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    buf.seek(0)
    return Image.open(buf)


def save(fig, name):
    """A JPEG preview (300 dpi) in Analysis/figures; with --submission also the journal file, a JPEG at least 7600 px wide, in
    Analysis/figures/submission, copied to the manuscript figures/ folder. Both have the same uniform margin."""
    paths.ANALYSIS_FIGURES.mkdir(parents=True, exist_ok=True)
    finish_jpeg(render(fig, PREVIEW_DPI), paths.ANALYSIS_FIGURES / f"{name}.jpg", PREVIEW_DPI)
    if SUBMISSION:
        width_in = fig.get_tightbbox(fig.canvas.get_renderer()).width
        dpi = int(np.ceil(7600 / width_in))
        out = submission_dir() / f"{name}.jpg"
        size = finish_jpeg(render(fig, dpi), out, dpi)
        shutil.copy2(out, paths.PAPER_FIGURES / f"{name}.jpg")  # the manuscript always has the current version
        print(f"  submission figures/{name}.jpg: {size[0]} x {size[1]} px")
    plt.close(fig)
    print(f"  saved figures/{name}.jpg")


def pretty(v, table=None):
    table = table or {}
    return table.get(v, str(v).replace("_", " ").capitalize())


# ------------------------------------------------------------------------------------------------------------------- data
_cache = {}


def load(name):
    key = f"compiled/{name}"
    if key not in _cache:
        _cache[key] = pd.read_csv(paths.COMPILED / f"{name}.csv")
    return _cache[key].copy()


def write(df, name):
    paths.ANALYSIS_DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.ANALYSIS_DATA / f"{name}.csv", index=False)
    return df


def papers():
    if "papers" not in _cache:
        with open(paths.PAPERS_CSV, encoding="utf-8") as f:
            _cache["papers"] = {r["paper_id"]: r for r in csv.DictReader(f) if r["paper_id"] and r["ft_decision"] == "Include"}
    return _cache["papers"]


def study_of():
    """paper ID -> study ID (the study_group when a study has several reports)."""
    return {pid: (p["study_group"] or pid) for pid, p in papers().items()}


def _norm(t):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", t)


def bib_entries():
    """Fields of every entry of the manuscript bibliography, by key."""
    if "bib" not in _cache:
        text = (paths.PAPER_BIB / "Systematic_Review.bib").read_text(encoding="utf-8")
        out = {}
        for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
            out[m.group(1)] = dict((k.lower(), v) for k, v in re.findall(r"\n\s*(\w+) = \{(.*)\},", m.group(2)))
        _cache["bib"] = out
    return _cache["bib"]


def bib_key():
    """paper ID -> bibliography key (matched by DOI, then by title)."""
    if "bibkey" not in _cache:
        by_doi, by_title = {}, {}
        for k, f in bib_entries().items():
            if f.get("doi"):
                by_doi[f["doi"].lower()] = k
            by_title[_norm(f.get("title", "").replace("{", "").replace("}", ""))[:60]] = k
        _cache["bibkey"] = {pid: (by_doi.get(p["doi"].lower()) if p["doi"] else None) or by_title.get(_norm(p["title"])[:60])
                            for pid, p in papers().items()}
    return _cache["bibkey"]


NUMBERED = False  # set by make_figures.py --numbered: "First author et al. [n]" with n from the compiled manuscript


def reference_numbers():
    """Reference number of every cited bib key, read from the compiled manuscript (out/main.aux)."""
    if not paths.PAPER_AUX.exists():  # the public repository: the numbers exported from the compiled manuscript
        if paths.PAPER_CITATIONS.exists():
            return {r["bib_key"]: int(r["number"]) for r in csv.DictReader(open(paths.PAPER_CITATIONS, encoding="utf-8"))}
        return {}
    text = paths.PAPER_AUX.read_text(encoding="utf-8", errors="ignore")
    return {k: int(n) for k, n in re.findall(r"\\bibcite\{([^}]+)\}\{\{(\d+)\}", text)}


def labels():
    """'First author et al. (year)' for every included paper, from the manuscript bibliography; with NUMBERED,
    'First author et al. [n]' using the reference numbers of the compiled manuscript (author-year where a paper is not cited)."""
    if "labels" in _cache:
        return _cache["labels"]
    out, who_of = {}, {}
    for pid, key in bib_key().items():
        f = bib_entries().get(key, {})
        names = [a.split(",")[0].replace("{", "").replace("}", "").strip() for a in f.get("author", "").split(" and ") if a.strip()]
        if names:
            who_of[pid] = names[0] if len(names) == 1 else f"{names[0]} and {names[1]}" if len(names) == 2 else f"{names[0]} et al."
            out[pid] = f"{who_of[pid]} ({f.get('year', '')})"
        else:
            out[pid] = f"{papers()[pid]['title'][:30]}… ({papers()[pid]['year']})"
    groups = defaultdict(list)
    for pid, lab in out.items():
        groups[lab].append(pid)
    for lab, pids in groups.items():
        if len({study_of()[q] for q in pids}) > 1:  # different studies with the same first author and year: 2025a, 2025b
            for i, q in enumerate(sorted(pids, key=lambda q: papers()[q]["title"].lower())):
                out[q] = lab[:-1] + "abcdefghij"[i] + ")"
    if NUMBERED:
        num, missing = reference_numbers(), []
        for pid, key in bib_key().items():
            if key in num and pid in who_of:
                out[pid] = f"{who_of[pid]} [{num[key]}]"
            else:
                missing.append(key)
        if missing:
            print(f"  warning: {len(missing)} included papers are not cited in the compiled manuscript; their labels stay author-year: "
                  + ", ".join(sorted(missing)[:8]) + (" ..." if len(missing) > 8 else ""))
    _cache["labels"] = out
    return out


def studies():
    """One row per study: year, architecture family (rule above), the basis of that family and the principal transformer."""
    if "studies" in _cache:
        return _cache["studies"].copy()
    sof = study_of()
    m = load("models")
    r = load("results")
    tr = m[m.transformer == True]  # noqa: E712
    rows = []
    for st, pids in pd.Series(list(sof.keys()), index=list(sof.values())).groupby(level=0):
        pids = list(pids)
        mt = tr[tr.paper_id.isin(pids)]
        prop = mt[mt.model_role == "proposed"]
        if len(prop):
            fam = max(FAMILIES, key=lambda k: ((prop.architecture_category == k).sum(), -FAMILIES.index(k)))
            pm, basis = prop[prop.architecture_category == fam].iloc[0], "proposed transformer"
        else:
            rr = r[r.paper_id.isin(pids) & (r.transformer == True) & (r.primary == True) & (r.canonical_metric == True)]  # noqa: E712
            rr = rr[rr.value_std.notna()]
            if len(rr):
                lower = rr.metric.isin(["mae", "rmse", "mape"])
                best = (rr[~lower].sort_values("value_std", ascending=False) if (~lower).any() else rr.sort_values("value_std")).iloc[0]
                pm, basis = mt[(mt.paper_id == best.paper_id) & (mt.item == best.model)].iloc[0], "best-scoring transformer (baseline only)"
            else:
                fam = max(FAMILIES, key=lambda k: ((mt.architecture_category == k).sum(), -FAMILIES.index(k)))
                pm, basis = mt[mt.architecture_category == fam].iloc[0], "transformer used, no comparable score"
        rows.append(dict(study=st, papers=";".join(pids), label=labels()[pids[0]], bib_key=bib_key()[pids[0]],
                         year=min(int(papers()[p]["year"]) for p in pids), family=pm.architecture_category, basis=basis,
                         principal_model=pm.model_name, principal_backbone=pm.backbone, principal_strategy=pm.training_strategy,
                         principal_pretraining=pm.pretraining))
    df = pd.DataFrame(rows).sort_values(["year", "study"]).reset_index(drop=True)
    _cache["studies"] = df
    return df.copy()


def study_values(col):
    """Study-level sets of a multi-valued field of compiled/studies.csv (the values of all reports of a study)."""
    s = load("studies")
    s = s[s.extracted == True]  # noqa: E712
    s["study"] = s.paper_id.map(study_of())
    out = defaultdict(set)
    for st, v in zip(s.study, s[col]):
        for x in str(v if pd.notna(v) else "not_reported").split(";"):
            out[st].add(x.strip())
    return out


def context(df, ref_col):
    """Model role, training strategy, pre-training and setting of the result behind each reference (paper:item)."""
    r = load("results")
    m = load("models").rename(columns={"item": "model"})
    ref = r.assign(ref=r.paper_id + ":" + r["item"])[["ref", "paper_id", "model", "setting"]]
    return df[[ref_col]].merge(ref, left_on=ref_col, right_on="ref", how="left").merge(
        m[["paper_id", "model", "model_role", "training_strategy", "pretraining"]], on=["paper_id", "model"], how="left")


# --------------------------------------------------------------------------------------------------- summaries and panels
def tukey(v):
    q1, q3 = np.percentile(v, [25, 75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def outcome(v):
    return "tie" if abs(v) <= TIE_PP else ("up" if v > 0 else "down")


def per_study(df, group, value):
    """One value per study and group (median of its points); quartiles and outcome counts of those values per group."""
    g = df.groupby(["study", group], as_index=False)[value].median()
    out = []
    for k, x in g.groupby(group):
        q1, md, q3 = np.percentile(x[value], [25, 50, 75])
        o = x[value].map(outcome)
        out.append(dict(group=k, studies=len(x), points=int((df[group] == k).sum()), q1=round(q1, 2), median=round(md, 2), q3=round(q3, 2),
                        up=int((o == "up").sum()), tie=int((o == "tie").sum()), down=int((o == "down").sum())))
    return pd.DataFrame(out)


def points_panel(ax, df, group, order, ticklabels, value, ylim, box_colour=None, label_outliers=False, rng=None, marker_of=None):
    """Points (colour = family; shape = metric unless marker_of says otherwise), per-study interquartile box and median, and a
    count box per group."""
    rng = rng or np.random.default_rng(2026)
    marker_of = marker_of or (lambda p: MARKER.get(p["metric_label"], "o"))
    summ = per_study(df, group, value)
    for i, g in enumerate(order):
        pts = df[df[group] == g]
        if pts.empty:
            continue
        s = summ[summ.group == g].iloc[0]
        bc = box_colour(g) if callable(box_colour) else (box_colour or INK)
        if s.studies >= 3:
            ax.add_patch(plt.Rectangle((i - 0.3, s.q1), 0.6, s.q3 - s.q1, fc=bc, alpha=0.16, ec=bc, lw=1.1, zorder=2))
        ax.plot([i - 0.3, i + 0.3], [s["median"], s["median"]], color=bc, lw=2.6, zorder=4, solid_capstyle="butt")
        xs = i + rng.uniform(-0.23, 0.23, len(pts))
        out_ = pts[pts.outlier] if label_outliers else pts.iloc[0:0]
        named = set(out_.loc[out_[value].abs().sort_values(ascending=False).index].drop_duplicates("study").index)
        for x, (ix, p) in zip(xs, pts.iterrows()):
            ax.scatter(x, min(max(p[value], ylim[0] + 1), ylim[1] - 1), s=40, marker=marker_of(p), color=COLOR.get(p["colour_key"], GREY),
                       edgecolor="white", linewidth=0.6, alpha=0.92, zorder=5)
            if ix in named:
                ax.annotate(p["label"], (x, p[value]), xytext=(6, 0), textcoords="offset points", fontsize=7, color=INK, va="center")
        ax.text(i, ylim[1] - 1.5, f"n = {s.studies}\nmedian {s['median']:+.1f}\n↑{s.up}  ={s.tie}  ↓{s.down}", ha="center", va="top",
                fontsize=7.8, color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=bc, lw=0.9))
    ax.axhline(0, color=INK, ls="--", lw=0.9, zorder=1)
    ax.axhspan(-TIE_PP, TIE_PP, color=GRID, zorder=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([ticklabels.get(g, g) for g in order], fontsize=8.8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    return summ


def legend_row(fig, specs, y=-0.075, gap=0.025):
    """Legends side by side under the figure, centred as a group with equal gaps between them.

    specs: list of (handles, title, ncol) or (handles, title, ncol, extra legend keywords). The legends are drawn once to measure
    their widths, then placed."""
    made = [legend(fig, sp[0], sp[1], loc="upper left", bbox_to_anchor=(0, y), ncol=sp[2], **(sp[3] if len(sp) > 3 else {})) for sp in specs]
    fig.canvas.draw()
    widths = [lg.get_window_extent().width / fig.bbox.width for lg in made]
    x = 0.5 - (sum(widths) + gap * (len(made) - 1)) / 2
    for lg, w in zip(made, widths):
        lg.set_bbox_to_anchor((x, y), transform=fig.transFigure)
        x += w + gap
    fig.canvas.draw()
    return min(lg.get_window_extent().transformed(fig.transFigure.inverted()).y0 for lg in made)


def box_handles(counts):
    """Legend entries that explain the per-study boxes, the tie zone and the three counts (counts: three phrases)."""
    return [Patch(fc=INK, alpha=0.16, ec=INK, label="interquartile range of the per-study values"),
            Line2D([], [], color=INK, lw=2.6, label="median of the per-study values"),
            Patch(color=GRID, label="tie zone (within ±1 pp)"),
            Line2D([], [], ls="", marker=r"$\uparrow$", color=INK, markersize=8, label=f"studies whose value {counts[0]}"),
            Line2D([], [], ls="", marker=r"$=$", color=INK, markersize=8, label=f"studies whose value {counts[1]}"),
            Line2D([], [], ls="", marker=r"$\downarrow$", color=INK, markersize=8, label=f"studies whose value {counts[2]}")]


def family_handles(families, extra_colour=None):
    return [Patch(color=COLOR[k], label=FAMILY[k]) for k in families] + ([Patch(color=GREY, label=extra_colour)] if extra_colour else [])
