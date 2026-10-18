"""Overview of the corpus as an evidence map: for every task, the studies by platform and sensor (a) and by tree setting (b).

Each cell is a pie whose area is the number of studies with that task and that platform-sensor combination (or setting), split
by architecture family. A study with several tasks, platforms, sensors or settings appears in every cell it belongs to; the bars
give the number of distinct studies per column and per task. counts() also gives the corpus statistics quoted in the text.
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Wedge
from matplotlib.transforms import blended_transform_factory

from common import (COLOR, FAMILIES, GREY, GRID, INK, TASK_GROUP, TASKS, family_handles, legend, panel_titles, pretty, save, set_style,
                    studies, study_values, write)

DIVIDER = "#D4D4D4"  # thin lines between platform groups

LABELS = {"uav": "UAV", "rgb": "RGB", "lidar": "LiDAR", "dsm": "DSM (surface model)", "ground_camera": "Ground camera",
          "mobile_lidar": "Mobile LiDAR", "terrestrial_lidar": "Terrestrial LiDAR", "not_reported": "Not reported", "polygon_mask": "Polygon or mask",
          "bounding_box": "Bounding box", "point_label": "Point-cloud labels", "image_label": "Image-level label",
          "regression_target": "Measured attribute", "point": "Tree point (dot)", "tree_instance": "Tree instance (box)",
          "crown_mask": "Crown mask", "tree_label": "Tree class label", "tree_attribute": "Tree attribute", "tree_component": "Tree component",
          "tree_points": "Tree point set", "tree_count": "Tree count", "natural_forest": "Natural forest"}
PANELS = [("task", "(a) Task"), ("output_unit", "(b) Output that is scored"), ("annotation_type", "(c) Annotation"),
          ("platform", "(d) Platform"), ("sensors", "(e) Sensor"), ("tree_setting", "(f) Tree setting")]


def counts():
    """Studies per value of each field (a study with several values counts under each)."""
    n = len(studies())
    rows = []
    for col, _ in PANELS:
        fmt = (lambda v: TASK_GROUP.get(v, pretty(v, LABELS))) if col == "task" else (lambda v: pretty(v, LABELS))
        cnt = Counter(v for vals in study_values(col).values() for v in {fmt(x) for x in vals})
        rows += [dict(field=col, value=k, studies=v, percent=round(100 * v / n, 1))
                 for k, v in sorted(cnt.items(), key=lambda t: (t[0] == "Not reported", -t[1]))]
    return pd.DataFrame(rows), n


PLATFORM_GROUP = {"uav": "UAV", "airborne": "Airborne", "satellite": "Satellite", "ground_camera": "Ground camera",
                  "terrestrial_lidar": "TLS or MLS", "mobile_lidar": "TLS or MLS", "multiple": "Several platforms", "synthetic": "Synthetic"}
SENSOR_GROUP = {"rgb": "RGB", "multispectral": "Spectral", "hyperspectral": "Spectral", "lidar": "LiDAR", "dsm": "DSM"}
PLATFORMS = ["UAV", "Airborne", "Satellite", "Ground camera", "TLS or MLS", "Several platforms", "Synthetic"]
SENSORS = ["RGB", "Spectral", "LiDAR", "DSM"]
SETTING_GROUP = {"natural_forest": "Natural forest", "mixed": "Mixed", "plantation": "Plantation", "urban": "Urban", "orchard": "Orchard",
                 "agroforestry": "Other", "mangrove": "Other", "synthetic": "Other", "not_reported": "Not reported"}
SETTINGS = ["Natural forest", "Mixed", "Plantation", "Urban", "Orchard", "Other", "Not reported"]


def evidence():
    """Study-level sets used by the evidence map: tasks, acquisition (platform x sensor), tree settings and family."""
    st = studies().set_index("study")
    task = study_values("task")
    plat, sens, sett = study_values("platform"), study_values("sensors"), study_values("tree_setting")
    rows = []
    for s_ in st.index:
        rows.append(dict(study=s_, family=st.loc[s_, "family"],
                         tasks={TASK_GROUP[v] for v in task[s_] if v in TASK_GROUP},
                         acquisition={(PLATFORM_GROUP[p], SENSOR_GROUP[q]) for p in plat[s_] if p in PLATFORM_GROUP
                                      for q in sens[s_] if q in SENSOR_GROUP},
                         settings={SETTING_GROUP.get(v, "Other") for v in sett[s_]}))
    return pd.DataFrame(rows)


def cells(ev, field, columns):
    """Studies per (task, column), with the family mix of each cell."""
    out = []
    for t in TASKS:
        for c in columns:
            sub = ev[ev.tasks.map(lambda ts: t in ts) & ev[field].map(lambda vs: c in vs)]
            if len(sub):
                fam = sub.family.value_counts()
                out.append(dict(task=t, column=c if isinstance(c, str) else " / ".join(c), studies=len(sub),
                                **{f"n_{k}": int(fam.get(k, 0)) for k in FAMILIES}))
    return pd.DataFrame(out)


def radius(n, nmax):
    return 0.48 * np.sqrt(n / nmax)  # area proportional to the number of studies


def pie(ax, x, y, r, fam):
    start, total = 90.0, sum(fam.values())
    for k in FAMILIES:
        if fam.get(k):
            sweep = 360.0 * fam[k] / total
            ax.add_patch(Wedge((x, y), r, start - sweep, start, fc=COLOR[k], ec="white", lw=0.6, zorder=3))
            start -= sweep


def matrix(ax, ev, field, columns, ticks, nmax, row_totals=False, partitions=(), sp=1.0):
    """One panel of the evidence map: pie bubbles per cell, distinct studies per column on top (and per row on the right).
    partitions: x positions of thin dividers between column groups (between platforms in panel a). sp: distance between
    columns in cell units (pie sizes do not change, so both panels stay comparable)."""
    for x in partitions:
        ax.plot([x, x], [-len(TASKS) + 0.5, 2.15], color=DIVIDER, lw=0.7, zorder=0, solid_capstyle="butt")
    for i, t in enumerate(TASKS):
        ax.axhline(-i, color=GRID, lw=0.6, zorder=0)
        for j, c in enumerate(columns):
            sub = ev[ev.tasks.map(lambda ts: t in ts) & ev[field].map(lambda vs: c in vs)]
            if sub.empty:
                continue
            r = radius(len(sub), nmax)
            pie(ax, j * sp, -i, r, sub.family.value_counts().to_dict())
            if r >= 0.3:
                ax.text(j * sp, -i, str(len(sub)), ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold", zorder=4,
                        path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])
            else:
                ax.text(j * sp + r + 0.05, -i + r * 0.55, str(len(sub)), ha="left", va="center", fontsize=8.5, color=INK, zorder=4)
    col_tot = [int(ev[field].map(lambda vs: c in vs).sum()) for c in columns]
    for j, v in enumerate(col_tot):
        h = 1.3 * v / max(col_tot)
        ax.add_patch(Rectangle((j * sp - 0.3, 0.75), 0.6, h, fc=GREY, ec="none", zorder=2))
        ax.text(j * sp, 0.8 + h, str(v), ha="center", va="bottom", fontsize=8.5, color=INK)
    last = (len(columns) - 1) * sp
    right = last + 0.5
    if row_totals:
        row_tot = [int(ev.tasks.map(lambda ts: t in ts).sum()) for t in TASKS]
        for i, v in enumerate(row_tot):
            w = 2.4 * v / max(row_tot)
            ax.add_patch(Rectangle((last + 0.9, -i - 0.22), w, 0.44, fc=GREY, ec="none"))
            ax.text(last + 0.95 + w, -i, f" {v}", ha="left", va="center", fontsize=9, color=INK)
        right = last + 3.9
        ax.text(last + 2.1, 0.95, "studies\nper task", ha="center", va="bottom", fontsize=9, color=INK, style="italic")
    ax.set_xlim(-0.6, right)
    ax.set_ylim(-len(TASKS) + 0.45, 2.45)
    ax.set_aspect("equal")
    ax.set_xticks([j * sp for j in range(len(columns))])
    ax.set_xticklabels(ticks, fontsize=9)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_bounds(-0.5, last + 0.5)
    ax.tick_params(axis="x", length=0)


def main():
    """Panel (a) spans the full width; panel (b) sits below it at the same scale, with the legends and the note beside it."""
    tab, n = counts()
    write(tab, "overview")
    ev = evidence()
    acq = [(p, q) for p in PLATFORMS for q in SENSORS if ev.acquisition.map(lambda vs: (p, q) in vs).any()]
    sets = [c for c in SETTINGS if ev.settings.map(lambda vs: c in vs).any()]
    ca, cb = cells(ev, "acquisition", acq), cells(ev, "settings", sets)
    write(pd.concat([ca.assign(panel="a"), cb.assign(panel="b")]), "overview_evidence_map")
    nmax = max(ca.studies.max(), cb.studies.max())
    W, H, label_in, SP_B = 12.5, 11.5, 2.2, 1.3  # inches: figure width, height, room for the task labels; column spacing in (b)
    unit = (W - label_in - 0.15) / (len(acq) + 0.1)  # inches per cell; both panels use the same scale
    rows = len(TASKS) + 2.0  # data height of a panel: the task rows and the bars on top

    def panel(width_units, top_in):
        return fig.add_axes([label_in / W, (H - top_in - rows * unit) / H, width_units * unit / W, rows * unit / H])

    def task_labels(ax):
        for i, t in enumerate(TASKS):
            ax.text(-0.75, -i, t, ha="right", va="center", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(-0.5, 1.55, "distinct studies", ha="right", va="center", fontsize=9, color=INK, style="italic")

    fig = plt.figure(figsize=(W, H))
    ax1 = panel(len(acq) + 0.1, 0.05)
    bounds = [j + 0.5 for j in range(len(acq) - 1) if acq[j][0] != acq[j + 1][0]]  # between platform groups
    matrix(ax1, ev, "acquisition", acq, [q for _, q in acq], nmax, partitions=bounds)
    task_labels(ax1)
    for p in PLATFORMS:  # platform groups under their sensor columns
        idx = [j for j, (pp, _) in enumerate(acq) if pp == p]
        if idx:
            ax1.annotate("", xy=(idx[0] - 0.4, -len(TASKS) + 0.05), xytext=(idx[-1] + 0.4, -len(TASKS) + 0.05), xycoords="data",
                         arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
            ax1.text((idx[0] + idx[-1]) / 2, -0.11, p.replace(" ", "\n", 1) if p in ("Ground camera", "Several platforms") else p,
                     transform=blended_transform_factory(ax1.transData, ax1.transAxes), ha="center", va="top", fontsize=10,
                     fontweight="bold")
    fig.canvas.draw()
    x_title = ax1.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x0  # left edge of the task labels
    low_a = panel_titles(fig, [(ax1, "(a) Task by platform and sensor")], size=12, x=x_title)
    ax2 = panel((len(sets) - 1) * SP_B + 4.5, H * (1 - low_a) + 0.4)
    matrix(ax2, ev, "settings", sets, [c.replace(" ", "\n", 1) if c in ("Natural forest", "Not reported") else c for c in sets], nmax,
           row_totals=True, sp=SP_B)
    task_labels(ax2)
    panel_titles(fig, [(ax2, "(b) Task by tree setting")], size=12, x=x_title)  # both titles start on the same line
    px = ax1.transData.transform((1, 0))[0] - ax1.transData.transform((0, 0))[0]
    diameter = {v: 2 * radius(v, nmax) * px * 72 / fig.dpi for v in (1, 5, 10, 20) if v <= nmax}  # marker diameters in points
    sizes = [Line2D([], [], ls="", marker="o", color=GREY, markersize=d, label=str(v)) for v, d in diameter.items()]
    fs = 9.5
    x_leg = ax2.get_position().x1 + 0.35 / W  # legends and note to the right of panel (b)
    top_b = ax2.get_position().y1
    fam = legend(fig, family_handles([k for k in FAMILIES if (ev.family == k).any()]), "Architecture family (slices)", loc="upper left",
                 bbox_to_anchor=(x_leg, top_b), fontsize=fs, title_fontsize=10)
    fig.canvas.draw()
    y_size = fam.get_window_extent().transformed(fig.transFigure.inverted()).y0 - 0.02
    size = legend(fig, sizes, "Studies in the cell (area)", loc="upper left", bbox_to_anchor=(x_leg, y_size), ncol=2, fontsize=fs,
                  title_fontsize=10, handleheight=max(diameter.values()) / fs + 0.6, handlelength=max(diameter.values()) / fs + 0.2,
                  borderpad=0.7, handletextpad=0.6, columnspacing=1.4)
    fig.add_artist(fam)
    fig.canvas.draw()
    y_note = size.get_window_extent().transformed(fig.transFigure.inverted()).y0 - 0.02
    fig.text(x_leg, y_note, f"n = {len(ev)} studies. A study with several tasks,\nplatforms, sensors or settings appears in\nevery cell it "
             "belongs to; the bars count\ndistinct studies.", ha="left", va="top", fontsize=9.5, style="italic", linespacing=1.35)
    save(fig, "overview")


if __name__ == "__main__":
    set_style()
    main()
