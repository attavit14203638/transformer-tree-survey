"""Timeline: studies per year by architecture family (a) and the yearly count of each family (b).

Curves are monotone cubic (PCHIP) interpolations between the yearly counts, drawn for readability; the dots are the counts.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import COLOR, FAMILIES, FAMILY, GREY, GRID, INK, legend_row, panel_titles, save, set_style, studies, write

YEARS = list(range(2017, 2026))


def smooth(y, grid):
    return np.clip(PchipInterpolator(YEARS, y)(grid), 0, None)


def main():
    st = studies()
    tab = pd.crosstab(st.year, st.family).reindex(index=YEARS, columns=FAMILIES, fill_value=0)
    write(st, "studies")
    write(tab.reset_index(), "timeline")
    grid = np.linspace(2019.8, YEARS[-1], 600)
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(15, 5.9), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.16})

    base = np.zeros_like(grid)
    for k in FAMILIES:
        top = base + smooth(tab[k].to_numpy(), grid)
        ax.fill_between(grid, base, top, color=COLOR[k], alpha=0.88, lw=0)
        ax.plot(grid, top, color="white", lw=0.9)
        base = top
    totals = tab.sum(axis=1)
    ax.plot(grid, base, color=INK, lw=1.4)
    for y, t in totals.items():
        if t:
            ax.scatter(y, t, s=26, color=INK, zorder=5)
            ax.annotate(f"{t}", (y, t), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10, fontweight="bold", color=INK)
    shown = [(y, t) for y, t in totals.items() if t]
    growth = []
    for (y0, t0), (y1, t1) in zip(shown, shown[1:]):  # year-on-year increase, written between the two years
        pct = 100 * (t1 - t0) / t0
        growth.append(dict(from_year=y0, to_year=y1, from_studies=int(t0), to_studies=int(t1), increase_pct=round(pct, 1)))
        ax.annotate(f"+{pct:.0f}%", ((y0 + y1) / 2, (t0 + t1) / 2), xytext=(-14, 14), textcoords="offset points", ha="center",
                    fontsize=9, color=INK, style="italic", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    (y0, t0), (y1, t1) = shown[0], shown[-1]
    total_pct = 100 * (t1 - t0) / t0
    growth.append(dict(from_year=y0, to_year=y1, from_studies=int(t0), to_studies=int(t1), increase_pct=round(total_pct, 1)))
    write(pd.DataFrame(growth), "timeline_growth")
    ax.text(0.01, 0.89, f"{t0} studies in {y0}, {t1} in {y1}: an increase of {total_pct:.0f}%", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color=INK, va="top")
    ax.set_xticks(range(2020, 2026))
    ax.set_xlim(2019.8, YEARS[-1] + 0.3)
    ax.set_ylim(0, totals.max() * 1.15)
    ax.set_ylabel("Number of studies")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.01, 0.96, f"{len(st)} studies; search window 2017 to 2025; no included study before 2022; curves join the yearly counts (dots)",
            transform=ax.transAxes,
            fontsize=9, style="italic", color=INK, va="top")

    for k in FAMILIES:
        y = tab[k].to_numpy()
        axb.plot(grid, smooth(y, grid), color=COLOR[k], lw=2.4)
        shown = [(yr, v) for yr, v in zip(YEARS, y) if yr >= 2021]
        axb.scatter([s[0] for s in shown], [s[1] for s in shown], s=22, color=COLOR[k], edgecolor="white", linewidth=0.8, zorder=5)
    ends = sorted(((tab.loc[YEARS[-1], k], k) for k in FAMILIES), reverse=True)
    placed = []
    for v, k in ends:  # end labels, nudged apart so they do not overlap
        yv = v
        while any(abs(yv - p) < 2.2 for p in placed):
            yv -= 2.2
        placed.append(yv)
        axb.annotate(f"{FAMILY[k]} ({int(tab[k].sum())})", (YEARS[-1], v), xytext=(YEARS[-1] + 0.12, yv), textcoords="data",
                     fontsize=9, color=COLOR[k], fontweight="bold", va="center")
    axb.set_xticks(range(2021, 2026))
    axb.set_xlim(2020.8, 2027.3)
    axb.set_ylim(-1.5, tab.max().max() * 1.12)
    axb.set_ylabel("Number of studies")
    axb.yaxis.grid(True, color=GRID, lw=0.6)
    axb.set_axisbelow(True)
    axb.spines["left"].set_visible(False)
    axb.tick_params(axis="y", length=0)
    axb.text(0.01, 0.96, "label: family (studies in total)", transform=axb.transAxes, fontsize=9, style="italic", color=GREY, va="top")
    bottom = panel_titles(fig, [(ax, "(a) Studies per year, stacked by architecture family"), (axb, "(b) Studies per year in each family, 2021 to 2025")])
    legend_row(fig, [([Patch(color=COLOR[k], label=FAMILY[k]) for k in FAMILIES], "Architecture family", 5)], y=bottom - 0.03)
    save(fig, "timeline")
    print(f"  {len(st)} studies; per year: {totals[totals > 0].to_dict()}; per family: "
          + ", ".join(f"{FAMILY[k]} {int(tab[k].sum())}" for k in FAMILIES))


if __name__ == "__main__":
    set_style()
    main()
