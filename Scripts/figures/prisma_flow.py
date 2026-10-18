"""PRISMA 2020 flow diagram (databases and other methods) from Literature_Extraction/Search/screening/prisma_counts.json."""
import json
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import INK, paths, save, set_style


def main():
    c = json.loads((paths.SEARCH / "screening" / "prisma_counts.json").read_text())
    idf, scr, om, inc = c["identification"], c["screening"], c["other_methods"], c["included"]
    fig, ax = plt.subplots(figsize=(13, 9.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white", bold=False, size=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008", fc=fc, ec=INK, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))

    def n(v):
        return f"{v:,}"

    def reasons(items, width):
        return "\n".join(textwrap.fill(f"{k} (n\u00a0=\u00a0{v})", width) for k, v in sorted(items.items(), key=lambda t: -t[1]))

    DX, DW, SX, SW, OX, OW, PX, PW = 0.05, 0.22, 0.30, 0.30, 0.64, 0.18, 0.845, 0.15
    box(DX, 0.935, SX + SW - DX, 0.05, "Identification of studies via databases", fc="#FCE4C4", bold=True, size=10)
    box(OX, 0.935, PX + PW - OX, 0.05, "Identification of studies via other methods", fc="#E3E3E3", bold=True, size=10)
    for y, h, lab in ((0.75, 0.17, "Identification"), (0.20, 0.53, "Screening"), (0.03, 0.15, "Included")):
        box(0.005, y, 0.03, h, "", fc="#CFE2F3")
        ax.text(0.02, y + h / 2, lab, rotation=90, ha="center", va="center", fontsize=10, fontweight="bold")
    dbs = "\n".join(f"{k} (n = {n(v)})" for k, v in idf["databases"].items())
    box(DX, 0.76, DW, 0.15, f"Records identified from databases\n(n = {n(idf['records_total'])}):\n{dbs}")
    box(SX, 0.79, SW, 0.09, f"Records removed before screening:\nduplicate records (n = {n(idf['duplicates_removed'])})")
    box(DX, 0.62, DW, 0.07, f"Records screened\n(n = {n(scr['records_screened'])})")
    box(SX, 0.62, SW, 0.07, f"Records excluded on title and abstract\n(n = {n(scr['records_excluded'])})")
    box(DX, 0.49, DW, 0.07, f"Reports sought for retrieval\n(n = {n(scr['reports_sought'])})")
    box(SX, 0.49, SW, 0.07, f"Reports not retrieved\n(n = {n(scr['reports_not_retrieved'])})")
    box(DX, 0.36, DW, 0.07, f"Reports assessed for eligibility\n(n = {n(scr['reports_assessed'])})")
    box(SX, 0.21, SW, 0.25, f"Reports excluded (n = {sum(scr['reports_excluded'].values())}):\n{reasons(scr['reports_excluded'], 46)}", size=8.3)
    meth = "\n".join(textwrap.fill(f"{k} (n\u00a0=\u00a0{v})", 26) for k, v in om["records_by_method"].items())
    box(OX, 0.79, OW, 0.09, f"Records identified from:\n{meth}")
    box(OX, 0.49, OW, 0.07, f"Reports sought for retrieval\n(n = {om['reports_sought']})")
    box(PX, 0.49, PW, 0.07, f"Reports not retrieved\n(n = {om['reports_not_retrieved']})", size=8.3)
    box(OX, 0.36, OW, 0.07, f"Reports assessed for\neligibility (n = {om['reports_assessed']})")
    box(PX, 0.21, PW, 0.25, f"Reports excluded\n(n = {sum(om['reports_excluded'].values())}):\n{reasons(om['reports_excluded'], 22)}", size=8)
    box(DX, 0.05, SX + SW - DX, 0.10, f"Studies included in the review (n = {inc['studies']})\nReports of included studies (n = {inc['reports']})",
        bold=True, size=10)
    ax.text(DX + (SX + SW - DX) / 2, 0.06, f"{inc['via_databases']} reports via databases and {inc['via_other_methods']} via other methods",
            ha="center", va="bottom", fontsize=8.5, style="italic")
    cx, ox = DX + DW / 2, OX + OW / 2
    for y1, y2 in ((0.76, 0.69), (0.62, 0.56), (0.49, 0.43), (0.36, 0.15)):
        arrow(cx, y1, cx, y2)
    for y in (0.835, 0.655, 0.525, 0.395):
        arrow(DX + DW, y, SX, y)
    arrow(ox, 0.79, ox, 0.56)
    arrow(ox, 0.49, ox, 0.43)
    arrow(OX + OW, 0.525, PX, 0.525)
    arrow(OX + OW, 0.395, PX, 0.395)
    ax.annotate("", xy=(SX + SW, 0.12), xytext=(ox, 0.36),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1, connectionstyle="angle,angleA=-90,angleB=0"))
    save(fig, "prisma_flow")


if __name__ == "__main__":
    set_style()
    main()
