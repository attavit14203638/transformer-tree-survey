"""Geography: study areas by country (a) and biomes (b). Countries are read from the study-area text recorded for each
paper; a study is placed in every country it names. Hemisphere uses the representative point of each country polygon."""
import re
from collections import Counter

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from common import LIGHT, legend, panel_titles, paths, pretty, save, set_style, studies, study_values, write

ALIAS = {"USA": "United States of America", "United States": "United States of America", "U.S.": "United States of America",
         "UK": "United Kingdom", "England": "United Kingdom", "Scotland": "United Kingdom", "UAE": "United Arab Emirates",
         "Czech Republic": "Czechia", "Korea": "South Korea", "Canadian": "Canada", "German": "Germany", "French Guiana": "France",
         "Inner Mongolia": "China"}
BINS = [(1, 1, "1"), (2, 4, "2 to 4"), (5, 9, "5 to 9"), (10, 19, "10 to 19"), (20, 10**6, "20 or more")]
SHADES = ["#C7E9B4", "#7FCDBB", "#41B6C4", "#1D91C0", "#0C2C84"]
BIOMES = ["tropical", "subtropical", "arid", "mediterranean", "temperate", "boreal", "multiple", "not_reported"]


def countries(text, names, world):
    t = str(text)
    found = set()
    for a, c in ALIAS.items():
        if re.search(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])", t):
            found.add(c)
    t = t.replace("Inner Mongolia", " ")
    if re.search(r"United States|USA|Athens", t):  # the US state, not the country
        t = t.replace("Georgia", " ")
    for nm in names:
        if re.search(r"(?<![A-Za-z])" + re.escape(nm) + r"(?![A-Za-z])", t):
            found.add(world.loc[(world.NAME == nm) | (world.NAME_LONG == nm) | (world.ADMIN == nm), "NAME"].iloc[0])
    return found


def table():
    """Countries, hemisphere and biomes of each study; studies per country."""
    world = gpd.read_file(f"zip://{paths.NATURAL_EARTH_ZIP}")
    names = sorted(set(world.NAME) | set(world.NAME_LONG) | set(world.ADMIN), key=len, reverse=True)
    world = world[world.NAME != "Antarctica"]
    lat = {r.NAME: r.geometry.representative_point().y for r in world.itertuples()}
    lab = studies().set_index("study").label
    areas, biome = study_values("study_area"), study_values("biome")
    rows, per_country = [], Counter()
    for st, vals in areas.items():
        cs = set().union(*(countries(v, names, world) for v in vals if v not in ("not_reported", "nan")))
        per_country.update(cs)
        rows.append(dict(study=st, label=lab.get(st, ""), countries=";".join(sorted(cs)),
                         northern=all(lat.get(c, 0) >= 0 for c in cs) if cs else None, biome=";".join(sorted(biome[st])),
                         study_area=" | ".join(sorted(vals))))
    return pd.DataFrame(rows), per_country, world, biome


def main():
    df, per_country, world, biome = table()
    write(df, "geography")
    located = df[df.countries != ""]
    north = int(located.northern.sum())
    world["studies"] = world.NAME.map(per_country).fillna(0)
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.08})
    world.plot(ax=ax, color="#F0F0F0", edgecolor="white", linewidth=0.4)
    for (lo, hi, _), col in zip(BINS, SHADES):
        sel = world[(world.studies >= lo) & (world.studies <= hi)]
        if len(sel):
            sel.plot(ax=ax, color=col, edgecolor="#666666", linewidth=0.4)
    ax.set_axis_off()
    legend(ax, [Patch(color=c, label=l) for (_, _, l), c in zip(BINS, SHADES)], "Studies per country", loc="lower left", fontsize=8)
    ax.text(0.0, 1.0, f"{len(located)} of {len(df)} studies name their study area by country; {north} of them lie wholly in the Northern Hemisphere",
            transform=ax.transAxes, fontsize=8.5, style="italic", va="bottom")
    cnt = Counter(v for vals in biome.values() for v in vals)
    keys = [k for k in BIOMES if cnt.get(k)] + [k for k in cnt if k not in BIOMES]
    axb.barh([pretty(k) for k in keys][::-1], [cnt[k] for k in keys][::-1],
             color=[LIGHT if k == "not_reported" else "#4C6A92" for k in keys][::-1], height=0.62)
    for i, k in enumerate(keys[::-1]):
        axb.text(cnt[k] + 0.6, i, f"{cnt[k]} ({100 * cnt[k] / len(df):.0f}%)", va="center", fontsize=8.5)
    axb.set_xlabel("Number of studies")
    axb.set_xlim(0, max(cnt.values()) * 1.3)
    panel_titles(fig, [(ax, "(a) Study areas"), (axb, "(b) Biomes")])
    save(fig, "geography")
    print(f"  {len(located)} studies with countries ({len(per_country)} countries), {north} wholly in the Northern Hemisphere")


if __name__ == "__main__":
    set_style()
    main()
