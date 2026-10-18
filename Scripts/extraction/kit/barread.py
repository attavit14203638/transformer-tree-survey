"""Read horizontal bar lengths from a rendered PDF page, calibrated on the tick labels found by pdftotext -bbox.

usage: barread.py PDF PAGE TICK_LO TICK_HI 'label1|label2|...' R,G,B [R,G,B ...]
Prints, for each colour, the value at the right end of each bar, ordered top to bottom and matched to the labels."""
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image

if len(sys.argv) < 7:
    sys.exit(__doc__)
pdf, page, lo, hi, labels = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5].split("|")
colours = [tuple(int(c) for c in s.split(",")) for s in sys.argv[6:]]
DPI = 300
TMP = os.path.realpath("/tmp") + "/kit"
os.makedirs(TMP, exist_ok=True)
for _f in __import__("glob").glob(TMP + "/_bar*.png"):
    os.remove(_f)
html = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-bbox", pdf, "-"], capture_output=True, text=True).stdout
words = [(float(a), float(b), float(c), float(d), w) for a, b, c, d, w in re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', html)]


def fmt(v):
    return str(int(v)) if v == int(v) else str(v)


lo_w = [w for w in words if w[4] == fmt(lo)]
hi_w = [w for w in words if w[4] == fmt(hi)]
# tick labels share the same baseline: pick the pair with equal yMin
pairs = [(a, b) for a in lo_w for b in hi_w if abs(a[1] - b[1]) < 1 and b[0] > a[0]]
if not pairs:
    sys.exit(f"tick labels {lo} / {hi} not found on one line")
a, b = pairs[0]
x_lo, x_hi = (a[0] + a[2]) / 2 * DPI / 72, (b[0] + b[2]) / 2 * DPI / 72
tick_y = a[1] * DPI / 72
subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(DPI), "-png", pdf, TMP + "/_bar"], check=True)
img = np.asarray(Image.open(sorted(__import__("glob").glob(TMP + "/_bar*.png"))[0]).convert("RGB")).astype(int)
lab_w = []
for lab in labels:
    toks = lab.split()
    hit = [w for w in words if w[4] == toks[0] and w[2] * DPI / 72 < x_lo + 5]
    if not hit:
        sys.exit(f"label {lab} not found")
    lab_w.append((lab, (hit[0][1] + hit[0][3]) / 2 * DPI / 72))
print(f"calibration: {fmt(lo)} at x={x_lo:.1f}px, {fmt(hi)} at x={x_hi:.1f}px; tick labels at y={tick_y:.0f}px")
for col in colours:
    mask = (np.abs(img - np.array(col)).sum(axis=2) < 60)
    mask[int(tick_y) - 5:, :] = False
    mask[:, : int(x_lo) - 10] = False
    rows = np.where(mask.sum(axis=1) > 20)[0]
    bars, start = [], None
    for i, r in enumerate(rows):
        if start is None:
            start = r
        if i == len(rows) - 1 or rows[i + 1] != r + 1:
            bars.append((start, r))
            start = None
    out = []
    for y0, y1 in bars:
        band = mask[y0:y1 + 1]
        xs = np.where(band.any(axis=0))[0]
        x_end = xs.max() + 0.5
        val = lo + (x_end - x_lo) / (x_hi - x_lo) * (hi - lo)
        yc = (y0 + y1) / 2
        lab = min(lab_w, key=lambda t: abs(t[1] - yc))[0]
        out.append((yc, lab, val, y1 - y0 + 1))
    print(f"colour {col}: {len(out)} bars")
    for yc, lab, val, h in out:
        print(f"   y={yc:6.0f} h={h:3d}  {lab:22s} {val:7.2f}")
