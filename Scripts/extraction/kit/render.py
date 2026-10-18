"""Render a PDF page (or part of it) to PNG so it can be read as an image.

  python render.py P### PAGE [--dpi 130] [--crop X0 Y0 X1 Y1] [--rotate 90]

--crop takes fractions of the page width and height (0 to 1), for example --crop 0 0.1 1 0.45 for a table in the top
half. Prints the path of the PNG. Use it for tables stored as images, rotated tables, scanned pages and charts, and to
check any value whose text layer looks garbled.
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pid")
ap.add_argument("page", type=int)
ap.add_argument("--dpi", type=int, default=130)
ap.add_argument("--crop", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
ap.add_argument("--rotate", type=int, default=0)
a = ap.parse_args()
pdf = glob.glob(f"{paths.PDF_LIBRARY}/*/{a.pid}_*.pdf")
if not pdf:
    sys.exit(f"no PDF for {a.pid} in the library")
out_dir = os.path.realpath("/tmp") + "/kit/render"
os.makedirs(out_dir, exist_ok=True)
base = f"{out_dir}/{a.pid}_p{a.page}_{a.dpi}"
subprocess.run(["pdftoppm", "-f", str(a.page), "-l", str(a.page), "-r", str(a.dpi), "-png", "-singlefile", pdf[0], base], check=True)
im = Image.open(base + ".png")
name = base
if a.crop:
    w, h = im.size
    x0, y0, x1, y1 = a.crop
    im = im.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
    name += "_crop_" + "_".join(f"{c:g}" for c in a.crop)
if a.rotate:
    im = im.rotate(a.rotate, expand=True)
    name += f"_r{a.rotate}"
im.save(name + ".png")
print(name + ".png", im.size)
