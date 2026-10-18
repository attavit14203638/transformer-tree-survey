"""Reproduce every number, table and figure of the review from the extraction data.

  python reproduce.py           rebuild all outputs, then check the data outputs against reproduce_manifest.json
  python reproduce.py --check   only check the current outputs against the manifest

Steps: compile the batch workbooks (Scripts/extraction/compile.py); recompute the PRISMA counts and the screening agreement
from the screening log (Scripts/search/screening.py prisma, kappa); rebuild the figures, figure data and tables
(Scripts/figures/make_figures.py --submission --numbered); regenerate every number (Scripts/figures/manuscript_numbers.py).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ["Scripts/extraction/compile.py"],
    ["Scripts/search/screening.py", "prisma"],
    ["Scripts/search/screening.py", "kappa"],
    ["Scripts/figures/make_figures.py", "--submission", "--numbered"],
    ["Scripts/figures/manuscript_numbers.py"],
]


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check():
    manifest = json.loads((ROOT / "reproduce_manifest.json").read_text())
    bad = [f for f, h in manifest.items() if not (ROOT / f).exists() or sha256(ROOT / f) != h]
    print(f"{len(manifest) - len(bad)} of {len(manifest)} data outputs identical to the published ones")
    for f in bad:
        print("  differs:", f)
    return not bad


def main():
    if "--check" not in sys.argv[1:]:
        for step in STEPS:
            print("$ python " + " ".join(step), flush=True)
            r = subprocess.run([sys.executable, *step], cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                print(r.stdout[-3000:], r.stderr[-3000:])
                raise SystemExit(f"step failed: {' '.join(step)}")
    sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
