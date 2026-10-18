"""Print the passages of a paper that matter for extraction, with PDF page numbers."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

if len(sys.argv) < 2:
    sys.exit("usage: excerpt.py P### [overview | tables [N] | page P | grep REGEX]")
PID = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "overview"
T = str(paths.PDF_TEXT)
raw = open(f"{T}/{PID}.txt", encoding="utf-8", errors="ignore").read().split("\f")
lay = open(f"{T}/{PID}.layout.txt", encoding="utf-8", errors="ignore").read().split("\f")

GROUPS = {
    "area": r"study (area|site)|located in|province|county|country|region|national park|forest farm|°\s*[NSEW]|latitude",
    "platform": r"\bUAV|drone|unmanned|aerial|satellite|airborne|LiDAR|laser scann|WorldView|Pleiades|GeoEye|PlanetScope|SkySat|camera|DJI|sensor",
    "res": r"\bGSD\b|ground (sampling|sample) distance|spatial resolution|resolution of|cm/pixel|cm per pixel|\d+(\.\d+)? ?cm\b|pts/m|points per square|point density",
    "data": r"dataset|images? (were|was)|tiles?|patches|annotat|label(l)?ed|bounding box|polygon|training set|test set|validation set|split",
    "count": r"\d[\d,]* (trees|crowns|images|tiles|patches|instances|annotations|samples|palms|seedlings)",
    "model": r"backbone|pre-?train|ImageNet|COCO|fine-?tun|from scratch|frozen|LoRA|adapter|zero-shot|few-shot|epochs|learning rate|optimizer",
    "eff": r"param|FLOPs|GFLOPs|FPS|frames per second|inference (time|speed)|latency|ms\b",
    "transfer": r"generali[sz]|transfer|cross-(site|region|domain|dataset|sensor)|unseen|out-of-distribution|domain (shift|gap|adaptation)|another (site|region)|different (site|region|area)",
    "avail": r"github|code (is|are|will be)? ?(publicly )?available|available (at|online|from)|data availability|zenodo|figshare|huggingface|upon (reasonable )?request|open[- ]source",
    "limit": r"limitation|future work|fail|struggle",
}


def hits(rx, pages, width=230, maxn=14):
    out = []
    for i, pg in enumerate(pages, start=1):
        for line in pg.split("\n"):
            if re.search(rx, line, re.I):
                s = re.sub(r"\s+", " ", line).strip()
                if len(s) > 15:
                    out.append(f"[p.{i}] {s[:width]}")
    return out[:maxn]


if MODE == "overview":
    first = re.sub(r"[ \t]+", " ", raw[0])
    print("=== PAGE 1 (raw, first 3000 chars)\n" + first[:3000])
    for g in ("area", "platform", "res", "data", "count", "model", "eff", "transfer", "avail"):
        h = hits(GROUPS[g], raw)
        print(f"\n=== {g.upper()} ({len(h)} shown)")
        print("\n".join(h))
    caps = hits(r"^\s*(Table|TABLE|Fig\.|Figure)\s*\d+[.:]?", lay, width=200, maxn=40)
    print("\n=== CAPTIONS\n" + "\n".join(c for c in caps if re.search(r"Table|TABLE", c)))
elif MODE == "tables":
    # print every table caption with the following lines (layout text)
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 28
    for i, pg in enumerate(lay, start=1):
        lines = pg.split("\n")
        for j, line in enumerate(lines):
            if re.match(r"^\s*(Table|TABLE)\s*[0-9IVX]+", line):
                print(f"\n##### p.{i} line {j}")
                print("\n".join(l.rstrip() for l in lines[j:j + n] if l.strip()))
elif MODE == "page":
    p = int(sys.argv[3])
    print(lay[p - 1])
elif MODE == "grep":
    for h in hits(sys.argv[3], raw, width=300, maxn=40):
        print(h)
