"""Check that every metric tag says no more than the paper prints.

  Scripts/.venv/bin/python Scripts/extraction/kit/metriccheck.py [--csv]

For every result, transfer pair and data-efficiency point in the batch workbooks, the metric tag is compared with the name
the paper prints (the verbatim evidence quote; our notes are not evidence) and, for metrics whose definition matters, with the paper's own text
(PDF_Text/P###.txt):
  ap50_95   needs the IoU range 0.5 to 0.95 in the paper (any common notation), or an explicit statement that the paper uses
            the COCO evaluation; a mean average precision "over several IoU thresholds" without the range is tagged map
  ap50      needs an IoU threshold of 0.5 (AP50, AP@0.5, mAP@0.5, mAP50, IoU of 0.5, ...)
  iou/miou  a quote that prints only one of "IoU" and "mIoU"/"mean IoU" must carry that tag
Prints one line per (paper, tag) that is not supported, and with --csv writes Extraction/reports/metriccheck.csv.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

METRIC_FIELDS = {"metric", "transfer_metric", "label_metric"}
# Tags checked by hand where the support is not a phrase a pattern can find (paper, tag): reason
VERIFIED = {}  # none: a tag the paper's words do not support is changed, not excused
RANGE = re.compile(r"(?:0?\.50?|50)\s*%?\s*(?:[:–\-]|to)\s*(?:0?\.05\s*[:–\-]\s*)?(?:0?\.95|95)\s*%?(?![\d.])"
                   r"|\[\s*0?\.50?\s*[:,]\s*(?:0?\.05\s*[:,]\s*)?0?\.95\s*\]|AP\s*@\s*\[\s*\.?50?\s*:\s*(?:\.?05\s*:\s*)?\.?95"
                   r"|50\s*:\s*5\s*:\s*95|ten IoU thresholds|10 IoU thresholds", re.I)
COCO = re.compile(r"COCO\)?\s*(?:\[\d+\]\s*)*[- ]?(?:style|standard|evaluation|evaluator|metric|metrics|protocol|API|AP|mAP)\b|(?:evaluat\w+|metrics?) (?:of|from|follow\w*|defined by|"
                  r"as in|according to) (?:the )?(?:MS[- ]?)?COCO|pycocotools|COCO evaluat", re.I)
AP50 = re.compile(r"\bm?AP\s*_?\{?\s*50\b(?!\s*[:–\-])|\bm?AP\s*@\s*(?:IoU\s*=\s*)?50\b(?!\s*[:–\-])|\bm?AP\s*@\s*0?\.50?\b(?!\s*[:–\-])|\bm?AP\s*0?\.50?\b(?!\s*[:–\-])"
                  r"|IoU\)?\s*(?:thresholds?\s*)?(?:of|=|≥|>=|>|at|is|was set to|set to|set at)\s*0?\.50?\b(?!\s*[:–\-])"
                  r"|IoU\)?\s*(?:with an? )?(?:threshold\s*)?(?:of|=|≥|>=|>|at|is)\s*50\s*%|IoU\)? with an? threshold of 0?\.50?\b|0?\.50?\s*IoU|IoU\s*0?\.50?\b(?!\s*[:–\-])", re.I)
MIOU = re.compile(r"\bmIoU\b|mean IoU|mean intersection", re.I)
IOU = re.compile(r"(?<![mM])IoU\b|intersection over union|Jaccard", re.I)


def rows():
    for f in sorted(paths.BATCHES.glob("B*.xlsx")):
        it = load_workbook(f, read_only=True, data_only=True)["values"].iter_rows(values_only=True)
        hdr = next(it)
        for r in it:
            yield f.stem, dict(zip(hdr, r))


def main(write_csv):
    items = defaultdict(dict)
    for batch, r in rows():
        key = (batch, r["paper_id"], r["item"])
        if r["field"] in METRIC_FIELDS:
            items[key].update(metric=str(r["value"]), quote=str(r["evidence"] or ""), page=r["page"], loc=r["location"],
                              mnote=str(r["note"] or ""))
        elif r["field"] in ("value", "transfer_value_in", "transfer_value_out", "label_value"):
            items[key].setdefault("vnote", "")
            items[key]["vnote"] += " " + str(r["note"] or "")
    text = {}
    def paper_text(pid):
        if pid not in text:
            p = paths.PDF_TEXT / f"{pid}.txt"
            parts = [q.read_text(encoding="utf-8", errors="ignore") for q in (p, p.with_suffix(".layout.txt")) if q.exists()]
            text[pid] = re.sub(r"\s+", " ", " ".join(parts))  # line breaks and layout columns would hide a definition
        return text[pid]
    findings = []
    for (batch, pid, item), d in sorted(items.items()):
        m = d.get("metric")
        if not m:
            continue
        local = d.get("quote", "")  # the paper's words only: our notes are interpretations, not evidence
        full = paper_text(pid)
        status = ""
        if (pid, m) in VERIFIED:
            status = "ok (verified by hand: " + VERIFIED[(pid, m)] + ")"
        elif m == "ap50_95":
            hit = RANGE.search(local) or RANGE.search(full)
            if hit:
                status = "ok (range stated)"
                d["hit"] = hit.string[max(0, hit.start() - 40):hit.end() + 20].replace("\n", " ")
            elif COCO.search(local) or COCO.search(full):
                hit = COCO.search(local) or COCO.search(full)
                status = "ok (COCO evaluation stated)"
                d["hit"] = hit.string[max(0, hit.start() - 40):hit.end() + 40].replace("\n", " ")
            else:
                status = "UNSUPPORTED: no IoU range or COCO evaluation in the paper"
        elif m == "ap50":
            hit = AP50.search(local) or AP50.search(full)
            status = "ok" if hit else "UNSUPPORTED: no IoU threshold of 0.5 in the paper"
            if hit:
                d["hit"] = hit.string[max(0, hit.start() - 40):hit.end() + 20].replace("\n", " ")
        elif m in ("iou", "miou"):
            q = d.get("quote", "")
            if m == "iou" and MIOU.search(q) and not IOU.search(q.replace("mIoU", "")):
                status = "MISMATCH: the quote prints mIoU"
            elif m == "miou" and IOU.search(q) and not MIOU.search(q):
                status = "MISMATCH: the quote prints IoU, not mIoU"
            else:
                status = "ok"
        else:
            continue
        findings.append(dict(batch=batch, paper_id=pid, item=item, metric=m, status=status, page=d.get("page"),
                             location=d.get("loc"), quote=d.get("quote", "")[:120], hit=d.get("hit", "")[:100], note=(d.get("mnote", "") + d.get("vnote", "")).strip()[:120]))
    bad = [f for f in findings if not f["status"].startswith("ok")]
    by = defaultdict(list)
    for f in bad:
        by[(f["paper_id"], f["metric"], f["status"])].append(f["item"])
    for (pid, m, s), its in sorted(by.items()):
        print(f"  {pid} {m:8s} {s}: {len(its)} item(s), e.g. {', '.join(its[:4])}")
    print(f"{len(findings)} tagged items checked; {len(bad)} not supported, in {len({f['paper_id'] for f in bad})} papers")
    if write_csv:
        out = paths.EXTRACTION / "reports" / "metriccheck.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(findings[0].keys()))
            w.writeheader()
            w.writerows(findings)
        print(f"  written to {out.relative_to(paths.SURVEY)}")


if __name__ == "__main__":
    main("--csv" in sys.argv[1:])
