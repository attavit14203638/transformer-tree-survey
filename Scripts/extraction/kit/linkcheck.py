"""Second-pass check: every recorded number must appear in its own evidence quote, and each result's model/dataset must exist.

Lists every result with its model name, dataset, setting and value so the table can be read against the papers."""
import re
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

BATCH = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: linkcheck.py B## [--list]")
ws = load_workbook(f"{paths.BATCHES}/{BATCH}.xlsx", read_only=True)["values"]
rows = list(ws.iter_rows(values_only=True))
h = {k: i for i, k in enumerate(rows[0])}
by = defaultdict(dict)
meta = {}
for r in rows[1:]:
    pid, item, field = r[h["paper_id"]], r[h["item"]], r[h["field"]]
    by[(pid, item)][field] = r[h["value"]]
    meta[(pid, item, field)] = r


def norm(s):
    """The evidence twice: as printed, and with thousands separators (7,824 or 110 937) joined."""
    s = str(s or "").replace("−", "-").replace("\u2009", " ").replace("\u202f", " ")
    return s + " || " + re.sub(r"(?<=\d)(?:,\s?|\s)(?=\d{3}\b)", "", s)


issues = 0
NUMF = ("value", "value_in_domain", "value_out_of_domain", "total_params", "trainable_params", "flops", "gsd", "point_density", "n_images",
        "n_trees", "label_value", "label_amount", "cf_alternative")
for (pid, item, field), r in meta.items():
    if field not in NUMF:
        continue
    v, ev, st, loc = str(r[h["value"]]), norm(r[h["evidence"]]), r[h["status"]], str(r[h["location"]] or "")
    if st in ("approximate", "derived", "not_reported") or "(image)" in loc or "rotated" in loc:
        continue
    pat = r"(?<![\d.])" + re.escape(v).replace(r"\-", "[-−]") + r"(?![\d])"
    if not re.search(pat, ev):
        # allow trailing-zero forms (e.g. 78.1 vs 78.10)
        alt = [v.rstrip("0").rstrip(".") if "." in v else v, v + "0"]
        if not any(re.search(r"(?<![\d.])" + re.escape(a) + r"(?![\d])", ev) for a in alt):
            issues += 1
            print(f"NOT IN EVIDENCE  {pid} {item} {field}={v} | {ev[:110]}")
# cross-reference
for (pid, item), f in by.items():
    item = item or ""
    if item.startswith("res"):
        for k in ("result_model", "result_dataset"):
            if f.get(k) and (pid, f[k]) not in by:
                issues += 1
                print(f"MISSING REF {pid} {item} {k}={f[k]}")
    if item.startswith("tr") and f.get("transfer_model") and (pid, f["transfer_model"]) not in by:
        issues += 1
        print(f"MISSING REF {pid} {item} transfer_model={f['transfer_model']}")
print(f"{BATCH}: {issues} issue(s)")
if "--list" in sys.argv:
    for (pid, item), f in sorted(by.items(), key=lambda t: (t[0][0], int(re.sub(r'\D', '', t[0][1] or '') or 0) if (t[0][1] or '').startswith(('res', 'tr')) else 0, t[0][1] or '')):
        item = item or ""
        if item.startswith("res"):
            m = by.get((pid, f.get("result_model")), {})
            print(f"{pid} {item:6s} {f.get('metric'):9s} {str(f.get('value')):>8s}  {m.get('model_name', '?')[:48]:48s} | {f.get('result_dataset')} | {str(f.get('result_setting'))[:60]}")
        elif item.startswith("tr"):
            m = by.get((pid, f.get("transfer_model")), {})
            print(f"{pid} {item:6s} {f.get('transfer_metric'):9s} {str(f.get('value_in_domain')):>8s} -> {str(f.get('value_out_of_domain')):>8s}  {m.get('model_name', '?')[:40]} | {str(f.get('transfer_target'))[:50]}")
