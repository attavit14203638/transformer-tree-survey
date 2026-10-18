"""Compile the batch workbooks into the analysis tables the figures read, and check that they are figure-ready.

  python compile.py            write Literature_Extraction/Extraction/compiled/*.csv and print the checks
  python compile.py --check    run the checks only

Tables (one row per ...):
  studies.csv          included paper: registry fields and study-level extraction fields (GSD in cm)
  datasets.csv         dataset item, with its dataset_key
  models.csv           model item: category, transformer flag, pre-training, training strategy, size and cost
  results.csv          result item with its protocol 6 tags; value_std puts bounded metrics on a % scale and lengths in cm
  comparisons.csv      study x task x attribute x level x metric x dataset, from primary results only: best transformer
                       against the strongest non-transformer deep model (protocol 4.2)
  transfer.csv         transfer pair, with the change in points and in percent
  data_efficiency.csv  data-efficiency point
  conflicts.csv        every value choice made under rules V1 and V2, with the value used and the value not used

A study is a paper, or the papers that share a study_group in papers.csv (reports of one study). Results that appear
identically in two reports of one study get duplicate_of and count once.

Errors (exit code 1) make a table unusable for figures; warnings need a look; notes are for information. A warning that
has been looked at and is fine is recorded in Extraction/compile_review.csv (kind, key, decision, reason) and is then
listed as reviewed instead.
"""
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

TRANSFORMER = {"pure_vit", "hierarchical_vit", "cnn_transformer_hybrid", "foundation_model", "vision_language_model"}
DEEP_BASELINE = {"cnn", "other"}
BOUNDED = {"miou", "iou", "f1", "ap50", "ap50_95", "map", "oa", "kappa", "precision", "recall"}
HIGHER_BETTER = BOUNDED | {"r2"}
ERRORS = {"rmse", "mae", "mape"}
LENGTH = {"mm": 0.1, "cm": 1.0, "m": 100.0}
REVIEW_CSV = paths.EXTRACTION / "compile_review.csv"
# One metric per study, task, attribute, level and dataset for one-point-per-study figures: the first available in this order
METRIC_ORDER = {
    "object": ["ap50", "f1", "ap50_95", "map", "precision", "recall"],
    "instance_mask": ["ap50", "f1", "iou", "miou", "ap50_95", "map", "precision", "recall"],
    "instance_3d": ["f1", "ap50", "ap50_95", "map", "precision", "recall"],
    "pixel": ["iou", "miou", "f1", "oa", "precision", "recall"],
    "point": ["miou", "iou", "oa", "f1"],
    "tree": ["oa", "f1", "kappa", "miou", "precision", "recall", "rmse", "mae", "r2", "mape"],
    "image": ["mae", "rmse", "r2", "oa", "f1"],
    "plot": ["rmse", "mae", "r2"],
}


FLOPS = {"MFLOPs": 0.001, "GFLOPs": 1.0, "TFLOPs": 1000.0}


def flops_g(r):
    """FLOPs in GFLOPs, converted by the printed unit (rule V5); None when absent or the unit is not printed."""
    if not r or r["status"] == "not_reported" or r["unit"] not in FLOPS:
        return None
    v = num(r["value"])
    return None if v is None else round(v * FLOPS[r["unit"]], 6)


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def load():
    with open(paths.PAPERS_CSV, encoding="utf-8") as f:
        papers = {r["paper_id"]: r for r in csv.DictReader(f) if r["paper_id"]}
    rows = []
    for book in sorted(paths.BATCHES.glob("B*.xlsx")):
        ws = load_workbook(book, read_only=True, data_only=True)["values"]
        it = ws.iter_rows(values_only=True)
        head = [str(h) for h in next(it)]
        for r in it:
            if r[0]:
                d = {h: ("" if v is None else str(v).strip()) for h, v in zip(head, r)}
                d["batch"] = book.stem
                rows.append(d)
    with open(paths.DATASETS_CSV, encoding="utf-8") as f:
        registry = {r["dataset_key"]: r for r in csv.DictReader(f)}
    review = {}
    if REVIEW_CSV.exists():
        with open(REVIEW_CSV, encoding="utf-8") as f:
            review = {(r["kind"], r["key"]): r for r in csv.DictReader(f)}
    return papers, rows, registry, review


def build(papers, rows):
    study, items = defaultdict(dict), defaultdict(dict)
    for r in rows:
        if r["item"]:
            items[(r["paper_id"], r["item"])][r["field"]] = r
        else:
            study[r["paper_id"]][r["field"]] = r
    return study, items


def val(f, k):
    return f[k]["value"] if k in f else ""


def std_value(metric, value, unit):
    """Value on the common scale (rule V5): bounded metrics in %, lengths in cm, everything else as printed."""
    v = num(value)
    if v is None:
        return None, ""
    if metric in BOUNDED:
        if unit == "%":
            return v, "%"
        if unit == "fraction":
            return round(v * 100, 6), "%"
        return None, unit
    if metric == "r2":
        return v, "fraction"
    if unit in LENGTH:
        return round(v * LENGTH[unit], 6), "cm"
    return v, unit


def write(name, table):
    paths.COMPILED.mkdir(parents=True, exist_ok=True)
    cols = list(table[0].keys()) if table else ["empty"]
    with open(paths.COMPILED / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)


def main():
    check_only = "--check" in sys.argv
    papers, rows, registry, review = load()
    unit_of = {pid: (p.get("study_group") or pid) for pid, p in papers.items()}
    study, items = build(papers, rows)
    err, warn, note = [], [], []

    # ------------------------------------------------------------------ models, datasets
    models, dsets = {}, {}
    for (pid, it), f in items.items():
        if it.startswith("model"):
            cat = val(f, "architecture_category")
            pr = lambda k, u="M": num(val(f, k)) if k in f and f[k]["unit"] == u and f[k]["status"] != "not_reported" else None  # noqa: E731
            models[(pid, it)] = dict(paper_id=pid, item=it, model_name=val(f, "model_name"), model_role=val(f, "model_role"),
                                     architecture_category=cat or "not_reported", transformer=cat in TRANSFORMER,
                                     backbone=val(f, "backbone"), pretraining=val(f, "pretraining") or "not_reported",
                                     training_strategy=val(f, "training_strategy") or "not_reported",
                                     total_params_M=pr("total_params"), trainable_params_M=pr("trainable_params"),
                                     trainable_params_pct=pr("trainable_params", "%"),
                                     flops_G=flops_g(f.get("flops")), inference_speed=val(f, "inference_speed"))
            fl = f.get("flops")
            if fl and fl["status"] != "not_reported" and fl["unit"] not in FLOPS:
                note.append(f"{pid} {it} flops: unit {fl['unit']!r} not convertible, left out of models.csv")
            for k in ("total_params", "trainable_params"):
                if k in f and f[k]["status"] != "not_reported" and f[k]["unit"] not in ("M", "%" if k == "trainable_params" else "M"):
                    warn.append(f"{pid} {it} {k}: unit {f[k]['unit']!r} is not M, left out of models.csv")
        elif it.startswith("dataset"):
            key = val(f, "dataset_key")
            dsets[(pid, it)] = dict(paper_id=pid, item=it, dataset_key=key, dataset_name=val(f, "dataset_name"),
                                    dataset_role=val(f, "dataset_role"), dataset_public=val(f, "dataset_public"),
                                    n_images=num(val(f, "n_images")), n_trees=num(val(f, "n_trees")))
            if key and key not in registry:
                err.append(f"{pid} {it}: dataset_key {key!r} missing from datasets.csv")
    used = Counter(d["dataset_key"] for d in dsets.values())
    unused = sorted(set(registry) - set(used))
    if unused:
        warn.append(f"datasets.csv keys used by no paper: {unused}")
    shared = {k: sorted({d['paper_id'] for d in dsets.values() if d['dataset_key'] == k}) for k in used}
    shared = {k: v for k, v in shared.items() if len(v) > 1}

    # ------------------------------------------------------------------ results
    results = []
    for (pid, it), f in sorted(items.items()):
        if not it.startswith("res"):
            continue
        m = models.get((pid, val(f, "result_model")), {})
        d = dsets.get((pid, val(f, "result_dataset")), {})
        v = f.get("value", {})
        metric, unit = val(f, "metric"), v.get("unit", "")
        vs, us = std_value(metric, v.get("value"), unit)
        x = num(v.get("value"))
        if metric in BOUNDED:
            low = -1 if metric == "kappa" else 0  # kappa can be negative (agreement below chance)
            if unit not in ("%", "fraction"):
                err.append(f"{pid} {it}: {metric} with unit {unit!r} (must be % or fraction)")
            elif x is not None and unit == "fraction" and not low <= x <= 1:
                err.append(f"{pid} {it}: {metric} = {x} with unit fraction is outside {low} to 1")
            elif x is not None and unit == "%" and not low * 100 <= x <= 100:
                err.append(f"{pid} {it}: {metric} = {x} % is outside {low * 100} to 100")
            elif x is not None and unit == "%" and 0 < x <= 1 and metric != "kappa":
                msg = f"{pid} {it}: {metric} = {x} % (check it is not a fraction)"
                rv = review.get(("percent_below_one", f"{pid}|{it}"))
                (note if rv else warn).append(msg + (f" [reviewed: {rv['decision']}, {rv['reason']}]" if rv else f" [review key: percent_below_one,{pid}|{it}]"))
        if metric == "r2" and unit != "fraction":
            err.append(f"{pid} {it}: r2 with unit {unit!r} (must be fraction)")
        results.append(dict(
            paper_id=pid, study=unit_of.get(pid, pid), batch=v.get("batch", ""), item=it, model=val(f, "result_model"), model_name=m.get("model_name", ""),
            architecture_category=m.get("architecture_category", ""), transformer=m.get("transformer", False),
            model_role=m.get("model_role", ""), dataset=val(f, "result_dataset"), dataset_key=d.get("dataset_key", ""),
            task=val(f, "result_task"), attribute=val(f, "attribute"), level=val(f, "level"), split=val(f, "split"),
            aggregation=val(f, "aggregation"), metric=metric, value=v.get("value", ""), unit=unit, value_std=vs, unit_std=us,
            status=v.get("status", ""), primary=val(f, "primary") == "yes", headline=val(f, "headline") == "yes",
            copied_from=val(f, "copied_from"), duplicate_of="", setting=val(f, "result_setting"), page=v.get("page", ""), location=v.get("location", ""),
            note=v.get("note", ""), metric_note=f.get("metric", {}).get("note", "")))

    # the same result reported by two reports of one study counts once
    first = {}
    for r in results:
        k = (r["study"], r["dataset_key"], r["model_name"].lower(), r["task"], r["attribute"], r["level"], r["metric"],
             r["split"], r["aggregation"], r["value_std"])
        if r["dataset_key"] and r["value_std"] is not None:
            if k in first and first[k] != r["paper_id"]:
                r["duplicate_of"] = first[k]
            first.setdefault(k, r["paper_id"])
    dups = Counter((r["paper_id"], r["duplicate_of"]) for r in results if r["duplicate_of"])
    if dups:
        note.append("results reported again by another report of the same study (counted once): "
                    + ", ".join(f"{a} repeats {n} of {b}" for (a, b), n in sorted(dups.items())))
    # possible copied numbers not flagged: same dataset key, model name, metric and value in two studies
    seen = defaultdict(set)
    for r in results:
        if r["dataset_key"] and not r["copied_from"] and not r["duplicate_of"] and r["value_std"] is not None:
            seen[(r["dataset_key"], r["model_name"].lower(), r["metric"], r["value_std"])].add((r["study"], r["paper_id"]))
    for k, hits in sorted(seen.items(), key=lambda t: str(t[0])):
        if len({h[0] for h in hits}) > 1:
            key = "|".join(str(x) for x in k)
            msg = f"same value in several studies (copied without copied_from?): {k} in {sorted(h[1] for h in hits)}"
            rv = review.get(("same_value", key))
            (note if rv else warn).append(msg + (f" [reviewed: {rv['decision']}, {rv['reason']}]" if rv else f" [review key: same_value,{key}]"))

    # ------------------------------------------------------------------ comparisons (protocol 4.2)
    comps = []
    groups = defaultdict(list)
    canon = {}
    for r in results:
        if (r["primary"] and r["status"] == "reported" and not r["copied_from"] and not r["duplicate_of"]
                and r["value_std"] is not None and r["split"] != "not_stated" and r["level"] != "not_stated"):
            groups[(r["study"], r["task"], r["attribute"], r["level"], r["metric"], r["dataset_key"], r["split"],
                    r["aggregation"])].append(r)
            unit_key = (r["study"], r["task"], r["attribute"], r["level"], r["dataset_key"])
            order = METRIC_ORDER.get(r["level"], [])
            rank = order.index(r["metric"]) if r["metric"] in order else len(order)
            canon[unit_key] = min(canon.get(unit_key, rank), rank)
    for r in results:
        k = (r["study"], r["task"], r["attribute"], r["level"], r["dataset_key"])
        order = METRIC_ORDER.get(r["level"], [])
        r["canonical_metric"] = bool(r["primary"] and k in canon and r["metric"] in order and order.index(r["metric"]) == canon[k]
                                     and not r["copied_from"] and not r["duplicate_of"] and r["status"] == "reported")
    for (study_id, task, attr, level, metric, dkey, split, agg), rs in sorted(groups.items()):
        tr = [r for r in rs if r["transformer"]]
        bl = [r for r in rs if r["architecture_category"] in DEEP_BASELINE]
        if not tr or not bl or metric not in HIGHER_BETTER | ERRORS:
            continue
        better = max if metric in HIGHER_BETTER else min
        bt, bb = better(tr, key=lambda r: r["value_std"]), better(bl, key=lambda r: r["value_std"])
        if metric in HIGHER_BETTER:
            diff = bt["value_std"] - bb["value_std"]
            scale = 100 if metric == "r2" else 1
            diff_pp = round(diff * scale, 4)
            tie = abs(diff_pp) <= 1
            diff_rel = round(100 * diff / bb["value_std"], 4) if bb["value_std"] else None
        else:
            diff_pp = None
            diff_rel = round(100 * (bb["value_std"] - bt["value_std"]) / bb["value_std"], 4) if bb["value_std"] else None
            tie = diff_rel is not None and abs(diff_rel) <= 2
        comps.append(dict(study=study_id, papers=";".join(sorted({r["paper_id"] for r in rs})), task=task, attribute=attr, level=level,
                          metric=metric, dataset_key=dkey, split=split, aggregation=agg,
                          n_transformer=len(tr), n_baseline=len(bl), best_transformer=bt["model_name"],
                          transformer_category=bt["architecture_category"], transformer_value=bt["value_std"],
                          best_baseline=bb["model_name"], baseline_category=bb["architecture_category"], baseline_value=bb["value_std"],
                          unit_std=bt["unit_std"], diff_pp=diff_pp, diff_rel_pct=diff_rel, tie=tie,
                          transformer_ref=f"{bt['paper_id']}:{bt['item']}", transformer_page=bt["page"],
                          baseline_ref=f"{bb['paper_id']}:{bb['item']}", baseline_page=bb["page"],
                          winner="tie" if tie else ("transformer" if (diff_pp if diff_pp is not None else diff_rel) > 0 else "baseline"),
                          canonical_metric=metric in METRIC_ORDER.get(level, []) and
                          METRIC_ORDER[level].index(metric) == canon.get((study_id, task, attr, level, dkey))))

    # ------------------------------------------------------------------ transfer
    trans = []
    for (pid, it), f in sorted(items.items()):
        if not it.startswith("tr"):
            continue
        m = models.get((pid, val(f, "transfer_model")), {})
        a, b = f.get("value_in_domain", {}), f.get("value_out_of_domain", {})
        metric = val(f, "transfer_metric")
        ia, ua = std_value(metric, a.get("value"), a.get("unit", ""))
        ib, ub = std_value(metric, b.get("value"), b.get("unit", ""))
        complete = ia is not None and ib is not None and a.get("status") != "not_reported" and b.get("status") != "not_reported"
        approx = "approximate" in (a.get("status"), b.get("status"))
        if complete and ua != ub:
            err.append(f"{pid} {it}: in-domain and out-of-domain values on different scales ({ua} vs {ub})")
        drop_pp = round((ib - ia) * (100 if metric == "r2" else 1), 4) if complete and metric in HIGHER_BETTER else None
        drop_rel = round(100 * (ib - ia) / ia, 4) if complete and ia else None
        trans.append(dict(paper_id=pid, item=it, model=val(f, "transfer_model"), model_name=m.get("model_name", ""),
                          architecture_category=m.get("architecture_category", ""), transformer=m.get("transformer", False),
                          metric=metric, level=val(f, "transfer_level"), shift_type=val(f, "shift_type"),
                          source=val(f, "transfer_source"), target=val(f, "transfer_target"), value_in_std=ia, value_out_std=ib,
                          unit_std=ua or ub, change_pp=drop_pp, change_rel_pct=drop_rel, complete=complete, approximate=approx,
                          page_in=a.get("page", ""), location_in=a.get("location", ""),
                          page_out=b.get("page", ""), location_out=b.get("location", "")))
    incomplete = [f"{t['paper_id']} {t['item']}" for t in trans if not t["complete"]]
    if incomplete:
        note.append(f"transfer pairs without both values (kept out of paired figures, rule V8): {incomplete}")

    # ------------------------------------------------------------------ data efficiency, conflicts
    de = []
    for (pid, it), f in sorted(items.items()):
        if it.startswith("de"):
            metric = val(f, "label_metric")
            v = f.get("label_value", {})
            vs, us = std_value(metric, v.get("value"), v.get("unit", ""))
            de.append(dict(paper_id=pid, item=it, model=val(f, "label_model"), label_amount=num(val(f, "label_amount")),
                           label_unit=val(f, "label_unit"), metric=metric, value_std=vs, unit_std=us, status=v.get("status", ""),
                           page=v.get("page", ""), location=v.get("location", ""), note=v.get("note", ""),
                           setting="; ".join(dict.fromkeys(n for n in (f.get(k, {}).get("note", "") for k in ("label_value", "label_metric")) if n)),
                           amount_note=f.get("label_amount", {}).get("note", "")))
    cfs = []
    for (pid, it), f in sorted(items.items()):
        if it.startswith("cf"):
            target = val(f, "cf_target")
            used_value = ""
            mt = re.fullmatch(r"(\w+)\.(\w+)", target)
            if mt and (pid, mt.group(1)) in items:
                used_value = items[(pid, mt.group(1))].get(mt.group(2), {}).get("value", "")
            elif target in study.get(pid, {}):
                used_value = study[pid][target]["value"]
            alt = f.get("cf_alternative", {})
            cfs.append(dict(paper_id=pid, item=it, target=target, value_used=used_value, value_not_used=alt.get("value", ""),
                            unit=alt.get("unit", ""), page=alt.get("page", ""), location=alt.get("location", ""),
                            evidence=alt.get("evidence", ""), rule=val(f, "cf_rule"), reason=f.get("cf_rule", {}).get("note", "")))
            if (mt or target in study.get(pid, {})) and not used_value:
                err.append(f"{pid} {it}: cf_target {target!r} does not point to a recorded value")

    # ------------------------------------------------------------------ studies
    studies = []
    extracted = {r["paper_id"] for r in rows}
    for pid, p in sorted(papers.items()):
        if p["ft_decision"] not in ("Include", "Pending"):
            continue
        s = study.get(pid, {})
        g = s.get("gsd", {})
        gsd_cm = None
        if g and g.get("status") != "not_reported" and g.get("unit") in LENGTH:
            gsd_cm = round(num(g["value"]) * LENGTH[g["unit"]], 6)
        elif g and g.get("status") != "not_reported":
            warn.append(f"{pid} gsd: unit {g.get('unit')!r} is not mm, cm or m")
        rs = [r for r in results if r["paper_id"] == pid]
        studies.append(dict(paper_id=pid, study_group=p.get("study_group", ""), batch=p["batch"], year=p["year"], origin=p["origin"],
                            collection=p["collection"], extracted=pid in extracted, task=val(s, "task"),
                            output_unit=val(s, "output_unit"), tree_setting=val(s, "tree_setting"), biome=val(s, "biome") or "not_reported",
                            platform=val(s, "platform"), sensors=val(s, "sensors"), gsd_cm=gsd_cm, annotation_type=val(s, "annotation_type"),
                            study_area=val(s, "study_area"), code_available=val(s, "code_available"),
                            weights_available=val(s, "weights_available"), data_available=val(s, "data_available"),
                            n_models=sum(1 for k in models if k[0] == pid), n_results=len(rs), n_primary=sum(r["primary"] for r in rs),
                            has_headline=any(r["headline"] for r in rs)))
    not_extracted = [s["paper_id"] for s in studies if not s["extracted"]]
    no_primary = [s["paper_id"] for s in studies if s["extracted"] and not s["n_primary"]]
    if no_primary:
        note.append(f"extracted papers without primary results (tables only): {no_primary}")

    if not check_only:
        write("studies.csv", studies)
        write("datasets.csv", list(dsets.values()))
        write("models.csv", list(models.values()))
        write("results.csv", results)
        write("comparisons.csv", comps)
        write("transfer.csv", trans)
        write("data_efficiency.csv", de)
        write("conflicts.csv", cfs)
    n_ex = len(studies) - len(not_extracted)
    print(f"{n_ex} of {len(studies)} included papers extracted; {len(results)} results ({sum(r['primary'] for r in results)} primary), "
          f"{len(comps)} transformer-versus-baseline comparisons, {len(trans)} transfer pairs, {len(cfs)} recorded value choices")
    print(f"datasets: {len(used)} keys used; shared by several papers: {shared}")
    for label, xs in (("ERROR", err), ("warning", warn), ("note", note)):
        for x in xs:
            print(f"  {label}: {x}")
    if not check_only:
        print(f"tables written to {paths.COMPILED}")
    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()
