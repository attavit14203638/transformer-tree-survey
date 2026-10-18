"""Row builders for paper extraction scripts (protocol sections 3, 4 and 6).

Every builder returns rows (item, field, value, unit, page, location, evidence, status, note). A paper script
(Literature_Extraction/Extraction/scripts/B##/P###.py) defines ROWS and is written into its batch workbook with
`kit/append.py B## P###`. Typical start of a script:

    import os, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "Scripts" / "extraction" / "kit"))
    from h import R, NR, D, avail, model, result, conflict, Counter, Tags, row  # noqa: E402

    PID = "P123"
    T = Tags(task="health_classification", level="object", split="test", agg="single")
    ROWS = [R("task", "detection;health_classification", 1, "Abstract", "..."), ...]
    ROWS += result(res(), "ap50", "model1", "dataset1", "91.2", 8, "Table 3", ev, **T, primary=True)
    ROWS += result(res(), "f1", "model1", "dataset1", "88.0", 9, "Table 4", ev, **T(split="validation"))

The builders refuse rows that break protocol section 6 (approximate or 'other' results marked primary, a missing
attribute for attribute_estimation, unknown tag values), so mistakes stop the script instead of reaching the workbook.
"""
import re as _re
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import paths as _paths  # noqa: E402

_TXT = str(_paths.PDF_TEXT)
_NUM = _re.compile(r"(?<![\w.±])[-−]?\d+(?:\.\d+)?(?![\w.])")
TASKS = {"detection", "counting", "crown_segmentation", "tree_segmentation_3d", "component_segmentation",
         "species_classification", "health_classification", "mortality_detection", "attribute_estimation"}
LEVELS = {"object", "instance_mask", "pixel", "instance_3d", "point", "tree", "image", "plot", "not_stated"}
SPLITS = {"test", "validation", "cross_validation", "train", "not_stated"}
AGGS = {"single", "mean", "median", "best", "not_stated"}
ATTRIBUTES = {"dbh", "height", "crown_diameter", "crown_area", "stem_volume", "biomass", "trunk_radius", "chlorophyll", "other"}
METRICS = {"miou", "iou", "f1", "ap50", "ap50_95", "map", "oa", "kappa", "precision", "recall", "r2", "rmse", "mae", "mape", "other"}
BOUNDED = {"miou", "iou", "f1", "ap50", "ap50_95", "map", "oa", "kappa", "precision", "recall"}  # 0 to 1 or 0 to 100
CATEGORIES = {"pure_vit", "hierarchical_vit", "cnn_transformer_hybrid", "foundation_model", "vision_language_model", "cnn",
              "other", "classical"}
RULES = {"V1a", "V1b", "V1c", "V2"}


def row(pid, page, rx, n=0, cols=None, drop_sd=True):
    """Numbers printed on the n-th layout line of a page that matches rx (after the match), and the line itself.

    cols=(a, b) keeps character positions a..b of each line (two-column pages). Numbers written as '± x' or '(±x)' are
    dropped when drop_sd is True. Returns (numbers, evidence); evidence is the whitespace-normalised printed line.
    Check the result: a model name that contains digits (EVA-2, YOLOv8) or a thousands separator (7,824) splits into
    extra numbers, so assert the count you expect."""
    lines = open(f"{_TXT}/{pid}.layout.txt", encoding="utf-8", errors="ignore").read().split("\f")[page - 1].split("\n")
    if cols:
        lines = [ln[cols[0]:cols[1]] for ln in lines]
    hits = [ln for ln in lines if _re.search(rx, ln)]
    if len(hits) <= n:
        raise ValueError(f"{pid} p.{page}: no line {n} matching {rx!r}")
    ln = hits[n]
    start = _re.search(rx, ln).end()
    if start < len(ln) and (ln[start].isdigit() or ln[start] == "."):  # the match ended inside a number
        while start > 0 and (ln[start - 1].isdigit() or ln[start - 1] == "."):
            start -= 1
    seg = ln[start:]
    if drop_sd:
        seg = _re.sub(r"\(\s*±\s*[\d.]+\s*\)|±\s*[\d.]+", " ", seg)
    return [x.replace("−", "-") for x in _NUM.findall(seg)], " ".join(ln.split())


def R(field, value, page, loc, ev="", item="", unit="", status="reported", note=""):
    """A value printed in the paper (or an approximate chart reading with status='approximate')."""
    if status not in ("reported", "approximate"):
        raise ValueError("use NR() for not_reported and D() for derived values")
    if not page or not (loc or ev):
        raise ValueError(f"{item} {field}: page and location or evidence are required")
    return (item, field, str(value), unit, page, loc, ev, status, note)


def NR(field, item="", note=""):
    """A required field the paper does not report."""
    return (item, field, "", "", "", "", "", "not_reported", note)


def D(field, value, note, item="", unit=""):
    """A derived value (rule V7: unit conversions, sums of printed counts, coded categories); the basis goes in note."""
    if not note:
        raise ValueError(f"{item} {field}: derived values need their basis in note")
    return (item, field, str(value), unit, "", "", "", "derived", note)


class Tags(dict):
    """Default result tags of a paper; call it with keyword arguments to override some of them for one result."""

    def __init__(self, task, level, split, agg, attribute=None):
        super().__init__(task=task, level=level, split=split, agg=agg)
        if attribute:
            self["attribute"] = attribute

    def __call__(self, **over):
        t = Tags(self["task"], self["level"], self["split"], self["agg"], self.get("attribute"))
        t.update(over)
        return t


def avail(code="no", weights="not_stated", data="no", code_url=None, page=None, loc="", ev="", note_code=None,
          note_w=None, note_d=None, data_page=None, data_loc="", data_ev=""):
    rows = []
    if code_url:
        rows.append(R("code_url", code_url, page, loc, ev))
        rows.append(R("code_available", "yes", page, loc, ev))
    elif code == "no":
        rows.append(D("code_available", "no", note_code or "No code repository or availability statement in the paper"))
    else:
        rows.append(R("code_available", code, page, loc, ev))
    if weights == "not_stated":
        rows.append(D("weights_available", "not_stated", note_w or "No statement about trained weights"))
    else:
        rows.append(R("weights_available", weights, page, loc, ev))
    if data_page:
        rows.append(R("data_available", data, data_page, data_loc, data_ev, note=note_d or ""))
    elif data == "no":
        rows.append(D("data_available", "no", note_d or "No data link or availability statement in the paper"))
    else:
        rows.append(D("data_available", data, note_d or ""))
    return rows


def model(item, name, role, cat, backbone, pretraining, strategy, page, loc, ev, pre_page=None, pre_loc="", pre_ev="",
          pre_status="reported", pre_note="", strat_page=None, strat_loc="", strat_ev="", strat_note="", cat_note=""):
    """A model item. cat=None records architecture_category as not reported (architecture not described)."""
    if cat is not None and cat not in CATEGORIES:
        raise ValueError(f"{item}: architecture_category {cat!r} not in {sorted(CATEGORIES)}")
    rows = [R("model_name", name, page, loc, ev, item=item),
            R("model_role", role, page, loc, ev, item=item),
            NR("architecture_category", item=item, note=cat_note or "architecture not described in the paper") if cat is None
            else R("architecture_category", cat, page, loc, ev, item=item, note=cat_note),
            R("backbone", backbone, page, loc, ev, item=item)]
    if pre_status == "not_reported" or pretraining is None:
        rows.append(NR("pretraining", item=item, note=pre_note))
    else:
        rows.append(R("pretraining", pretraining, pre_page or page, pre_loc or loc, pre_ev or ev, item=item, note=pre_note))
    if strategy is None:
        rows.append(NR("training_strategy", item=item, note=strat_note))
    else:
        rows.append(R("training_strategy", strategy, strat_page or page, strat_loc or loc, strat_ev or ev, item=item,
                      note=strat_note))
    return rows


def result(item, metric, mdl, ds, value, page, loc, ev, *, task, level, split, agg, unit="%", primary=False,
           headline=False, setting=None, attribute=None, copied_from=None, note="", status="reported"):
    """A result item with its protocol 6 tags. The tag rows share the value's page, location and evidence."""
    bad = [(k, v) for k, v, ok in (("metric", metric, METRICS), ("task", task, TASKS), ("level", level, LEVELS),
                                   ("split", split, SPLITS), ("agg", agg, AGGS)) if v not in ok]
    if bad:
        raise ValueError(f"{item}: {bad} not allowed")
    if task == "attribute_estimation" and attribute not in ATTRIBUTES:
        raise ValueError(f"{item}: attribute_estimation needs attribute in {sorted(ATTRIBUTES)}")
    if primary and (status == "approximate" or metric == "other"):
        raise ValueError(f"{item}: approximate values and metric 'other' are never primary (protocol 6.3)")
    if metric in BOUNDED and unit not in ("%", "fraction"):
        raise ValueError(f"{item}: {metric} needs unit % or fraction")
    src = (page, loc, ev)
    rows = [R("metric", metric, *src, item=item, note=note if metric == "other" else ""),
            R("result_model", mdl, *src, item=item),
            R("result_dataset", ds, *src, item=item)]
    if setting:
        rows.append(R("result_setting", setting, *src, item=item))
    rows.append(R("value", value, *src, item=item, unit=unit, note=note if metric != "other" else "", status=status))
    if headline:
        rows.append(R("headline", "yes", *src, item=item))
    rows.append(R("result_task", task, *src, item=item))
    if task == "attribute_estimation":
        rows.append(R("attribute", attribute, *src, item=item))
    rows += [R("level", level, *src, item=item), R("split", split, *src, item=item), R("aggregation", agg, *src, item=item)]
    if primary:
        rows.append(R("primary", "yes", *src, item=item))
    if copied_from:
        rows.append(R("copied_from", copied_from, *src, item=item))
    return rows


def conflict(item, target, alternative, page, loc, ev, rule, reason, unit=""):
    """A cf item (rules V1 and V2): the value NOT used, with its own page, location and evidence, and why."""
    if rule not in RULES:
        raise ValueError(f"{item}: rule {rule!r} not in {sorted(RULES)}")
    if not reason:
        raise ValueError(f"{item}: give the reason")
    return [R("cf_target", target, page, loc, ev, item=item), R("cf_alternative", alternative, page, loc, ev, item=item, unit=unit),
            R("cf_rule", rule, page, loc, ev, item=item, note=reason)]


class Counter:
    """Item numbering: res = Counter('res'); res() -> 'res1', 'res2', ..."""

    def __init__(self, prefix):
        self.prefix, self.n = prefix, 0

    def __call__(self):
        self.n += 1
        return f"{self.prefix}{self.n}"
