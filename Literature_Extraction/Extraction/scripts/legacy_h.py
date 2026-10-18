"""Row builders for batch extraction. Each row: (item, field, value, unit, page, location, evidence, status, note)."""
import re as _re

_TXT = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "Literature_Extraction" / "PDF_Text")
_NUM = _re.compile(r"(?<![\w.±])[-−]?\d+(?:\.\d+)?(?![\w.])")


def row(pid, page, rx, n=0, cols=None, drop_sd=True):
    """Numbers printed on the n-th layout line of a page that matches rx (after the match), and the line itself.

    cols=(a, b) restricts the line to character positions a..b (two-column pages). Numbers written as '± x' or '(±x)' are
    dropped when drop_sd is True. Returns (numbers, evidence) where evidence is the whitespace-normalised printed line."""
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
    return (item, field, str(value), unit, page, loc, ev, status, note)


def NR(field, item="", note=""):
    return (item, field, "", "", "", "", "", "not_reported", note)


def D(field, value, note, item="", unit=""):
    """Derived value (computed or coded from the whole paper); the basis goes in note."""
    return (item, field, str(value), unit, "", "", "", "derived", note)


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
    rows = [R("model_name", name, page, loc, ev, item=item),
            R("model_role", role, page, loc, ev, item=item),
            R("architecture_category", cat, page, loc, ev, item=item, note=cat_note),
            R("backbone", backbone, page, loc, ev, item=item)]
    if pre_status == "not_reported":
        rows.append(NR("pretraining", item=item, note=pre_note))
    else:
        rows.append(R("pretraining", pretraining, pre_page or page, pre_loc or loc, pre_ev or ev, item=item, note=pre_note))
    if strategy is None:
        rows.append(NR("training_strategy", item=item, note=strat_note))
    else:
        rows.append(R("training_strategy", strategy, strat_page or page, strat_loc or loc, strat_ev or ev, item=item,
                      note=strat_note))
    return rows


def result(item, metric, mdl, ds, value, page, loc, ev, unit="%", headline=False, setting=None, note="", status="reported"):
    rows = [R("metric", metric, page, loc, ev, item=item, note=note if metric == "other" else ""),
            R("result_model", mdl, page, loc, ev, item=item),
            R("result_dataset", ds, page, loc, ev, item=item)]
    if setting:
        rows.append(R("result_setting", setting, page, loc, ev, item=item))
    rows.append(R("value", value, page, loc, ev, item=item, unit=unit, note=note if metric != "other" else "", status=status))
    if headline:
        rows.append(R("headline", "yes", page, loc, ev, item=item))
    return rows


class Counter:
    def __init__(self, prefix):
        self.prefix, self.n = prefix, 0

    def __call__(self):
        self.n += 1
        return f"{self.prefix}{self.n}"
