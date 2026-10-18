"""Systematic arXiv search via the public arXiv API.

Step 1 (API): retrieve candidates matching Block A AND Block B (explicit word forms, because the
API does not support truncation) in cs.CV, eess.IV, cs.AI or cs.LG, submitted 1 Jan 2017 to
31 Dec 2025.
Step 2 (local): keep candidates whose title+abstract satisfy all three concept blocks using the
same matcher as the recall test (query_terms.matches).

Outputs (Literature_Extraction/Search/results/):
  arxiv_candidates.csv  all API candidates
  arxiv_records.csv     candidates passing the full three-block query (records identified)
  arxiv_run.json        run timestamp (UTC), API query and counts
"""
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import query_terms as q

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}
OUT = str(paths.SEARCH / "results")
PAGE = 100
DELAY_S = 3.5


CATEGORIES = ["cs.CV", "eess.IV", "cs.AI", "cs.LG"]


def api_query():
    def forms(block):
        return " OR ".join(f"all:{x.split(':', 1)[1]}" for t in q.BLOCKS[block] for x in q._arxiv_terms(t))
    cats = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    return (f"({forms('A_transformer')}) AND ({forms('B_tree')}) AND ({cats}) "
            "AND submittedDate:[201701010000 TO 202512312359]")


def fetch(query, start, retries=5):
    # Note: sortBy=submittedDate or large pages make the API return HTTP 500 for this query.
    params = urllib.parse.urlencode({"search_query": query, "start": start, "max_results": PAGE})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{API}?{params}", timeout=120) as r:
                return ET.fromstring(r.read())
        except urllib.error.HTTPError as e:
            if attempt == retries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or (60 * (attempt + 1) if e.code == 429 else DELAY_S * (attempt + 2))
            print(f"HTTP {e.code} at start={start}; retrying in {wait}s")
            time.sleep(wait)


def parse(entry):
    text = lambda path: (entry.findtext(path, default="", namespaces=NS) or "").strip()
    return {
        "arxiv_id": re.sub(r"v\d+$", "", text("a:id").rsplit("/abs/", 1)[-1]),
        "title": " ".join(text("a:title").split()),
        "abstract": " ".join(text("a:summary").split()),
        "authors": "; ".join(a.findtext("a:name", default="", namespaces=NS) for a in entry.findall("a:author", NS)),
        "published": text("a:published")[:10],
        "primary_category": entry.find("arxiv:primary_category", NS).get("term") if entry.find("arxiv:primary_category", NS) is not None else "",
        "categories": " ".join(c.get("term") for c in entry.findall("a:category", NS)),
        "doi": text("arxiv:doi"),
        "journal_ref": text("arxiv:journal_ref"),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    query = api_query()
    run_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    first = fetch(query, 0)
    total = int(first.findtext("os:totalResults", namespaces=NS))
    records = [parse(e) for e in first.findall("a:entry", NS)]
    for start in range(PAGE, total, PAGE):
        time.sleep(DELAY_S)
        page = fetch(query, start)
        records += [parse(e) for e in page.findall("a:entry", NS)]
        print(f"retrieved {len(records)}/{total}")
    seen, uniq = set(), []
    for r in records:
        if r["arxiv_id"] not in seen:
            seen.add(r["arxiv_id"])
            uniq.append(r)
    if len(uniq) != total:
        print(f"WARNING: {len(uniq)} unique records retrieved but API reported {total}")
    for r in uniq:
        ok, hits = q.matches(r["title"], r["abstract"])
        r["passes_query"] = ok
        r["blocks_matched"] = ";".join(k for k, v in hits.items() if v)
    fields = list(uniq[0])
    with open(os.path.join(OUT, "arxiv_candidates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(uniq)
    kept = [r for r in uniq if r["passes_query"]]
    with open(os.path.join(OUT, "arxiv_records.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    meta = {"run_at_utc": run_at, "api_endpoint": API, "api_query": query,
            "api_total_results": total, "candidates_unique": len(uniq), "records_passing_query": len(kept)}
    json.dump(meta, open(os.path.join(OUT, "arxiv_run.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
