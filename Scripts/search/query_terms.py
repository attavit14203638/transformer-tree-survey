"""Single source of truth for the systematic search strategy.

The search combines three concept blocks with AND; terms within a block are combined with OR.
`render_scopus()`, `render_wos()` and `render_arxiv()` produce the exact strings used in each
source, and `matches()` approximates Scopus TITLE-ABS-KEY matching (case-insensitive, loose
phrases, trailing-wildcard truncation, simple plurals) for the recall test on known studies.
"""
import re

# Block A: transformer architectures and transformer-based foundation / vision-language models.
BLOCK_A = [
    "transformer*", "vision transformer*", "self-attention",
    "swin", "segformer", "detr", "mask2former",
    "foundation model*", "segment anything", "vision-language", "large language model*",
    "grounding dino", "clip",
]

# Block B: tree-level targets and closely related forest structure / change tasks.
BLOCK_B = [
    "tree*", "individual tree*", "tree crown*", "crown delineation", "canopy", "canopies",
    "date palm*", "mangrove*", "forest*", "deforestation", "woodland*", "orchard*",
    "wetland*", "vegetation", "biomass", "plantation*",
]

# Block C: remote sensing platforms and sensors.
BLOCK_C = [
    "remote sensing", "remotely sensed", "earth observation", "uav*", "drone*",
    "unmanned aerial", "aerial", "airborne", "satellite*", "lidar", "laser scanning",
    "point cloud*", "hyperspectral", "multispectral", "sar", "synthetic aperture radar",
    "sentinel*", "landsat", "orthophoto*", "orthomosaic*", "low-altitude",
]

BLOCKS = {"A_transformer": BLOCK_A, "B_tree": BLOCK_B, "C_remote_sensing": BLOCK_C}

YEAR_FROM, YEAR_TO = 2017, 2025
INDEX_CUTOFF = "20260101"  # records loaded/indexed before 1 January 2026


def _q(term):
    return f'"{term}"' if (" " in term or "-" in term) else term


def render_scopus():
    blocks = " AND ".join("(" + " OR ".join(_q(t) for t in terms) + ")" for terms in BLOCKS.values())
    return (
        f"TITLE-ABS-KEY({blocks}) AND PUBYEAR > {YEAR_FROM - 1} AND PUBYEAR < {YEAR_TO + 1} "
        f"AND ORIG-LOAD-DATE BEF {INDEX_CUTOFF} AND LANGUAGE(english) "
        "AND (DOCTYPE(ar) OR DOCTYPE(cp) OR DOCTYPE(ch))"
    )


def render_scopus_fallback():
    """Narrower variant (tree terms restricted to title and author keywords), for use only if
    the primary query returns an unmanageable number of records."""
    a = " OR ".join(_q(t) for t in BLOCKS["A_transformer"])
    b = " OR ".join(_q(t) for t in BLOCKS["B_tree"])
    c = " OR ".join(_q(t) for t in BLOCKS["C_remote_sensing"])
    return (
        f"TITLE-ABS-KEY({a}) AND (TITLE({b}) OR AUTHKEY({b})) AND TITLE-ABS-KEY({c}) "
        f"AND PUBYEAR > {YEAR_FROM - 1} AND PUBYEAR < {YEAR_TO + 1} "
        f"AND ORIG-LOAD-DATE BEF {INDEX_CUTOFF} AND LANGUAGE(english) "
        "AND (DOCTYPE(ar) OR DOCTYPE(cp) OR DOCTYPE(ch))"
    )


def render_wos():
    blocks = " AND ".join("(" + " OR ".join(_q(t) for t in terms) + ")" for terms in BLOCKS.values())
    return (
        f"TS=({blocks}) AND PY=({YEAR_FROM}-{YEAR_TO}) AND LA=(English) "
        "AND DT=(Article OR Proceedings Paper OR Early Access)"
    )


# Explicit word forms for arXiv, whose API has no trailing wildcard.
ARXIV_FORMS = {"forest*": ["forest", "forests", "forestry"], "uav*": ["uav", "uavs"],
               "sentinel*": ["sentinel"], "satellite*": ["satellite", "satellites"]}


def _arxiv_terms(term):
    forms = ARXIV_FORMS.get(term)
    if forms is None:
        stem = term.rstrip("*").replace("-", " ")
        forms = [stem, stem + "s"] if term.endswith("*") else [stem]
    return [f'abs:"{f}"' if " " in f else f"abs:{f}" for f in forms]


def render_arxiv():
    # abs: searches the abstract; the API also matches the title through the abstract index
    # only partially, so title terms are added with ti: in arxiv_search.py.
    blocks = " AND ".join(
        "(" + " OR ".join(x for t in terms for x in _arxiv_terms(t)) + ")" for terms in BLOCKS.values()
    )
    return blocks


# ---------------------------------------------------------------------------
# Local matcher (approximation of Scopus TITLE-ABS-KEY semantics)
# ---------------------------------------------------------------------------

def _tokens(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()


def _token_match(tok, pat):
    if pat.endswith("*"):
        return tok.startswith(pat[:-1])
    return tok in (pat, pat + "s", pat + "es")


def term_in(term, toks):
    parts = re.sub(r"[^a-z0-9*]+", " ", term.lower()).split()
    n = len(parts)
    for i in range(len(toks) - n + 1):
        if all(_token_match(toks[i + j], parts[j]) for j in range(n)):
            return True
    return False


def matches(title, abstract, keywords=""):
    toks = _tokens(" ".join([title or "", abstract or "", keywords or ""]))
    hits = {name: [t for t in terms if term_in(t, toks)] for name, terms in BLOCKS.items()}
    return all(hits.values()), hits


if __name__ == "__main__":
    print("SCOPUS:\n" + render_scopus() + "\n")
    print("SCOPUS (fallback):\n" + render_scopus_fallback() + "\n")
    print("WEB OF SCIENCE:\n" + render_wos() + "\n  then refine: Index Date 2017-01-01 to 2025-12-31\n")
    print("ARXIV API (search_query):\n" + render_arxiv())
