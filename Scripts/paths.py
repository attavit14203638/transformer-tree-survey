"""Single source of every path used by the project scripts.

Scripts in subfolders add this folder to sys.path and `import paths`.
"""
from pathlib import Path

SURVEY = Path(__file__).resolve().parent.parent
SCRIPTS = SURVEY / "Scripts"
CACHE = SURVEY / ".cache"

# Manuscript (Overleaf repository). In the public repository the manuscript is absent and paper/ holds what the scripts read
# and write for it: the bibliography, the generated numbers and tables, the figures and the reference numbers.
PAPER = SURVEY / "paper"
PAPER_FIGURES = PAPER / "figures"
PAPER_BIB = PAPER / "bib"
PAPER_GENERATED = PAPER / "generated"
PAPER_AUX = PAPER / "out" / "main.aux"       # reference numbers of the compiled manuscript (numbered figure labels)
PAPER_CITATIONS = PAPER / "citation_numbers.csv"  # the same numbers exported for the public repository (bib_key, number)
PAPER_SUPPLEMENT = PAPER / "supplementary"   # supplementary tables, without internal identifiers

# Literature: the single source of truth for the review corpus
LIT = SURVEY / "Literature_Extraction"
SEARCH = LIT / "Search"                      # database searches and screening (PRISMA)
SEARCH_EXPORTS = SEARCH / "exports"
CORPUS = LIT / "Corpus"
PAPERS_CSV = CORPUS / "papers.csv"           # one row per full-text report: IDs, decisions, collection, PDF
PDF_LIBRARY = LIT / "PDF_Library"            # PDFs arranged as the Zotero collections
PDF_TEXT = LIT / "PDF_Text"                  # machine text of each corpus PDF (rebuildable)
EXTRACTION = LIT / "Extraction"
FIELDS_CSV = EXTRACTION / "fields.csv"       # field dictionary for extraction
DATASETS_CSV = EXTRACTION / "datasets.csv"   # registry of dataset keys (protocol rule V9)
EXTRACTION_SCRIPTS = EXTRACTION / "scripts"  # one script per paper that writes its rows (scripts/B##/P###.py)
BATCHES = EXTRACTION / "batches"             # extracted values, one CSV per batch
COMPILED = EXTRACTION / "compiled"           # analysis tables compiled from the batches

# Analysis: figure-ready tables and draft figures built from COMPILED by Scripts/figures
ANALYSIS = LIT / "Analysis"
ANALYSIS_DATA = ANALYSIS / "figure_data"     # the exact values behind every figure, with paper, item and page
ANALYSIS_FIGURES = ANALYSIS / "figures"      # JPEG previews; submission/ holds the journal files (PDF, PNG); copied to PAPER_FIGURES

# Figure script assets
NATURAL_EARTH_ZIP = SCRIPTS / "figures" / "_data" / "ne_110m_admin_0_countries.zip"
