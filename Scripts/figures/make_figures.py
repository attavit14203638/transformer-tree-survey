"""Build all figures (or the named ones): Scripts/.venv/bin/python Scripts/figures/make_figures.py [--submission] [prisma_flow ...].

Without options: JPEG previews in Literature_Extraction/Analysis/figures. With --submission: also the files for the journal in
Literature_Extraction/Analysis/figures/submission (JPEGs at least 7600 px wide, with a uniform margin), copied to the
manuscript figures/ folder. With --numbered: studies are
labelled "First author et al. [n]" with the reference numbers of the compiled manuscript (run after the text is final)."""
import importlib
import sys

import common
from common import set_style

FIGURES = ["prisma_flow", "timeline", "taxonomy", "overview", "geography", "rq1_comparisons", "rq2_transfer", "rq3_training",
           "study_breakdown"]  # the last one writes Table 3

if __name__ == "__main__":
    set_style()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    common.SUBMISSION = "--submission" in sys.argv[1:]
    common.NUMBERED = "--numbered" in sys.argv[1:]
    for name in args or FIGURES:
        print(name)
        importlib.import_module(name).main()
