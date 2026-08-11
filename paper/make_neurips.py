#!/usr/bin/env python3
"""Derive the NeurIPS workshop submission from the canonical ACL source.

`paper/main.tex` stays the single source of truth. This script rewrites only the
things that are genuinely venue-specific -- preamble, anonymisation, single-column
geometry, bibliography style -- and emits `paper/neurips/main.tex`. Re-run it after
every edit to main.tex rather than hand-maintaining two copies that will drift.

Target: TAE (Trust-AI-Eval), NeurIPS 2026 workshop. Double-blind, non-archival,
8 pages excluding references and appendices.

Usage:  python paper/make_neurips.py && (cd paper/neurips && tectonic -X compile main.tex)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "main.tex")
OUTDIR = os.path.join(HERE, "neurips")
OUT = os.path.join(OUTDIR, "main.tex")

WORKSHOP = "TAE (Trust-AI-Eval): Can We Trust AI Evaluation?"

# neurips_2026.sty itself loads only environ, natbib, geometry and lineno -- the
# package list below lives in the official neurips_2026.tex template, not the style
# file, so it has to be reproduced here verbatim (plus the four this paper adds).
PREAMBLE = r"""%% GENERATED FILE -- DO NOT EDIT.
%% Produced from ../main.tex by paper/make_neurips.py. Edit that source instead.
%%
%% Venue: TAE (Trust-AI-Eval), NeurIPS 2026 workshop.
%%   double-blind  -> the style file prints "Anonymous Author(s)" automatically
%%   non-archival  -> compatible with a later ACL/ARR submission of the same work
%%   8 pages, excluding references and appendices
\documentclass{article}

\usepackage[dblblindworkshop]{neurips_2026}
\workshoptitle{%s}

%% --- verbatim from the official neurips_2026.tex template ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
%% --- added by this paper ---
\usepackage{graphicx}
\usepackage{amssymb}
\usepackage{inconsolata}
""" % WORKSHOP


def convert(s):
    steps = []

    # 1. Replace everything from \documentclass up to the paper's own macros.
    i = s.index("\\newcommand{\\ona}")
    s = PREAMBLE + "\n" + s[i:]
    steps.append("preamble -> neurips_2026 [dblblindworkshop] + \\workshoptitle")

    # 2. Drop the author block. Under double-blind the style file ignores its content
    #    and prints "Anonymous Author(s)"; removing it outright means the real names
    #    cannot leak into the submitted PDF's metadata.
    m = re.search(r"\\author\{.*?\n\}\n", s, re.S)
    if m:
        s = s[:m.start()] + "% author block removed: double-blind submission\n" + s[m.end():]
        steps.append("author block removed (double-blind)")

    # 3. Single column: \columnwidth is the full text width here.
    n = s.count("\\columnwidth")
    s = s.replace("\\columnwidth", "\\linewidth")
    steps.append(f"\\columnwidth -> \\linewidth ({n})")

    # 4. Let the tables breathe. They were hand-tightened to fit ACL's 3.05in column;
    #    NeurIPS gives 5.5in, where that padding just looks cramped.
    n = len(re.findall(r"\\setlength\{\\tabcolsep\}\{[^}]*\}\n", s))
    s = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}\n", "", s)
    steps.append(f"removed {n} ACL tabcolsep overrides (default 6pt)")

    # 5. acl.sty issued \bibliographystyle for us; neurips_2026.sty does not.
    #    NeurIPS accepts any consistent style, and the paper cites author-year.
    s = s.replace("\\bibliography{references}",
                  "\\bibliographystyle{plainnat}\n\\bibliography{references}")
    steps.append("added \\bibliographystyle{plainnat}")

    # 6. Anonymise the code link. This is the one substantive claim in the paper that
    #    de-anonymises it, and a double-blind submission must not carry it.
    if "github.com/idrock-ai" in s:
        s = re.sub(r"\\url\{https://github\.com/idrock-ai/[^}]*\}",
                   r"\\url{https://anonymous.4open.science/r/prompt-optimization-audit}", s)
        steps.append("code URL anonymised -> anonymous.4open.science")

    # 6b. Neutralise ownership language. The DTM citations themselves are fine --
    #     citing a publicly released dataset in the third person is permitted under
    #     double-blind -- but "our benchmark" next to "DTM [Hazratov et al.]" lets a
    #     reviewer infer that the dataset's authors and this paper's authors are the
    #     same people. The ACL version keeps the natural phrasing.
    n = s.count("our benchmark")
    if n:
        s = s.replace("our benchmark", "this benchmark")
        steps.append(f"de-anonymising phrase 'our benchmark' -> 'this benchmark' ({n})")

    # 7. Fit the 8-page limit. NeurIPS counts neither references nor appendices, so
    #    the sanctioned lever is to move supporting material back. Nothing is deleted:
    #    every table and statement still ships, just after \appendix. We move only
    #    material the argument does not depend on -- two secondary tables, and two
    #    statements that are pointers rather than evidence.
    moved = []

    def lift(pattern, what):
        """Cut a self-contained block out of the body and return it."""
        nonlocal s
        m = re.search(pattern, s, re.S)
        if not m:
            return ""
        s = s[:m.start()] + s[m.end():]
        moved.append(what)
        return m.group(0).rstrip() + "\n\n"

    tables = ""
    for label, what in (("tab:allors", "Table: every odds ratio"),
                        ("tab:permodel", "Table: powered audit per model"),
                        ("tab:dose", "Table: budget dose-response"),
                        # Sec. 5 itself calls this arm an observation it will not promote
                        # to a finding, so it is the most expendable float in the body.
                        ("tab:substitution", "Table: demonstration ablation")):
        tables += lift(r"\\begin\{table\}(?:(?!\\end\{table\}).)*?\\label\{" + label
                       + r"\}.*?\\end\{table\}\n", what)
    # Sec. 6's figure: its every number is stated in the text, so the float is
    # supporting rather than load-bearing.
    tables += lift(r"\\begin\{figure\}(?:(?!\\end\{figure\}).)*?\\label\{fig:decomp\}"
                   r".*?\\end\{figure\}\n", "Figure: flip decomposition")

    stmts = ""
    for head, what in (("Ethics Statement", "Ethics Statement"),
                       ("Reproducibility Statement", "Reproducibility Statement")):
        stmts += lift(r"\\section\*\{" + head + r"\}.*?(?=\\section\*|% acl\.sty|\\bibliography)",
                      what)

    if tables or stmts:
        extra = "\\section{Supporting Tables and Statements}\\label{app:supporting}\n\n"
        s = s.replace("\\appendix\n", "\\appendix\n\n" + extra + tables + stmts, 1)
        steps.append("moved to appendix (8-page limit): " + "; ".join(moved))

    return s, steps


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC}")
    os.makedirs(OUTDIR, exist_ok=True)
    out, steps = convert(open(SRC).read())
    with open(OUT, "w") as f:
        f.write(out)
    print(f"wrote {OUT}")
    for s in steps:
        print(f"  - {s}")
    for name in ("references.bib", "figures"):
        link = os.path.join(OUTDIR, name)
        if not os.path.exists(link):
            os.symlink(os.path.join("..", name), link)
            print(f"  - linked {name}")
    if not os.path.exists(os.path.join(OUTDIR, "neurips_2026.sty")):
        print("\nWARNING: paper/neurips/neurips_2026.sty is missing. Download it from")
        print("https://media.neurips.cc/Conferences/NeurIPS2026/"
              "Formatting_Instructions_For_NeurIPS_2026.zip")


if __name__ == "__main__":
    main()
