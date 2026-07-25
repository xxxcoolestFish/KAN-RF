# AAAI-27 Paper Workspace

The submission draft is written in English LaTeX using the official AAAI-27
style files downloaded from the AAAI Author Kit on 2026-07-22.

## Files

- `main.tex`: single-source anonymous paper draft required by the author kit.
- `references.bib`: references cited by the draft.
- `aaai2027.sty`, `aaai2027.bst`: unmodified official style files.
- `official/`: selected official template and checklist sources for reference.

## Build

From this directory, a standard TeX installation can build the manuscript with:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript was compiled successfully with TeX Live 2026 installed at
`C:\texlive\2026`. If that directory is not yet on `PATH`, invoke
`C:\texlive\2026\bin\windows\pdflatex.exe` and `bibtex.exe` directly.

## Draft Status

This is an internal scientific draft, not a submission-ready manuscript.
Claims about a task-relevant effect metric and superiority to inverse-dynamics
transfer remain pending direct experiments. Do not remove those boundaries
until the claim--evidence map is updated.
