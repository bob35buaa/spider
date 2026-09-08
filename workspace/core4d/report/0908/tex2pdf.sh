#!/usr/bin/env bash
# Compile a .tex in this folder to PDF.
# Usage:  ./tex2pdf.sh [file.tex]   (default: algorithm_preview.tex)
set -euo pipefail

cd "$(dirname "$0")"

TEX="${1:-algorithm_preview.tex}"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$TEX"
else
  # fallback: run twice to resolve cross-references
  pdflatex -interaction=nonstopmode -halt-on-error "$TEX"
  pdflatex -interaction=nonstopmode -halt-on-error "$TEX"
fi

echo "PDF -> $(pwd)/${TEX%.tex}.pdf"
