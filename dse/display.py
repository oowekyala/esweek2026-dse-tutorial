"""Showing MLIR in a notebook with syntax highlighting."""

from __future__ import annotations

from pathlib import Path

from IPython.display import HTML
from pygments import highlight
from pygments.formatters import HtmlFormatter

from .mlir_lexer import MlirLexer

# Inline styles rather than a stylesheet: the output then survives nbconvert,
# an editor's notebook view and Binder alike, none of which share a <head>.
_FORMATTER = HtmlFormatter(style="friendly", noclasses=True, nowrap=True)


def show_mlir(source: str | Path, *, max_lines: int | None = None) -> HTML:
    """Render MLIR text -- or the file at ``source`` -- highlighted.

    ``max_lines`` truncates long IR, saying how much was left out.
    """
    if isinstance(source, Path):
        text = source.read_text()
    else:
        text = source
    lines = text.rstrip("\n").split("\n")
    note = ""
    if max_lines is not None and len(lines) > max_lines:
        note = f"<div style='color:#52514e;font-family:monospace'>… {len(lines) - max_lines} more lines</div>"
        lines = lines[:max_lines]
    body = highlight("\n".join(lines) + "\n", MlirLexer(), _FORMATTER)
    return HTML(
        "<pre style='margin:0;padding:.6em .8em;line-height:1.35;"
        "font-size:.9em;overflow-x:auto;background:#fcfcfb;"
        "border-left:3px solid #2a78d6'>"
        f"{body}</pre>{note}"
    )
