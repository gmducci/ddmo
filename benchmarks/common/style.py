"""Shared publication plotting style."""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl

METHOD_LABELS = {
    "fixed_library": "Oracle library",
    "adaptive_library": "Adaptive library",
}

METHOD_COLORS = {
    "fixed_library": "#3B4CC0",
    "adaptive_library": "#B40426",
}

METHOD_MARKERS = {
    "fixed_library": "s",
    "adaptive_library": "o",
}


@contextmanager
def publication_style():
    """Apply a compact journal-friendly Matplotlib style temporarily."""
    settings = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 5.5,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with mpl.rc_context(settings):
        yield
