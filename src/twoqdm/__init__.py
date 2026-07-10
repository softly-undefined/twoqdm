"""twoqdm: tqdm with a live rate trend graph and smarter ETA."""

from __future__ import annotations

from .core import EtaEstimate, TrendTqdm, available_ascii_spinners, tqdm, trange

__version__ = "0.3.0"

__all__ = [
    "EtaEstimate",
    "TrendTqdm",
    "__version__",
    "available_ascii_spinners",
    "tqdm",
    "trange",
]
