"""twoqdm: tqdm with a live rate trend graph and smarter ETA."""

from __future__ import annotations

from .core import EtaEstimate, TrendTqdm, tqdm, trange

__version__ = "0.1.0"

__all__ = ["EtaEstimate", "TrendTqdm", "__version__", "tqdm", "trange"]
