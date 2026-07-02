#!/usr/bin/env python3
"""Local twoqdm feature tour."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from twoqdm import tqdm, trange


def sleep_jitter(base: float, jitter: float = 0.0) -> None:
    time.sleep(base + random.uniform(0.0, jitter))


def iterator_demo() -> None:
    tqdm.write("\n1/4 iterator mode: normal tqdm kwargs pass through")
    for item in tqdm(
        range(80),
        desc="iterator",
        unit="row",
        unit_scale=True,
        dynamic_ncols=True,
        mininterval=0.08,
    ):
        sleep_jitter(0.045 + item * 0.0008, 0.025)


def manual_demo() -> None:
    tqdm.write("\n2/4 manual mode: tqdm(total=N) plus update(...)")
    with tqdm(total=32, desc="manual", unit="chunk", leave=False) as bar:
        for batch in range(8):
            sleep_jitter(0.32, 0.12)
            bar.update(4)
            if batch == 3:
                tqdm.write("  status line written with tqdm.write(...)")


def nested_demo() -> None:
    tqdm.write("\n3/4 nested bars: inner bar uses position=1")
    for group in trange(3, desc="outer", unit="task", position=0):
        for _ in tqdm(
            range(10),
            desc=f"inner {group + 1}",
            unit="step",
            position=1,
            leave=False,
        ):
            sleep_jitter(0.12, 0.06)


def slow_rate_demo() -> None:
    tqdm.write("\n4/4 slow rates: below 1 it/s displays as s/it")
    with tqdm(
        total=4,
        desc="slow",
        unit="item",
        mininterval=0.0,
        graph_refresh_interval=0.0,
    ) as bar:
        for _ in range(4):
            sleep_jitter(1.85, 0.25)
            bar.update(1)


def main() -> None:
    tqdm.write(f"Using local checkout: {SRC_DIR}")
    iterator_demo()
    manual_demo()
    nested_demo()
    slow_rate_demo()
    tqdm.write("\nDone.")


if __name__ == "__main__":
    main()
