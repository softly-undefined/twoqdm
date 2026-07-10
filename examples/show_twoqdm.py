#!/usr/bin/env python3
"""Local twoqdm feature tour."""

from __future__ import annotations

import random
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from twoqdm import available_ascii_spinners, tqdm, trange


def sleep_jitter(base: float, jitter: float = 0.0) -> None:
    time.sleep(base + random.uniform(0.0, jitter))


def iterator_demo(twoqdm_kwargs: dict[str, Any], *, quick: bool) -> None:
    tqdm.write("\n1/4 iterator mode: normal tqdm kwargs pass through")
    count = 32 if quick else 240
    for item in tqdm(
        range(count),
        desc="iterator",
        unit="row",
        unit_scale=True,
        mininterval=0.08,
        **twoqdm_kwargs,
    ):
        base_sleep = 0.025 + item * 0.0005 if quick else 0.035 + item * 0.00065
        sleep_jitter(base_sleep, 0.025)


def manual_demo(twoqdm_kwargs: dict[str, Any], *, quick: bool) -> None:
    tqdm.write("\n2/4 manual mode: tqdm(total=N) plus update(...)")
    total = 16 if quick else 72
    batch_size = 4
    with tqdm(
        total=total,
        desc="manual",
        unit="chunk",
        leave=False,
        **twoqdm_kwargs,
    ) as bar:
        for batch in range(total // batch_size):
            sleep_jitter(0.14 if quick else 0.42, 0.12)
            bar.update(batch_size)
            if batch in {5, 12} and not quick:
                tqdm.write("  status line written with tqdm.write(...)")


def nested_demo(twoqdm_kwargs: dict[str, Any], *, quick: bool) -> None:
    tqdm.write("\n3/4 nested bars: inner bar uses position=1")
    groups = 2 if quick else 5
    steps = 6 if quick else 16
    for group in trange(
        groups,
        desc="outer",
        unit="task",
        position=0,
        **twoqdm_kwargs,
    ):
        for _ in tqdm(
            range(steps),
            desc=f"inner {group + 1}",
            unit="step",
            position=1,
            leave=False,
            **twoqdm_kwargs,
        ):
            sleep_jitter(0.05 if quick else 0.10, 0.07)


def slow_rate_demo(twoqdm_kwargs: dict[str, Any], *, quick: bool) -> None:
    tqdm.write("\n4/4 slow rates: below 1 it/s displays as s/it")
    total = 2 if quick else 8
    with tqdm(
        total=total,
        desc="slow",
        unit="item",
        mininterval=0.0,
        graph_refresh_interval=0.0,
        **twoqdm_kwargs,
    ) as bar:
        for _ in range(total):
            sleep_jitter(0.75 if quick else 1.85, 0.35)
            bar.update(1)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Local twoqdm feature tour. The default run takes about a minute."
    )
    parser.add_argument(
        "--ascii-spinner",
        choices=("random", "none", *available_ascii_spinners()),
        default="random",
        help="choose the smart ETA animation, random by default",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a shorter tour for smoke testing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    twoqdm_kwargs: dict[str, Any] = {}
    if args.ascii_spinner == "none":
        twoqdm_kwargs["ascii_spinner"] = False
    elif args.ascii_spinner != "random":
        twoqdm_kwargs["ascii_spinner"] = args.ascii_spinner

    tqdm.write(f"Using local checkout: {SRC_DIR}")
    spinner_label = (
        "random per progress bar"
        if args.ascii_spinner == "random"
        else args.ascii_spinner
    )
    tqdm.write(f"ASCII spinner: {spinner_label}")
    iterator_demo(twoqdm_kwargs, quick=args.quick)
    manual_demo(twoqdm_kwargs, quick=args.quick)
    nested_demo(twoqdm_kwargs, quick=args.quick)
    slow_rate_demo(twoqdm_kwargs, quick=args.quick)
    tqdm.write("\nDone.")


if __name__ == "__main__":
    main()
