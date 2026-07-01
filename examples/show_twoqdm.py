#!/usr/bin/env python3
"""Minimal twoqdm usage example."""

from __future__ import annotations

import random
import time

from twoqdm import tqdm


def main() -> None:
    for item in tqdm(range(120), desc="simple twoqdm demo"):
        work_time = 0.025 + item * 0.00035 + random.uniform(0.0, 0.015)
        time.sleep(work_time)


if __name__ == "__main__":
    main()
