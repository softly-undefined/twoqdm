"""Trend-aware tqdm wrapper with a terminal rate graph and smarter ETA."""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TypeVar

try:
    from tqdm import tqdm as base_tqdm
except ImportError as exc:  # pragma: no cover - useful when copied into projects
    raise SystemExit(
        "tqdm is not installed. Install it with: python3 -m pip install tqdm"
    ) from exc


T = TypeVar("T")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
CYAN = "\x1b[38;5;45m"
GREEN = "\x1b[38;5;82m"
YELLOW = "\x1b[38;5;220m"
ORANGE = "\x1b[38;5;214m"
RED = "\x1b[38;5;203m"
BLUE = "\x1b[38;5;111m"
GRAY = "\x1b[38;5;245m"
DARK_GRAY = "\x1b[38;5;240m"
UP_GRADIENT = [
    YELLOW,
    "\x1b[38;5;190m",
    "\x1b[38;5;154m",
    "\x1b[38;5;118m",
    GREEN,
]
DOWN_GRADIENT = [
    YELLOW,
    "\x1b[38;5;222m",
    ORANGE,
    "\x1b[38;5;208m",
    RED,
]


@dataclass(frozen=True)
class RateBucket:
    low: float
    mean: float
    high: float
    count: int


@dataclass(frozen=True)
class RateGraph:
    lines: list[str]
    scale_low: float
    scale_high: float
    visible_samples: int
    clipped_samples: int
    trace_deltas: list[float | None]


@dataclass(frozen=True)
class EtaEstimate:
    seconds: float | None
    label: str
    confidence: float | None = None
    capped: bool = False


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    pct = min(max(pct, 0.0), 1.0)
    position = pct * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compact_rate(value: float) -> str:
    if value >= 1000:
        return f"{value:.0f}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def bucket_rate_stats(values: list[float], width: int) -> list[RateBucket]:
    if len(values) <= width:
        return [RateBucket(value, value, value, 1) for value in values]

    bucket = len(values) / width
    result = []
    for i in range(width):
        start = int(i * bucket)
        stop = max(int((i + 1) * bucket), start + 1)
        sample = values[start:stop]
        result.append(
            RateBucket(min(sample), sum(sample) / len(sample), max(sample), len(sample))
        )
    return result


def smooth_values(values: list[float], radius: int = 2) -> list[float]:
    if radius <= 0 or len(values) <= 2:
        return values

    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        sample = values[start:stop]
        smoothed.append(sum(sample) / len(sample))
    return smoothed


def rate_scale(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0

    if len(values) >= 18:
        low = min(percentile(values, 0.03), values[-1])
        high = max(percentile(values, 0.97), values[-1])
    else:
        low = min(values)
        high = max(values)

    if math.isclose(low, high):
        padding = max(abs(high) * 0.08, 1e-6)
    else:
        padding = (high - low) * 0.10

    return max(0.0, low - padding), high + padding


def value_to_graph_row(value: float, low: float, high: float, height: int) -> int:
    if height <= 1 or math.isclose(low, high):
        return max(0, height // 2)

    value = min(max(value, low), high)
    return round((high - value) / (high - low) * (height - 1))


def render_rate_graph(
    values: list[float],
    width: int = 68,
    height: int = 8,
    history_factor: int = 3,
) -> RateGraph:
    """Return a terminal graph of recent it/s values."""
    if not values:
        return RateGraph(
            [" " * width for _ in range(height)],
            0.0,
            1.0,
            0,
            0,
            [None] * width,
        )

    history_limit = max(width, width * history_factor)
    visible_values = values[-history_limit:]
    scale_low, scale_high = rate_scale(visible_values)
    buckets = bucket_rate_stats(visible_values, width)
    rows = [[" " for _ in range(width)] for _ in range(height)]
    last_mean_row: int | None = None
    mean_values = smooth_values([bucket.mean for bucket in buckets])
    mean_rows: list[int] = []
    trace_deltas: list[float | None] = [None] * width
    band_threshold = max((scale_high - scale_low) * 0.08, 1e-9)

    for col, bucket in enumerate(buckets):
        if bucket.count <= 1 or bucket.high - bucket.low < band_threshold:
            continue

        high_row = value_to_graph_row(bucket.high, scale_low, scale_high, height)
        low_row = value_to_graph_row(bucket.low, scale_low, scale_high, height)
        for row in range(min(high_row, low_row), max(high_row, low_row) + 1):
            rows[row][col] = "."

    for col, mean_value in enumerate(mean_values):
        mean_row = value_to_graph_row(mean_value, scale_low, scale_high, height)
        mean_rows.append(mean_row)
        trace_deltas[col] = mean_value - mean_values[col - 1] if col > 0 else 0.0

        if last_mean_row is not None:
            start_row = min(last_mean_row, mean_row)
            stop_row = max(last_mean_row, mean_row) + 1
            for row in range(start_row, stop_row):
                rows[row][col] = ":"
        rows[mean_row][col] = "-"
        last_mean_row = mean_row

    if mean_rows:
        rows[mean_rows[-1]][len(mean_rows) - 1] = ">"

    clipped = sum(value < scale_low or value > scale_high for value in visible_values)
    return RateGraph(
        ["".join(row) for row in rows],
        scale_low,
        scale_high,
        len(visible_values),
        clipped,
        trace_deltas,
    )


def fitted_graph_label(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[:width]


def visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def fitted_terminal_line(text: str, width: int) -> str:
    if visible_len(text) <= width:
        return text

    visible = 0
    index = 0
    result = []
    while index < len(text) and visible < width:
        match = ANSI_RE.match(text, index)
        if match:
            result.append(match.group(0))
            index = match.end()
            continue
        result.append(text[index])
        visible += 1
        index += 1

    if ANSI_RE.search("".join(result)):
        result.append(RESET)
    return "".join(result)


def color_enabled(file) -> bool:
    return (
        file.isatty()
        and os.environ.get("TQDM_TREND_NO_COLOR") is None
        and os.environ.get("TERM") != "dumb"
    )


def colored(text: str, color: str, *, enabled: bool) -> str:
    if not enabled or not text:
        return text
    return f"{color}{text}{RESET}"


def trace_color(delta: float | None, span: float) -> str:
    if delta is None or span <= 0:
        return YELLOW

    if math.isclose(delta, 0.0, abs_tol=span * 0.002):
        return YELLOW

    gradient = UP_GRADIENT if delta > 0 else DOWN_GRADIENT
    strength = min(1.0, abs(delta) / max(span * 0.035, 1e-9))
    index = round(strength * (len(gradient) - 1))
    return gradient[index]


def color_graph_line(line: str, *, graph: RateGraph, enabled: bool) -> str:
    if not enabled:
        return line

    result = []
    active = ""
    span = graph.scale_high - graph.scale_low
    for col, char in enumerate(line):
        if char == ".":
            color = DARK_GRAY
        elif char in {"-", ":", ">"}:
            delta = graph.trace_deltas[col] if col < len(graph.trace_deltas) else None
            color = trace_color(delta, span)
            if char == ">":
                color += BOLD
        else:
            color = ""
        if color != active:
            if active:
                result.append(RESET)
            if color:
                result.append(color)
            active = color
        result.append(char)
    if active:
        result.append(RESET)
    return "".join(result)


def label_color(label: str) -> str:
    if label.startswith("fast"):
        return CYAN
    if label.startswith("avg"):
        return YELLOW
    if label.startswith("slow"):
        return RED
    return GRAY


def info_color(text: str) -> str:
    if text.startswith("smart ETA"):
        return CYAN + BOLD
    if "exp slowing" in text or text.startswith("falling"):
        return RED
    if "slowing" in text:
        return ORANGE
    if text.startswith("rising") or "speeding up" in text:
        return GREEN
    if text.startswith("flat"):
        return BLUE
    if "warming" in text:
        return YELLOW
    if text.startswith("fit"):
        return YELLOW
    return GRAY


def terminal_size() -> os.terminal_size:
    return shutil.get_terminal_size((100, 24))


def info_width_for_terminal(columns: int) -> int:
    if columns >= 120:
        return 32
    if columns >= 90:
        return 28
    if columns >= 64:
        return 22
    return max(10, columns // 4)


def graph_width_for_terminal(
    requested_width: int,
    label_width: int,
    info_width: int,
    columns: int,
) -> int:
    available = max(4, columns - label_width - info_width - 4)
    return min(requested_width, available)


def render_rate_panel(
    graph: RateGraph,
    *,
    height: int,
    label_width: int,
    info_width: int,
    visible_avg: float,
    info: list[str],
    use_color: bool,
) -> list[str]:
    labels = [""] * height
    if height:
        labels[0] = f"fast {compact_rate(graph.scale_high)}"
        labels[height // 2] = f"avg {compact_rate(visible_avg)}"
        labels[-1] = f"slow {compact_rate(graph.scale_low)}"

    panel = []
    for row, (label, line) in enumerate(zip(labels, graph.lines)):
        label = fitted_graph_label(label, label_width)
        label_text = f"{label:<{label_width}}"
        label_text = colored(label_text, label_color(label), enabled=use_color)
        rail = colored("|", GRAY, enabled=use_color)
        graph_line = color_graph_line(line, graph=graph, enabled=use_color)
        right = info[row] if row < len(info) else ""
        right = fitted_terminal_line(right, info_width)
        right_text = f"{right:<{info_width}}"
        right_text = colored(right_text, info_color(right), enabled=use_color)
        panel.append(f"{label_text}{rail}{graph_line}{rail} {right_text}")
    return panel


def smart_info_lines(
    estimate: EtaEstimate,
    *,
    direction: str,
    current_rate: float,
    graph: RateGraph,
    completed: int,
    total: int | None,
    height: int,
) -> list[str]:
    confidence = (
        f"fit {estimate.confidence:.0%}" if estimate.confidence is not None else ""
    )
    clipped = f", clipped {graph.clipped_samples}" if graph.clipped_samples else ""
    lines = [
        f"smart ETA {format_seconds(estimate.seconds)}",
        f"model {estimate.label}",
        confidence,
        direction,
        f"now {current_rate:.2f} it/s",
        f"window {graph.visible_samples}{clipped}",
        f"{completed}/{total}" if total is not None else f"{completed}",
    ]
    return (lines + [""] * height)[:height]


def reserve_terminal_panel(file, height: int) -> bool:
    if height <= 0 or not file.isatty():
        return False

    file.write("\n" * height)
    file.flush()
    return True


def draw_terminal_panel(file, lines: list[str], columns: int) -> None:
    if not lines:
        return

    height = len(lines)
    file.write("\x1b7")
    file.write(f"\x1b[{height}A")
    for row, line in enumerate(lines):
        file.write("\r\x1b[2K")
        file.write(fitted_terminal_line(line, columns - 1))
        if row < height - 1:
            file.write("\n")
    file.write("\x1b8")
    file.flush()


def rate_direction(values: list[float]) -> str:
    if len(values) < 10:
        return f"warming {len(values)}/10"

    recent = values[-min(60, len(values)) :]
    half = len(recent) // 2
    older = sum(recent[:half]) / half
    newer = sum(recent[half:]) / (len(recent) - half)
    change = (newer - older) / older if older else 0.0
    mean = sum(recent) / len(recent)
    variance = sum((value - mean) ** 2 for value in recent) / len(recent)
    jitter = math.sqrt(variance) / mean if mean else 0.0
    if change <= -0.2:
        return f"falling {abs(change):.0%}, jitter {jitter:.0%}"
    if change >= 0.2:
        return f"rising {change:.0%}, jitter {jitter:.0%}"
    return f"flat, jitter {jitter:.0%}"


def smart_eta(durations: list[float], remaining: int) -> EtaEstimate:
    """Estimate remaining time from recent duration trend."""
    if remaining <= 0:
        return EtaEstimate(0.0, "done", confidence=1.0)
    if len(durations) < 8:
        return EtaEstimate(None, "warming")

    recent = durations[-min(40, len(durations)) :]
    avg_eta = sum(recent[-min(12, len(recent)) :]) / min(12, len(recent)) * remaining
    xs = list(range(len(recent)))
    ys = [math.log(max(value, 1e-9)) for value in recent]
    x_avg = sum(xs) / len(xs)
    y_avg = sum(ys) / len(ys)
    denom = sum((x - x_avg) ** 2 for x in xs)
    slope = sum((x - x_avg) * (y - y_avg) for x, y in zip(xs, ys)) / denom
    intercept = y_avg - slope * x_avg
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    total_variance = sum((y - y_avg) ** 2 for y in ys)
    r_squared = 1.0 - residual / total_variance if total_variance else 0.0

    if abs(slope) < 0.02 or r_squared < 0.55:
        return EtaEstimate(avg_eta, "recent avg")

    projected = 0.0
    capped = False
    for step in range(len(recent), len(recent) + remaining):
        projected += math.exp(intercept + slope * step)
        if projected > avg_eta * 50 and projected > 60:
            capped = True
            break

    if slope > 0.03:
        label = "exp slowing"
    elif slope > 0:
        label = "slowing"
    else:
        label = "speeding up"
    if capped:
        label += ", unstable"
    return EtaEstimate(projected, label, confidence=r_squared, capped=capped)


class TrendTqdm(Iterable[T]):
    """A tqdm-compatible iterable wrapper with a rate trend panel."""

    def __init__(
        self,
        iterable: Iterable[T],
        *,
        desc: str = "",
        total: int | None = None,
        graph_width: int = 68,
        graph_height: int = 8,
        graph_refresh_interval: float = 0.12,
    ) -> None:
        self.iterable = iterable
        if total is None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        self.total = total
        self.desc = desc
        self.graph_width = graph_width
        self.graph_height = graph_height
        self.graph_refresh_interval = graph_refresh_interval
        self.durations: list[float] = []
        self.rates: list[float] = []

    def __iter__(self) -> Iterator[T]:
        output = sys.stderr
        size = terminal_size()
        panel_height = min(self.graph_height, max(3, size.lines - 4))
        label_width = 12
        info_width = info_width_for_terminal(size.columns)
        use_color = color_enabled(output)
        panel_reserved = reserve_terminal_panel(output, panel_height)
        progress = base_tqdm(
            total=self.total,
            desc=self.desc,
            leave=True,
            dynamic_ncols=True,
            colour="cyan" if use_color else None,
            bar_format=(
                "{l_bar}{bar}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
            ),
            file=output,
        )

        try:
            last = time.perf_counter()
            next_graph_refresh = 0.0
            completed = 0
            for item in self.iterable:
                yield item
                now = time.perf_counter()
                duration = now - last
                last = now
                if duration <= 0:
                    continue
                progress.update(1)
                completed += 1
                self.durations.append(duration)
                self.rates.append(1.0 / duration)
                if self.total is None:
                    estimate = EtaEstimate(None, "unknown total")
                else:
                    remaining = max(self.total - completed, 0)
                    estimate = smart_eta(self.durations, remaining)
                direction = rate_direction(self.rates)
                short_direction = direction.split(",", maxsplit=1)[0]
                progress.set_postfix_str(
                    (
                        f"eta={format_seconds(estimate.seconds)} "
                        f"{estimate.label}, {short_direction}"
                    ),
                    refresh=False,
                )
                if (
                    now < next_graph_refresh
                    and (self.total is None or completed < self.total)
                ):
                    continue

                size = terminal_size()
                info_width = info_width_for_terminal(size.columns)
                graph_width = graph_width_for_terminal(
                    self.graph_width,
                    label_width,
                    info_width,
                    size.columns,
                )
                rendered = render_rate_graph(
                    self.rates,
                    width=graph_width,
                    height=panel_height,
                )
                visible_rates = self.rates[-rendered.visible_samples :]
                visible_avg = sum(visible_rates) / len(visible_rates)
                info_lines = smart_info_lines(
                    estimate,
                    direction=direction,
                    current_rate=self.rates[-1],
                    graph=rendered,
                    completed=completed,
                    total=self.total,
                    height=panel_height,
                )
                panel_lines = render_rate_panel(
                    rendered,
                    height=panel_height,
                    label_width=label_width,
                    info_width=info_width,
                    visible_avg=visible_avg,
                    info=info_lines,
                    use_color=use_color,
                )
                if panel_reserved:
                    draw_terminal_panel(output, panel_lines, size.columns)
                progress.refresh()
                next_graph_refresh = now + self.graph_refresh_interval
        finally:
            progress.close()


def tqdm(
    iterable: Iterable[T],
    *,
    desc: str = "",
    total: int | None = None,
    graph_width: int = 68,
    graph_height: int = 8,
    graph_refresh_interval: float = 0.12,
) -> TrendTqdm[T]:
    """Wrap an iterable with a trend-aware tqdm progress display."""
    return TrendTqdm(
        iterable,
        desc=desc,
        total=total,
        graph_width=graph_width,
        graph_height=graph_height,
        graph_refresh_interval=graph_refresh_interval,
    )


def trange(
    *args: int,
    desc: str = "",
    graph_width: int = 68,
    graph_height: int = 8,
    graph_refresh_interval: float = 0.12,
) -> TrendTqdm[int]:
    """Trend-aware equivalent of tqdm.trange."""
    return tqdm(
        range(*args),
        desc=desc,
        total=len(range(*args)),
        graph_width=graph_width,
        graph_height=graph_height,
        graph_refresh_interval=graph_refresh_interval,
    )
