"""Trend-aware tqdm wrapper with a terminal rate graph and smarter ETA."""

from __future__ import annotations

import math
import os
import re
import secrets
import shutil
import sys
import time
from dataclasses import dataclass
from typing import TypeVar

try:
    from tqdm import tqdm as base_tqdm
except ImportError as exc:  # pragma: no cover - useful when copied into projects
    raise SystemExit(
        "tqdm is not installed. Install it with: python3 -m pip install tqdm"
    ) from exc


T = TypeVar("T")
_ACTIVE_PANEL_DEPTH = 0

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
SPINNER_RAINBOW = (
    RED,
    YELLOW,
    GREEN,
    CYAN,
    BLUE,
    "\x1b[38;5;177m",
    "\x1b[38;5;213m",
)
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
# Compact terminal loops are embedded to keep twoqdm's only runtime dependency
# as tqdm.
ASCII_SPINNERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "coffee": (
        (
            "   (   )  ",
            "    ) (   ",
            "   (   )  ",
            "  .----.  ",
            "  |~~~~|] ",
            "  |____|  ",
            "   `--'   ",
            "          ",
        ),
        (
            "    ) (   ",
            "   (   )  ",
            "    ) (   ",
            "  .----.  ",
            "  |~~~~|] ",
            "  |____|  ",
            "   `--'   ",
            "          ",
        ),
        (
            "   ) (    ",
            "    (     ",
            "   ) (    ",
            "  .----.  ",
            "  |-~~-|] ",
            "  |____|  ",
            "   `--'   ",
            "          ",
        ),
        (
            "    (     ",
            "   ) (    ",
            "    (     ",
            "  .----.  ",
            "  |~~~~|] ",
            "  |____|  ",
            "   `--'   ",
            "          ",
        ),
    ),
    "train": (
        (
            "       ( )  ",
            "      (@)   ",
            "     __||_  ",
            " __|_[]_|__ ",
            "|  _     _| ",
            "'-(o)---(o)'",
            "=_========_=",
            "            ",
        ),
        (
            "     ( )    ",
            "       (@)  ",
            "     __||_  ",
            " __|_[]_|__ ",
            "|  _     _| ",
            "'-(O)---(O)'",
            "==_========_",
            "            ",
        ),
        (
            "    ( )     ",
            "     (@)    ",
            "     __||_  ",
            " __|_[]_|__ ",
            "|  _     _| ",
            "'-(o)---(o)'",
            "===_========",
            "            ",
        ),
        (
            "      ( )   ",
            "    (@)     ",
            "     __||_  ",
            " __|_[]_|__ ",
            "|  _     _| ",
            "'-(O)---(O)'",
            "_========_==",
            "            ",
        ),
    ),
    "conveyor": (
        (
            "            ",
            "+-+   +-+   ",
            "|#|   |#|   ",
            "+-+===+-+===",
            "o o o o o o ",
            "============",
            "            ",
            "            ",
        ),
        (
            "            ",
            " +-+   +-+  ",
            " |#|   |#|  ",
            "=+-+===+-+==",
            " o o o o o o",
            "============",
            "            ",
            "            ",
        ),
        (
            "            ",
            "  +-+   +-+ ",
            "  |#|   |#| ",
            "==+-+===+-+=",
            "o o o o o o ",
            "============",
            "            ",
            "            ",
        ),
        (
            "            ",
            "   +-+   +-+",
            "   |#|   |#|",
            "===+-+===+-+",
            " o o o o o o",
            "============",
            "            ",
            "            ",
        ),
        (
            "            ",
            "+   +-+   +-",
            "|   |#|   |#",
            "+===+-+===+-",
            "o o o o o o ",
            "============",
            "            ",
            "            ",
        ),
        (
            "            ",
            "-+   +-+   +",
            "#|   |#|   |",
            "-+===+-+===+",
            " o o o o o o",
            "============",
            "            ",
            "            ",
        ),
    ),
    "sand-pile": (
        (
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "____________",
        ),
        (
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "____/##\\____",
        ),
        (
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     #      ",
            "____/##\\____",
        ),
        (
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "    /##\\    ",
            "___/####\\___",
        ),
        (
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     #      ",
            "    /##\\    ",
            "___/####\\___",
        ),
        (
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "    /##\\    ",
            "   /####\\   ",
            "__/######\\__",
        ),
        (
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "     #      ",
            "    /##\\    ",
            "   /####\\   ",
            "__/######\\__",
        ),
        (
            "     .      ",
            "     .      ",
            "     .      ",
            "     .      ",
            "    /##\\    ",
            "   /####\\   ",
            "  /######\\  ",
            "_/########\\_",
        ),
        (
            "     .      ",
            "     .      ",
            "     .      ",
            "     #      ",
            "    /##\\    ",
            "   /####\\   ",
            "  /######\\  ",
            "_/########\\_",
        ),
        (
            "     .      ",
            "     .      ",
            "     #      ",
            "    /##\\    ",
            "   /####\\   ",
            "  /######\\  ",
            " /########\\ ",
            "/##########\\",
        ),
    ),
}


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


def compact_number(value: float) -> str:
    if value >= 1000:
        text = f"{value:.0f}"
    elif value >= 100:
        text = f"{value:.1f}"
    elif value >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_rate(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        return "unknown"
    if value < 1:
        return f"{compact_number(1.0 / value)} s/it"
    return f"{compact_number(value)} it/s"


def compact_count(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric)}"
    if numeric >= 1000:
        return f"{numeric:.0f}"
    if numeric >= 100:
        return f"{numeric:.1f}"
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


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


def file_isatty(file) -> bool:
    isatty = getattr(file, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


def color_enabled(file) -> bool:
    return (
        file_isatty(file)
        and os.environ.get("TQDM_TREND_NO_COLOR") is None
        and os.environ.get("TERM") != "dumb"
    )


def colored(text: str, color: str, *, enabled: bool) -> str:
    if not enabled or not text:
        return text
    return f"{color}{text}{RESET}"


def available_ascii_spinners() -> tuple[str, ...]:
    return tuple(ASCII_SPINNERS)


def normalize_ascii_spinner(value: str | bool | None) -> str | None:
    if value is None or value is False:
        return None
    if value is True:
        return secrets.choice(available_ascii_spinners())

    key = str(value).strip().lower().replace("_", "-")
    if key in {"", "0", "false", "no", "none", "off"}:
        return None
    if key == "random":
        return secrets.choice(available_ascii_spinners())

    if key not in ASCII_SPINNERS:
        choices = ", ".join(("random", *available_ascii_spinners()))
        raise ValueError(f"unknown ascii_spinner {value!r}; choose one of: {choices}")
    return key


def ascii_spinner_lines(
    name: str | None,
    *,
    elapsed: float,
    height: int,
    interval: float,
    progress: float | None = None,
) -> list[str]:
    if not name or height <= 0:
        return []

    frames = ASCII_SPINNERS[name]
    if name == "sand-pile" and progress is not None and math.isfinite(progress):
        fraction = min(max(progress, 0.0), 1.0)
        frame_index = int(fraction * (len(frames) - 1) + 0.5)
    else:
        frame_interval = max(interval, 1e-9)
        frame_index = int(elapsed / frame_interval) % len(frames)
    frame = frames[frame_index]
    width = max(visible_len(line) for candidate in frames for line in candidate)
    return [f"{line:<{width}}" for line in (list(frame) + [""] * height)[:height]]


def info_sidecar_widths(
    width: int,
    sidecar: list[str],
    *,
    min_info_width: int = 14,
) -> tuple[int, int] | None:
    if not sidecar:
        return None

    sidecar_width = max((visible_len(line) for line in sidecar), default=0)
    if sidecar_width <= 0:
        return None

    gap = 1
    info_width = width - sidecar_width - gap
    if info_width < min_info_width:
        return None

    return info_width, sidecar_width


def rainbow_spinner_text(
    text: str,
    *,
    row: int,
    phase: int,
    enabled: bool,
) -> str:
    if not enabled or not text:
        return text

    color = SPINNER_RAINBOW[phase % len(SPINNER_RAINBOW)]
    return colored(text, color, enabled=True)


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
    sidecar: list[str] | None = None,
    sidecar_color_phase: int = 0,
) -> list[str]:
    labels = [""] * height
    if height:
        labels[0] = f"fast {format_rate(graph.scale_high)}"
        labels[height // 2] = f"avg {format_rate(visible_avg)}"
        labels[-1] = f"slow {format_rate(graph.scale_low)}"

    sidecar = sidecar or []
    sidecar_widths = info_sidecar_widths(info_width, sidecar)
    text_width = sidecar_widths[0] if sidecar_widths else info_width
    spinner_width = sidecar_widths[1] if sidecar_widths else 0

    panel = []
    for row, (label, line) in enumerate(zip(labels, graph.lines)):
        label = fitted_graph_label(label, label_width)
        label_text = f"{label:<{label_width}}"
        label_text = colored(label_text, label_color(label), enabled=use_color)
        rail = colored("|", GRAY, enabled=use_color)
        graph_line = color_graph_line(line, graph=graph, enabled=use_color)
        right = info[row] if row < len(info) else ""
        right = fitted_terminal_line(right, text_width)
        right_text = f"{right:<{text_width}}"
        right_text = colored(right_text, info_color(right), enabled=use_color)
        if sidecar_widths:
            spinner = sidecar[row] if row < len(sidecar) else ""
            spinner = fitted_terminal_line(spinner, spinner_width)
            spinner_text = f"{spinner:<{spinner_width}}"
            spinner_text = rainbow_spinner_text(
                spinner_text,
                row=row,
                phase=sidecar_color_phase,
                enabled=use_color,
            )
            right_text = f"{right_text} {spinner_text}"
        panel.append(f"{label_text}{rail}{graph_line}{rail} {right_text}")
    return panel


def smart_info_lines(
    estimate: EtaEstimate,
    *,
    direction: str,
    current_rate: float,
    graph: RateGraph,
    completed: float,
    total: float | int | None,
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
        f"now {format_rate(current_rate)}",
        f"window {graph.visible_samples}{clipped}",
        (
            f"{compact_count(completed)}/{compact_count(total)}"
            if total is not None
            else compact_count(completed)
        ),
    ]
    return (lines + [""] * height)[:height]


def reserve_terminal_panel(file, height: int) -> bool:
    if height <= 0 or not file_isatty(file):
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


def clear_terminal_panel(file, height: int) -> None:
    if height <= 0:
        return

    file.write("\x1b7")
    file.write(f"\x1b[{height}A")
    for row in range(height):
        file.write("\r\x1b[2K")
        if row < height - 1:
            file.write("\n")
    file.write("\x1b8")
    file.flush()


def resize_terminal_panel(
    file,
    panel_lines: list[str],
    bar_width: int,
    *,
    columns: int,
    new_height: int,
) -> None:
    """Clear wrapped display rows and re-anchor the resized panel."""
    columns = max(columns, 1)
    line_widths = [visible_len(line) for line in panel_lines]
    physical_rows = sum(max(1, math.ceil(width / columns)) for width in line_widths)
    physical_rows += max(1, math.ceil(max(bar_width, 1) / columns))

    file.write("\r")
    for row in range(physical_rows):
        file.write("\x1b[2K")
        if row < physical_rows - 1:
            file.write("\x1b[1A")
    file.write("\r")
    file.write("\n" * new_height)
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


def smart_eta(durations: list[float], remaining: float) -> EtaEstimate:
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

    try:
        first = math.exp(intercept + slope * len(recent))
        ratio = math.exp(slope)
        if math.isclose(ratio, 1.0):
            projected = first * remaining
        else:
            projected = first * (math.pow(ratio, remaining) - 1.0) / (ratio - 1.0)
    except OverflowError:
        projected = float("inf")

    capped = projected > avg_eta * 50 and projected > 60
    if capped:
        projected = max(avg_eta * 50, 60)

    if slope > 0.03:
        label = "exp slowing"
    elif slope > 0:
        label = "slowing"
    else:
        label = "speeding up"
    if capped:
        label += ", unstable"
    return EtaEstimate(projected, label, confidence=r_squared, capped=capped)


class TrendTqdm(base_tqdm):
    """A tqdm-compatible progress bar with an optional rate trend panel."""

    def __init__(
        self,
        *args,
        graph_width: int = 68,
        graph_height: int = 8,
        graph_refresh_interval: float = 0.12,
        ascii_spinner: str | bool | None = True,
        ascii_spinner_interval: float = 0.10,
        **kwargs,
    ) -> None:
        self.disable = True
        self._twoqdm_panel_height = 0
        self._twoqdm_panel_reserved = False
        self._twoqdm_panel_owner = False
        self._twoqdm_resize_ready = False
        self._twoqdm_last_panel_lines: list[str] = []
        self._twoqdm_last_bar_width = 0
        self.graph_width = graph_width
        self.graph_height = graph_height
        self.graph_refresh_interval = graph_refresh_interval
        self.ascii_spinner = normalize_ascii_spinner(ascii_spinner)
        self.ascii_spinner_interval = ascii_spinner_interval
        self.durations: list[float] = []
        self.rates: list[float] = []
        self._twoqdm_output = kwargs.get("file") or sys.stderr
        self._twoqdm_last_sample = time.perf_counter()
        self._twoqdm_spinner_started = self._twoqdm_last_sample
        self._twoqdm_next_graph_refresh = 0.0
        self._twoqdm_terminal_size = terminal_size()
        self._twoqdm_panel_height = self._panel_height(self._twoqdm_terminal_size)
        self._twoqdm_manage_postfix = "postfix" not in kwargs
        self._twoqdm_setting_auto_postfix = False

        use_color = color_enabled(self._twoqdm_output)
        kwargs.setdefault("colour", "cyan" if use_color else None)
        if "dynamic_ncols" not in kwargs:
            kwargs["dynamic_ncols"] = (
                "ncols" not in kwargs and file_isatty(self._twoqdm_output)
            )
        self._reserve_panel_if_available(kwargs)
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            self._release_panel()
            raise
        self._twoqdm_last_panel_lines = [""] * self._twoqdm_panel_height
        self._twoqdm_last_bar_width = visible_len(str(self))
        self._twoqdm_resize_ready = True

    def _panel_height(self, size: os.terminal_size | None = None) -> int:
        size = size or terminal_size()
        available = size.lines - 4
        if self.graph_height <= 0 or available < 3:
            return 0
        return min(self.graph_height, available)

    def _reserve_panel_if_available(self, kwargs) -> None:
        global _ACTIVE_PANEL_DEPTH

        position = kwargs.get("position")
        if (
            kwargs.get("disable") is True
            or self._twoqdm_panel_height <= 0
            or position not in (None, 0)
            or _ACTIVE_PANEL_DEPTH > 0
        ):
            return

        if reserve_terminal_panel(self._twoqdm_output, self._twoqdm_panel_height):
            self._twoqdm_panel_reserved = True
            self._twoqdm_panel_owner = True
            _ACTIVE_PANEL_DEPTH += 1

    def _release_panel(self) -> None:
        global _ACTIVE_PANEL_DEPTH

        if self._twoqdm_panel_owner:
            _ACTIVE_PANEL_DEPTH = max(0, _ACTIVE_PANEL_DEPTH - 1)
            self._twoqdm_panel_owner = False

    def _resize_panel_if_needed(self, size: os.terminal_size) -> bool:
        if size == self._twoqdm_terminal_size:
            return False

        new_height = self._panel_height(size)
        resize_terminal_panel(
            self._twoqdm_output,
            self._twoqdm_last_panel_lines,
            self._twoqdm_last_bar_width,
            columns=size.columns,
            new_height=new_height,
        )
        self.sp = self.status_printer(self.fp)
        self._twoqdm_terminal_size = size
        self._twoqdm_panel_height = new_height
        self._twoqdm_last_panel_lines = [""] * new_height
        self._twoqdm_last_bar_width = 0
        self._twoqdm_next_graph_refresh = 0.0
        return True

    def _refresh_progress_line(self) -> None:
        self._twoqdm_output.write("\r\x1b[2K")
        self._twoqdm_output.flush()
        # tqdm otherwise pads back to its cached pre-resize line length.
        self.sp = self.status_printer(self.fp)
        super().refresh()
        self._twoqdm_last_bar_width = visible_len(str(self))

    def __iter__(self):
        if self.disable:
            for obj in self.iterable:
                yield obj
            return

        try:
            for obj in self.iterable:
                yield obj
                self.update(1)
        finally:
            self.close()

    def update(self, n=1):
        if self._twoqdm_resize_ready and self._twoqdm_panel_reserved:
            self._resize_panel_if_needed(terminal_size())
        result = super().update(n)
        self._record_trend_sample(n)
        return result

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        if not getattr(self, "_twoqdm_setting_auto_postfix", False):
            self._twoqdm_manage_postfix = False
        return super().set_postfix(ordered_dict, refresh=refresh, **kwargs)

    def set_postfix_str(self, s="", refresh=True):
        if not getattr(self, "_twoqdm_setting_auto_postfix", False):
            self._twoqdm_manage_postfix = False
        return super().set_postfix_str(s, refresh=refresh)

    def _record_trend_sample(self, n) -> None:
        if getattr(self, "disable", False):
            return

        now = time.perf_counter()
        duration = now - self._twoqdm_last_sample
        self._twoqdm_last_sample = now
        try:
            increment = float(n)
        except (TypeError, ValueError):
            return
        if increment <= 0 or duration <= 0:
            return

        self.durations.append(duration / increment)
        self.rates.append(increment / duration)
        estimate = self._eta_estimate()
        direction = rate_direction(self.rates)
        self._set_auto_postfix(estimate, direction)
        self._render_trend_panel(now, estimate, direction)

    def _eta_estimate(self) -> EtaEstimate:
        if self.total is None:
            return EtaEstimate(None, "unknown total")

        try:
            remaining = max(float(self.total) - float(self.n), 0.0)
        except (TypeError, ValueError):
            return EtaEstimate(None, "unknown total")
        return smart_eta(self.durations, remaining)

    def _set_auto_postfix(self, estimate: EtaEstimate, direction: str) -> None:
        if not self._twoqdm_manage_postfix:
            return

        short_direction = direction.split(",", maxsplit=1)[0]
        self._twoqdm_setting_auto_postfix = True
        try:
            super().set_postfix_str(
                (
                    f"eta={format_seconds(estimate.seconds)} "
                    f"{estimate.label}, {short_direction}"
                ),
                refresh=False,
            )
        finally:
            self._twoqdm_setting_auto_postfix = False

    def _render_trend_panel(
        self,
        now: float,
        estimate: EtaEstimate,
        direction: str,
    ) -> None:
        if not self._twoqdm_panel_reserved:
            return
        size = terminal_size()
        resized = self._resize_panel_if_needed(size)
        if not resized and now < self._twoqdm_next_graph_refresh and (
            self.total is None or self.n < self.total
        ):
            return
        if self._twoqdm_panel_height <= 0:
            self._refresh_progress_line()
            self._twoqdm_next_graph_refresh = now + self.graph_refresh_interval
            return

        label_width = 16
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
            height=self._twoqdm_panel_height,
        )
        visible_rates = self.rates[-rendered.visible_samples :]
        visible_avg = sum(visible_rates) / len(visible_rates)
        info_lines = smart_info_lines(
            estimate,
            direction=direction,
            current_rate=self.rates[-1],
            graph=rendered,
            completed=float(self.n),
            total=self.total,
            height=self._twoqdm_panel_height,
        )
        spinner_progress = None
        if self.total is not None:
            try:
                completed = float(self.n)
                total = float(self.total)
                if math.isfinite(completed) and math.isfinite(total):
                    if total > 0:
                        spinner_progress = completed / total
                    elif total == 0:
                        spinner_progress = 1.0
            except (TypeError, ValueError):
                pass
        spinner_lines = ascii_spinner_lines(
            self.ascii_spinner,
            elapsed=now - self._twoqdm_spinner_started,
            height=self._twoqdm_panel_height,
            interval=self.ascii_spinner_interval,
            progress=spinner_progress,
        )
        spinner_color_phase = int((now - self._twoqdm_spinner_started) / 1.20)
        panel_lines = render_rate_panel(
            rendered,
            height=self._twoqdm_panel_height,
            label_width=label_width,
            info_width=info_width,
            visible_avg=visible_avg,
            info=info_lines,
            use_color=color_enabled(self._twoqdm_output),
            sidecar=spinner_lines,
            sidecar_color_phase=spinner_color_phase,
        )
        drawn_panel_lines = [
            fitted_terminal_line(line, size.columns - 1) for line in panel_lines
        ]
        draw_terminal_panel(
            self._twoqdm_output,
            drawn_panel_lines,
            size.columns,
        )
        self._twoqdm_last_panel_lines = drawn_panel_lines
        self._refresh_progress_line()
        self._twoqdm_next_graph_refresh = now + self.graph_refresh_interval

    def close(self) -> None:
        if getattr(self, "_twoqdm_panel_reserved", False) and not getattr(
            self, "leave", True
        ):
            clear_terminal_panel(self._twoqdm_output, self._twoqdm_panel_height)
        try:
            super().close()
        finally:
            self._release_panel()


tqdm = TrendTqdm


def trange(
    *args: int,
    **kwargs,
) -> TrendTqdm[int]:
    """Trend-aware equivalent of tqdm.trange."""
    iterable = range(*args)
    kwargs.setdefault("total", len(iterable))
    return TrendTqdm(iterable, **kwargs)
