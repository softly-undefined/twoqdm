from __future__ import annotations

import os
from io import StringIO

import pytest
import twoqdm.core as core
from twoqdm import tqdm as drop_in_tqdm
from tqdm import tqdm as base_tqdm

from twoqdm.core import (
    ASCII_SPINNERS,
    EtaEstimate,
    SPINNER_RAINBOW,
    TrendTqdm,
    ascii_spinner_lines,
    available_ascii_spinners,
    format_rate,
    format_seconds,
    rainbow_spinner_text,
    rate_direction,
    render_rate_graph,
    smart_eta,
    tqdm,
    trange,
)


def test_format_seconds() -> None:
    assert format_seconds(None) == "unknown"
    assert format_seconds(5.2) == "0:05"
    assert format_seconds(65) == "1:05"
    assert format_seconds(3661) == "1:01:01"


def test_format_rate_switches_slow_rates_to_seconds_per_iteration() -> None:
    assert format_rate(0.2) == "5 s/it"
    assert format_rate(2) == "2 it/s"


def test_render_rate_graph_returns_fixed_height_and_width() -> None:
    graph = render_rate_graph([1, 2, 3, 4, 5], width=12, height=4)

    assert len(graph.lines) == 4
    assert all(len(line) == 12 for line in graph.lines)
    assert graph.visible_samples == 5


def test_smart_eta_warms_up_before_enough_samples() -> None:
    estimate = smart_eta([0.1] * 7, remaining=10)

    assert estimate == EtaEstimate(None, "warming")


def test_smart_eta_done_when_no_remaining_items() -> None:
    estimate = smart_eta([0.1] * 10, remaining=0)

    assert estimate.seconds == 0
    assert estimate.label == "done"
    assert estimate.confidence == 1


def test_rate_direction_warms_up_before_enough_samples() -> None:
    assert rate_direction([1.0] * 9) == "warming 9/10"


def test_tqdm_supports_manual_mode_and_standard_kwargs() -> None:
    progress = tqdm(
        total=2,
        unit="item",
        unit_scale=True,
        leave=False,
        file=StringIO(),
        disable=False,
    )

    progress.update(1)
    progress.update(1)
    progress.close()

    assert progress.n == 2


def test_ascii_spinner_names_are_available() -> None:
    assert available_ascii_spinners() == (
        "coffee",
        "train",
        "conveyor",
        "sand-pile",
    )


def test_default_and_random_selectors_choose_per_progress_bar(monkeypatch) -> None:
    selections = iter(("coffee", "train", "sand-pile"))
    choice_options = []

    def choose(options):
        choice_options.append(tuple(options))
        return next(selections)

    monkeypatch.setattr(core.secrets, "choice", choose)
    default_progress = tqdm(
        total=1,
        file=StringIO(),
        disable=False,
    )
    random_progress = tqdm(
        total=1,
        ascii_spinner="random",
        file=StringIO(),
        disable=False,
    )
    true_progress = tqdm(
        total=1,
        ascii_spinner=True,
        file=StringIO(),
        disable=False,
    )

    assert default_progress.ascii_spinner == "coffee"
    assert random_progress.ascii_spinner == "train"
    assert true_progress.ascii_spinner == "sand-pile"
    assert choice_options == [available_ascii_spinners()] * 3
    default_progress.close()
    random_progress.close()
    true_progress.close()


@pytest.mark.parametrize("value", (False, None, "none", "off"))
def test_ascii_spinner_can_be_disabled(value) -> None:
    progress = tqdm(
        total=1,
        ascii_spinner=value,
        file=StringIO(),
        disable=False,
    )

    assert progress.ascii_spinner is None
    progress.close()


def test_tqdm_is_a_drop_in_for_standard_arguments_and_ascii_bar() -> None:
    common_kwargs = {
        "total": 3,
        "initial": 1,
        "desc": "compat",
        "unit": "row",
        "unit_scale": True,
        "unit_divisor": 1024,
        "leave": False,
        "ncols": 48,
        "mininterval": 0.0,
        "maxinterval": 1.0,
        "miniters": 1,
        "smoothing": 0.2,
        "ascii": True,
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}",
        "postfix": {"mode": "test"},
        "colour": "green",
        "delay": 0.0,
    }
    expected_output = StringIO()
    actual_output = StringIO()
    expected = base_tqdm(range(2), file=expected_output, **common_kwargs)
    expected_items = list(expected)
    actual = tqdm(range(2), file=actual_output, **common_kwargs)
    actual_items = list(actual)

    assert drop_in_tqdm is tqdm
    assert issubclass(tqdm, base_tqdm)
    assert actual_items == expected_items == [0, 1]
    for attribute in (
        "n",
        "total",
        "desc",
        "unit",
        "unit_scale",
        "unit_divisor",
        "ascii",
        "bar_format",
        "ncols",
    ):
        assert getattr(actual, attribute) == getattr(expected, attribute)
    assert actual_output.getvalue() == expected_output.getvalue()


def test_ascii_spinner_uses_rainbow_that_can_shift_phase() -> None:
    first = rainbow_spinner_text("line", row=0, phase=0, enabled=True)
    shifted = rainbow_spinner_text("line", row=0, phase=1, enabled=True)

    assert first == f"{SPINNER_RAINBOW[0]}line\x1b[0m"
    assert shifted == f"{SPINNER_RAINBOW[1]}line\x1b[0m"
    assert first != shifted


@pytest.mark.parametrize("spinner", available_ascii_spinners())
def test_tqdm_accepts_ascii_spinner_kwarg(spinner: str) -> None:
    progress = tqdm(
        total=1,
        ascii_spinner=spinner,
        file=StringIO(),
        disable=False,
    )

    progress.update(1)
    progress.close()

    assert progress.n == 1


@pytest.mark.parametrize("spinner", available_ascii_spinners())
def test_ascii_spinner_frames_are_terminal_safe_and_stable(spinner: str) -> None:
    rendered_frames = [
        ascii_spinner_lines(
            spinner,
            elapsed=float(frame_index),
            height=8,
            interval=1.0,
        )
        for frame_index in range(len(ASCII_SPINNERS[spinner]))
    ]

    assert all(len(frame) == 8 for frame in rendered_frames)
    assert all(line.isascii() for frame in rendered_frames for line in frame)
    assert len({len(line) for frame in rendered_frames for line in frame}) == 1
    assert max(len(line) for line in rendered_frames[0]) <= 13
    assert len({tuple(frame) for frame in rendered_frames}) > 1


def test_conveyor_moves_one_column_right_and_loops_smoothly() -> None:
    frames = ASCII_SPINNERS["conveyor"]

    assert frames[0][1].count("+-+") == 2
    for current, following in zip(frames, frames[1:] + frames[:1]):
        for row in (1, 2, 3, 4):
            assert following[row] == current[row][-1] + current[row][:-1]


def test_sand_pile_frames_follow_completion_progress() -> None:
    frames = ASCII_SPINNERS["sand-pile"]
    rendered_frames = [
        ascii_spinner_lines(
            "sand-pile",
            elapsed=999.0,
            height=8,
            interval=0.1,
            progress=index / (len(frames) - 1),
        )
        for index in range(len(frames))
    ]

    assert [tuple(frame) for frame in rendered_frames] == list(frames)
    sand_amounts = [sum(line.count("#") for line in frame) for frame in frames]
    assert sand_amounts == sorted(sand_amounts)
    assert len(set(sand_amounts)) == len(frames)


def test_tqdm_passes_completion_progress_to_sand_pile(monkeypatch) -> None:
    class TerminalBuffer(StringIO):
        def isatty(self) -> bool:
            return True

    captured_progress = []
    original = core.ascii_spinner_lines

    def capture_progress(*args, **kwargs):
        captured_progress.append(kwargs["progress"])
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "terminal_size", lambda: os.terminal_size((140, 24)))
    monkeypatch.setattr(core, "ascii_spinner_lines", capture_progress)
    with tqdm(
        total=4,
        ascii_spinner="sand-pile",
        file=TerminalBuffer(),
        disable=False,
        mininterval=0.0,
        graph_refresh_interval=0.0,
    ) as progress:
        progress.update(1)

    assert captured_progress[-1] == pytest.approx(0.25)


def test_tqdm_reflows_bar_and_panel_when_terminal_size_changes(monkeypatch) -> None:
    class TerminalBuffer(StringIO):
        def isatty(self) -> bool:
            return True

    current_size = {"value": os.terminal_size((140, 24))}
    panel_draws = []
    panel_resizes = []
    original_resize = core.resize_terminal_panel

    def capture_draw(file, lines, columns):
        panel_draws.append((columns, len(lines), max(map(core.visible_len, lines))))

    def capture_resize(
        file,
        panel_lines,
        bar_width,
        *,
        columns,
        new_height,
    ):
        panel_resizes.append((len(panel_lines), new_height, columns))
        original_resize(
            file,
            panel_lines,
            bar_width,
            columns=columns,
            new_height=new_height,
        )

    monkeypatch.setattr(core, "terminal_size", lambda: current_size["value"])
    monkeypatch.setattr(core, "draw_terminal_panel", capture_draw)
    monkeypatch.setattr(core, "resize_terminal_panel", capture_resize)

    with tqdm(
        total=5,
        file=TerminalBuffer(),
        disable=False,
        mininterval=0.0,
        graph_refresh_interval=999.0,
    ) as progress:
        assert callable(progress.dynamic_ncols)
        progress.update(1)

        current_size["value"] = os.terminal_size((100, 24))
        progress.update(1)

        current_size["value"] = os.terminal_size((80, 8))
        progress.update(1)

        current_size["value"] = os.terminal_size((70, 5))
        progress.update(1)

        current_size["value"] = os.terminal_size((120, 24))
        progress.update(1)

    assert panel_resizes == [
        (8, 8, 100),
        (8, 4, 80),
        (4, 0, 70),
        (0, 8, 120),
    ]
    assert [(columns, height) for columns, height, _ in panel_draws] == [
        (140, 8),
        (100, 8),
        (80, 4),
        (120, 8),
    ]
    assert all(width < columns for columns, _, width in panel_draws)


def test_resize_clears_every_physical_row_created_by_line_wrapping() -> None:
    output = StringIO()

    core.resize_terminal_panel(
        output,
        ["x" * 119] * 8,
        119,
        columns=60,
        new_height=4,
    )

    rendered = output.getvalue()
    assert rendered.count("\x1b[2K") == 18
    assert rendered.count("\x1b[1A") == 17
    assert rendered.endswith("\n" * 4)


def test_explicit_bar_width_disables_automatic_ncols() -> None:
    class TerminalBuffer(StringIO):
        def isatty(self) -> bool:
            return True

    progress = tqdm(
        total=1,
        file=TerminalBuffer(),
        disable=False,
        ncols=40,
    )

    assert progress.dynamic_ncols is False
    progress.close()


def test_tqdm_rejects_unknown_ascii_spinner() -> None:
    with pytest.raises(ValueError):
        tqdm(total=1, ascii_spinner="nope", file=StringIO(), disable=False)


def test_tqdm_exposes_write_and_can_be_subclassed() -> None:
    class ChildTqdm(tqdm):
        pass

    output = StringIO()
    tqdm.write("summary", file=output)
    progress = ChildTqdm(total=1, file=StringIO(), disable=False)
    progress.update(1)
    progress.close()

    assert output.getvalue() == "summary\n"
    assert isinstance(progress, TrendTqdm)


def test_tqdm_accepts_position_for_nested_bars() -> None:
    progress = tqdm(range(2), position=1, file=StringIO(), disable=False)

    assert list(progress) == [0, 1]


def test_trange_sets_total_from_range_args() -> None:
    progress = trange(2, 10, 2, file=StringIO(), disable=False)

    assert progress.total == 4
    progress.close()
