from __future__ import annotations

from twoqdm.core import (
    EtaEstimate,
    format_seconds,
    rate_direction,
    render_rate_graph,
    smart_eta,
    trange,
)


def test_format_seconds() -> None:
    assert format_seconds(None) == "unknown"
    assert format_seconds(5.2) == "0:05"
    assert format_seconds(65) == "1:05"
    assert format_seconds(3661) == "1:01:01"


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


def test_trange_sets_total_from_range_args() -> None:
    progress = trange(2, 10, 2)

    assert progress.total == 4
