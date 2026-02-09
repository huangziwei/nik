import argparse

from nik import cli


def test_cli_has_expected_commands() -> None:
    parser = cli.build_parser()
    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert subparsers
    choices = set(subparsers[0].choices.keys())
    for name in (
        "ingest",
        "sanitize",
        "rechunk",
        "clone",
        "synth",
        "sample",
        "merge",
        "boundary-report",
        "play",
    ):
        assert name in choices


def test_clone_parser_supports_auto_text() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["clone", "voice.wav", "--whisper-model", "tiny"])
    assert args.auto_text is True
    assert args.whisper_model == "tiny"


def test_clone_parser_allows_disabling_auto_text() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["clone", "voice.wav", "--no-auto-text"])
    assert args.auto_text is False


def test_synth_parser_supports_backend() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["synth", "--book", "out", "--voice", "voice.json", "--backend", "mlx"])
    assert args.backend == "mlx"


def _entry(
    delta_ms: float,
    output_latency_ms: float | None = None,
    *,
    trigger: str = "gapless",
    preloaded: bool = True,
    playback_rate: float = 1.0,
    pad_ms: int = 280,
) -> dict:
    payload = {
        "trigger": trigger,
        "preloaded": preloaded,
        "playback_rate": playback_rate,
        "delta_ms": delta_ms,
        "pad_ms": pad_ms,
    }
    if output_latency_ms is not None:
        payload["output_latency_ms"] = output_latency_ms
    return payload


def test_boundary_report_requires_output_latency_coverage() -> None:
    entries = [
        _entry(20.0),
        _entry(24.0),
        _entry(30.0),
        _entry(35.0),
        _entry(40.0),
        _entry(45.0),
    ]
    payload = cli._boundary_report_payload(entries, min_samples=5)
    recommendation = payload["recommendation"]

    assert payload["samples_filtered"] == 6
    assert payload["samples_with_output_latency"] == 0
    assert recommendation["recommended_pad_adjust_ms"] is None
    assert recommendation["basis"] == "insufficient_data"
    assert "coverage" in recommendation["reason"].lower()


def test_boundary_report_recommends_pad_adjustment_with_latency_data() -> None:
    entries = [
        _entry(10.0, output_latency_ms=20.0),
        _entry(12.0, output_latency_ms=20.0),
        _entry(14.0, output_latency_ms=20.0),
        _entry(16.0, output_latency_ms=20.0),
        _entry(18.0, output_latency_ms=20.0),
        _entry(999.0, output_latency_ms=20.0, trigger="ended"),
    ]
    payload = cli._boundary_report_payload(entries, min_samples=5)
    recommendation = payload["recommendation"]

    assert payload["samples_filtered"] == 5
    assert payload["samples_with_output_latency"] == 5
    assert payload["output_latency_coverage"] == 1.0
    assert recommendation["basis"] == "p50(delta_ms + output_latency_ms)"
    assert recommendation["recommended_pad_adjust_ms"] == 34
