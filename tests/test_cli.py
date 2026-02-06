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
