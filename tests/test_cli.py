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
    for name in ("ingest", "sanitize", "clone", "synth", "sample", "merge", "play"):
        assert name in choices
