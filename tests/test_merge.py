from nik import merge


def test_parse_timecode_ms() -> None:
    assert merge._parse_timecode("01:02:03.5") == 3723500


def test_metadata_tags_from_metadata() -> None:
    tags = merge._metadata_tags(
        {
            "title": "Book",
            "authors": ["Alice", "Bob"],
            "year": "2020",
            "language": "ja",
        }
    )
    assert tags["title"] == "Book"
    assert tags["album"] == "Book"
    assert tags["artist"] == "Alice, Bob"
    assert tags["album_artist"] == "Alice, Bob"
    assert tags["date"] == "2020"
    assert tags["language"] == "ja"


def test_plan_chapter_splits_balances_parts() -> None:
    chapters = [
        {"title": "a", "duration_ms": 1000, "segments": []},
        {"title": "b", "duration_ms": 2000, "segments": []},
        {"title": "c", "duration_ms": 3000, "segments": []},
        {"title": "d", "duration_ms": 4000, "segments": []},
    ]
    parts = merge._plan_chapter_splits(chapters, 2)
    assert len(parts) == 2
    assert [p["title"] for p in parts[0]] == ["a", "b"]
    assert [p["title"] for p in parts[1]] == ["c", "d"]


def test_auto_split_count() -> None:
    eight_hours_ms = int(8 * 3600 * 1000)
    assert merge._auto_split_count(eight_hours_ms, 8.0) == 1
    assert merge._auto_split_count(eight_hours_ms + 1, 8.0) == 2
