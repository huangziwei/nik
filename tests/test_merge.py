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
