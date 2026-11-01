from pathlib import Path

import pytest

from nik import latin
from nik import tts as tts_util

_UNIDIC_AVAILABLE = (tts_util._default_unidic_dir() / "lex.csv").exists()
requires_unidic = pytest.mark.skipif(
    not _UNIDIC_AVAILABLE, reason="UniDic lex.csv is needed for the loanword lexicon"
)


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch):
    lexicon = latin._Lexicon(
        exact={
            "New York": [["ニューヨーク", 2201]],
            "Netflix": [["ネットフリックス", 4805], ["ネトフリ", 4805]],
        },
        folded={
            "line": [["ライン", 2750]],
            "the": [["ザ", 3320]],
            "win": [["ウィン", 3000]],
            "tv": [["テレビ", 808]],
            "book": [["ブック", 1000]],
            "story": [["ストーリー", 1000]],
            "of": [["オブ", 3000]],
        },
    )
    unpronounceable = {"sns", "tv", "adhd", "of"}
    monkeypatch.setattr(latin, "_lexicon", lambda: lexicon)
    monkeypatch.setattr(latin, "_c2k", lambda word: f"<{word.lower()}>")
    monkeypatch.setattr(
        latin, "_pronounceable", lambda word: word.lower() not in unpronounceable
    )
    latin.word_reading.cache_clear()
    yield
    latin.word_reading.cache_clear()


@pytest.mark.usefixtures("fake_models")
@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("T", "ティー"),
        ("v", "ブイ"),
        ("SNS", "エスエヌエス"),  # unknown short token → spelled
        ("sns", "エスエヌエス"),
        ("TV", "ティーブイ"),  # two capitals are spelled even when a word exists
        ("ON", "オン"),  # …except the curated two-letter words
        ("LINE", "ライン"),  # caps word found case-folded
        ("WIN", "ウィン"),  # short caps word that reads like one
        ("of", "オブ"),  # lowercase dictionary words skip the acronym gate
        ("ADHD", "エーディーエイチディー"),  # long but unpronounceable
        ("books", "ブックス"),  # plural of a dictionary word
        ("stories", "ストーリーズ"),
        ("Wireshark", "<wireshark>"),  # unknown → character model
        ("Mr", "ミスター"),
        ("email", "イーメール"),  # curated over the dictionary's French émail
    ],
)
def test_word_reading_resolution_order(word: str, expected: str) -> None:
    assert latin.word_reading(word) == expected


@pytest.mark.usefixtures("fake_models")
def test_cost_tie_goes_to_reading_closest_to_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(latin, "_c2k", lambda word: "ネットフリックス")
    assert latin.word_reading("Netflix") == "ネットフリックス"


@pytest.mark.usefixtures("fake_models")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("New Yorkへ行く", "ニューヨークへ行く"),  # phrase entry beats per-word
        ("『I love the book』", "『アイ・<love>・ザ・ブック』"),
        ("以降は LINE、と", "以降はライン、と"),  # space after hiragana dropped
        ("Netflix 上で", "ネットフリックス上で"),  # space before kanji dropped
        ("YouTube やSNSに", "<youtube>やエスエヌエスに"),  # before a particle
        ("間奏 Interlude", "間奏 <interlude>"),  # gloss after a kanji term keeps it
        ("Mr. Wireshark", "ミスター・<wireshark>"),
        ("高知東生 @Wireshark", "高知東生 <wireshark>"),
        ("新聞 https://www.example.com/a?b=1 より", "新聞より"),
        ("連絡は foo.bar@example.co.jp まで", "連絡はまで"),
        ("ＬＩＮＥした", "ラインした"),  # fullwidth Latin
        ("GOSICK VIII", "<gosick> VIII"),  # roman numerals are left to numbers
    ],
)
def test_latin_to_katakana(text: str, expected: str) -> None:
    assert latin.latin_to_katakana(text) == expected


@pytest.mark.usefixtures("fake_models")
def test_latin_reading_spans_index_original_text() -> None:
    text = "以降は　ＬＩＮＥ、Netflixで https://x.jp"
    spans = latin.latin_reading_spans(text)
    assert [(text[s["start"] : s["end"]], s["reading"]) for s in spans] == [
        ("ＬＩＮＥ", "ライン"),
        ("Netflix", "ネットフリックス"),
        ("https://x.jp", ""),
    ]


def test_build_lexicon_drops_clippings_and_ranks_case(tmp_path: Path) -> None:
    rows = [
        # surface,left,right,cost,…,lForm(10),lemma(11)
        ("ビル", 826, "ビル-building"),
        ("ビルディング", 3047, "ビルディング-building"),
        ("パリ", 4746, "パリ-Paris"),
        ("パリス", 6638, "パリス-Paris"),
        ("ナイス", 3000, "ナイス-nice"),
        ("ニース", 2000, "ニース-Nice"),
        ("ο", 5340, "ο-omicron"),  # not katakana: skipped
        ("スケジュール", 2779, "スケジュール-schedule"),
    ]
    lex_csv = tmp_path / "lex.csv"
    lex_csv.write_text(
        "\n".join(
            f"{surface},1,1,{cost},名詞,普通名詞,一般,*,*,*,{surface},{lemma}"
            for surface, cost, lemma in rows
        ),
        encoding="utf-8",
    )
    lexicon = latin._build_lexicon(lex_csv)
    assert lexicon.exact["building"] == [["ビルディング", 3047]]
    assert [kana for kana, _ in lexicon.exact["Paris"]] == ["パリ", "パリス"]
    assert lexicon.folded["nice"][0][0] == "ナイス"  # common word over the city
    assert lexicon.exact["Nice"][0][0] == "ニース"
    assert "omicron" not in lexicon.exact
    assert lexicon.folded["schedule"] == [["スケジュール", 2779]]


@requires_unidic
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Real lines from the library: 平場の月, 「自傷的自己愛」の精神分析, 家事代行…３.
        (
            "「こないだハコにＬＩＮＥしたんだって。久しぶり、元気？　みたいな」",
            "こないだハコにラインしたんだって。久しぶり、元気? みたいな",
        ),
        (
            "二〇〇〇年代以降は LINE、Facebook、Twitter、Instagram などのSNSが",
            (
                "二千年代以降はライン、フェイスブック、ツイッター、インスタグラムなどの"
                "エスエヌエスが"
            ),
        ),
        (
            "知性に自信がない人々は、YouTube やSNSに流れてくる",
            "知性に自信がない人々は、ユーチューブやエスエヌエスに流れてくる",
        ),
        ("このTシャツ、彼氏が着てたら", "このティーシャツ、彼氏が着てたら"),
        (
            "アメリカ精神医学会（APA）の編纂した診断基準である「DSM‐5」",
            "アメリカ精神医学会(エーピーエー)の編纂した診断基準であるディーエスエム‐ご",
        ),
        ("WIN‐WINの関係", "ウィン・ウィンの関係"),
        ("そのNO・208とね", "そのナンバー二百八とね"),
        ("buildingとanimation", "ビルディングとアニメーション"),
        ("Parisのbarで", "パリのバーで"),
    ],
)
def test_prepare_tts_pipeline_reads_latin_in_katakana(text: str, expected: str) -> None:
    latin.word_reading.cache_clear()
    pipeline = tts_util._prepare_tts_pipeline(text, add_short_punct=False)
    assert pipeline.prepared == expected


@requires_unidic
def test_prepare_tts_pipeline_user_override_beats_latin_reading() -> None:
    latin.word_reading.cache_clear()
    pipeline = tts_util._prepare_tts_pipeline(
        "Boogiepopは笑わない",
        merged_overrides=[{"base": "Boogiepop", "reading": "ブギーポップ"}],
        add_short_punct=False,
    )
    assert pipeline.prepared == "ブギーポップは笑わない"
