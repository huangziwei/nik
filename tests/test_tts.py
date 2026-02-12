import json
from pathlib import Path

import pytest

from nik import tts as tts_util
from nik.text import SECTION_BREAK
from nik import voice as voice_util


@pytest.fixture(autouse=True)
def _reset_first_token_separator_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(tts_util.FIRST_TOKEN_SEPARATOR_ENV, raising=False)


def _default_first_token_separator() -> str:
    return tts_util._first_token_separator()


def test_make_chunk_spans_splits_japanese_sentences() -> None:
    text = "今日は良い天気です。明日も晴れるでしょう。"
    spans = tts_util.make_chunk_spans(text, max_chars=10, chunk_mode="japanese")
    assert len(spans) >= 2


def test_chunking_splits_on_commas_and_linebreaks() -> None:
    text = "私は、猫です。\n次の行です。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["私は、猫です。", "次の行です。"]


def test_chunking_splits_on_section_break() -> None:
    text = f"一。\n\n{SECTION_BREAK}\n\n二。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["一。", "二。"]


def test_chunking_splits_on_fullwidth_digits() -> None:
    text = "１\n\n本文です。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["１", "本文です。"]


def test_chunking_splits_on_space_with_digits() -> None:
    text = "さくら荘のペットな彼女　全13巻"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["さくら荘のペットな彼女", "全13巻"]


def test_chunking_packs_sentences_toward_max_chars() -> None:
    text = "一。二。三。四。"
    spans = tts_util.make_chunk_spans(text, max_chars=4, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["一。二。", "三。四。"]


def test_chunking_keeps_balanced_quote_as_own_chunk() -> None:
    text = "彼は「いいえ。ありがとうございます」と言った。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["彼は", "「いいえ。ありがとうございます」", "と言った。"]


def test_chunking_splits_long_quote_when_over_max_chars() -> None:
    text = "「あいうえお。かきくけこ。さしすせそ。」"
    spans = tts_util.make_chunk_spans(text, max_chars=10, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["「あいうえお。", "かきくけこ。", "さしすせそ。」"]


def test_normalize_numbers_hatachi() -> None:
    assert tts_util._normalize_numbers("20歳を過ぎた") == "はたちを過ぎた"
    assert tts_util._normalize_numbers("二十歳") == "はたち"


def test_chunking_splits_on_commas_when_too_long() -> None:
    text = "これはとても長い文章なので、途中で区切る必要があります。"
    spans = tts_util.make_chunk_spans(text, max_chars=15, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert any("、" in chunk for chunk in chunks)


def test_chunking_keeps_leading_ellipsis_with_sentence() -> None:
    text = "……なんで八奈見がここにいるんだ。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["……なんで八奈見がここにいるんだ。"]


def test_chunking_splits_on_punct_only_line() -> None:
    text = "「……」\n\n「空太？」"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["「……」", "「空太？」"]


def test_prepare_tts_text_preserves_unicode_ellipsis() -> None:
    assert tts_util.prepare_tts_text("…………") == "…………"


def test_prepare_tts_pipeline_appends_chunk_tail_separator() -> None:
    sep = _default_first_token_separator()
    pipeline = tts_util._prepare_tts_pipeline(
        "こんにちは",
        kana_normalize=False,
        add_short_punct=False,
    )
    assert pipeline.prepared == f"こんにちは{sep}"


def test_prepare_tts_pipeline_does_not_duplicate_chunk_tail_separator() -> None:
    sep = _default_first_token_separator()
    pipeline = tts_util._prepare_tts_pipeline(
        f"こんにちは{sep}",
        kana_normalize=False,
        add_short_punct=False,
    )
    assert pipeline.prepared == f"こんにちは{sep}"


def test_prepare_tts_pipeline_skips_chunk_tail_separator_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(tts_util.FIRST_TOKEN_SEPARATOR_ENV, "none")
    pipeline = tts_util._prepare_tts_pipeline(
        "こんにちは",
        kana_normalize=False,
        add_short_punct=False,
    )
    assert pipeline.prepared == "こんにちは"


def test_prepare_tts_pipeline_strips_leading_chunk_separator() -> None:
    sep = _default_first_token_separator()
    pipeline = tts_util._prepare_tts_pipeline(
        f"『{sep}ばんがいへん{sep}』",
        kana_normalize=False,
        add_short_punct=True,
    )
    assert pipeline.prepared == f"ばんがいへん{sep}。{sep}"


def test_compute_chunk_pause_multipliers_detects_title_and_section_breaks() -> None:
    text = f"序章\n\n本文一。\n\n{SECTION_BREAK}\n\n本文二。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["序章", "本文一。", "本文二。"]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert len(multipliers) == len(spans)
    assert multipliers[0] >= 5
    assert multipliers[1] >= 3
    assert multipliers[2] == 1


@pytest.mark.parametrize(
    "dialogue",
    [
        "「静かすぎる」",
        "「いい出してくれて、よかったわ」",
    ],
)
def test_compute_chunk_pause_multipliers_does_not_promote_dialogue_lines(
    dialogue: str,
) -> None:
    text = f"前の段落。\n\n{dialogue}\n\n次の段落。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["前の段落。", dialogue, "次の段落。"]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers == [1, 1, 1]


def test_compute_chunk_pause_multipliers_does_not_promote_split_dialogue_fragments() -> None:
    text = (
        "前の段落。\n\n"
        "「いいえ。もしお嫌じゃなかったら、一度どうぞ、のみにいらして下さい」\n\n"
        "「ありがとうございます。はずかしいわ」\n\n"
        "次の段落。"
    )
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)

    assert "「いいえ。もしお嫌じゃなかったら、一度どうぞ、のみにいらして下さい」" in chunks
    assert "「ありがとうございます。はずかしいわ」" in chunks
    for chunk, pause in zip(chunks, multipliers):
        if chunk in {
            "「いいえ。もしお嫌じゃなかったら、一度どうぞ、のみにいらして下さい」",
            "「ありがとうございます。はずかしいわ」",
        }:
            assert pause == 1


def test_compute_chunk_pause_multipliers_does_not_promote_dialogue_bridge_between_quotes() -> None:
    text = (
        "前文。\n\n"
        "「実をいうとちょっと面白いことがあったんだ。いいか。こいつはここだけの話だぜ」\n"
        "男は首を突き出しながら声をひそめ、\n"
        "「五年くらい前のことかな。博士の研究室にたびたび大きな荷物が送られてくるんだよ。\n\n"
        "後文。"
    )
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)

    for target in (
        "「実をいうとちょっと面白いことがあったんだ。いいか。こいつはここだけの話だぜ」",
        "男は首を突き出しながら声をひそめ、",
        "「五年くらい前のことかな。博士の研究室にたびたび大きな荷物が送られてくるんだよ。",
    ):
        idx = next(i for i, chunk in enumerate(chunks) if chunk == target)
        assert multipliers[idx] == 1


def test_compute_chunk_pause_multipliers_does_not_promote_sentence_continuation_line() -> None:
    text = (
        "どれだけ自分を、歪めていたんだろう。\n"
        "それはきっと悲しいことだ。なのに、\n"
        "──僕にとっては、能力で歪んでしまった君が、相麻菫なんだよ。"
    )
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == [
        "どれだけ自分を、歪めていたんだろう。",
        "それはきっと悲しいことだ。なのに、",
        "──僕にとっては、能力で歪んでしまった君が、相麻菫なんだよ。",
    ]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers == [1, 1, 1]


def test_compute_chunk_pause_multipliers_does_not_promote_continuation_before_dash_line() -> None:
    text = "それと、\n──応援してるから"
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["それと、", "──応援してるから"]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers == [1, 1]


def test_compute_chunk_pause_multipliers_does_not_promote_continuation_before_plain_line() -> None:
    text = "歌詞を見るととても日本語とは思えない。これが実はヘブライ語だと提唱したのは神学博士で、翻訳すれば、\n聖前に主を讃えよ"
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == [
        "歌詞を見るととても日本語とは思えない。これが実はヘブライ語だと提唱したのは神学博士で、翻訳すれば、",
        "聖前に主を讃えよ",
    ]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers == [1, 1]


def test_compute_chunk_pause_multipliers_keeps_section_break_after_continuation() -> None:
    text = f"それと、\n\n{SECTION_BREAK}\n\n次。"
    spans = tts_util.make_chunk_spans(text, max_chars=220, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["それと、", "次。"]
    multipliers = tts_util.compute_chunk_pause_multipliers(text, spans)
    assert multipliers[0] >= 3
    assert multipliers[1] == 1


def test_chunk_book_writes_manifest(tmp_path: Path) -> None:
    book_dir = tmp_path / "book"
    clean_dir = book_dir / "clean" / "chapters"
    clean_dir.mkdir(parents=True)
    chapter_path = clean_dir / "0001-chapter.txt"
    chapter_path.write_text("今日は良い天気です。", encoding="utf-8")

    toc = {
        "created_unix": 0,
        "metadata": {"title": "Sample"},
        "chapters": [
            {
                "index": 1,
                "title": "Chapter 1",
                "path": chapter_path.relative_to(book_dir).as_posix(),
            }
        ],
    }
    toc_path = book_dir / "clean" / "toc.json"
    toc_path.write_text(json.dumps(toc, ensure_ascii=False), encoding="utf-8")

    manifest = tts_util.chunk_book(book_dir)
    assert manifest["chapters"]
    assert (book_dir / "tts" / "manifest.json").exists()


def test_prepare_manifest_applies_chapter_boundary_pause_multiplier(
    tmp_path: Path,
) -> None:
    chapters = [
        tts_util.ChapterInput(
            index=1,
            id="c1",
            title="Chapter 1",
            text="第一章本文。",
            path="clean/chapters/0001-c1.txt",
        ),
        tts_util.ChapterInput(
            index=2,
            id="c2",
            title="Chapter 2",
            text="第二章本文。",
            path="clean/chapters/0002-c2.txt",
        ),
    ]
    manifest, _chunks = tts_util._prepare_manifest(
        chapters=chapters,
        out_dir=tmp_path / "tts",
        voice="voice",
        max_chars=120,
        pad_ms=300,
        rechunk=True,
    )
    first = manifest["chapters"][0]["pause_multipliers"]
    second = manifest["chapters"][1]["pause_multipliers"]
    assert first[-1] >= 5
    assert second[-1] == 1


def test_prepare_voice_prompt_accepts_language_param(tmp_path: Path) -> None:
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF")

    class ModelWithLanguage:
        def __init__(self) -> None:
            self.kwargs = None

        def create_voice_clone_prompt(self, ref_audio, ref_text=None, language=None, x_vector_only_mode=False):
            self.kwargs = {
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "language": language,
                "x_vector_only_mode": x_vector_only_mode,
            }
            return {"ok": True}

    model = ModelWithLanguage()
    config = voice_util.VoiceConfig(
        name="man",
        ref_audio=str(audio_path),
        ref_text="こんにちは",
        language="Japanese",
        x_vector_only_mode=False,
    )
    prompt, language = tts_util._prepare_voice_prompt(model, config)
    assert prompt == {"ok": True}
    assert language == "Japanese"
    assert model.kwargs["language"] == "Japanese"


def test_prepare_voice_prompt_ignores_language_when_unsupported(tmp_path: Path) -> None:
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF")

    class ModelNoLanguage:
        def __init__(self) -> None:
            self.kwargs = None

        def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
            self.kwargs = {
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "x_vector_only_mode": x_vector_only_mode,
            }
            return {"ok": True}

    model = ModelNoLanguage()
    config = voice_util.VoiceConfig(
        name="man",
        ref_audio=str(audio_path),
        ref_text="こんにちは",
        language="Japanese",
        x_vector_only_mode=False,
    )
    prompt, language = tts_util._prepare_voice_prompt(model, config)
    assert prompt == {"ok": True}
    assert language == "Japanese"
    assert "language" not in model.kwargs


def test_generate_audio_prefers_voice_clone() -> None:
    class ModelVoiceClone:
        def __init__(self) -> None:
            self.kwargs = None

        def generate_voice_clone(self, text, language=None, voice_clone_prompt=None, non_streaming_mode=False):
            self.kwargs = {
                "text": text,
                "language": language,
                "voice_clone_prompt": voice_clone_prompt,
                "non_streaming_mode": non_streaming_mode,
            }
            return [tts_util.np.zeros(1)], 24000

    model = ModelVoiceClone()
    wavs, rate = tts_util._generate_audio(
        model=model, text="こんにちは", prompt={"p": 1}, language="Japanese"
    )
    assert rate == 24000
    assert model.kwargs["language"] == "Japanese"
    assert model.kwargs["voice_clone_prompt"] == {"p": 1}


def test_apply_reading_overrides() -> None:
    text = "私は漢字と東京。"
    overrides = [
        {"base": "漢字", "reading": "かんじ"},
        {"base": "東京", "reading": "とうきょう"},
    ]
    assert (
        tts_util.apply_reading_overrides(text, overrides)
        == "私はかんじととうきょう。"
    )


def test_apply_reading_overrides_first_only() -> None:
    text = "山田山田"
    overrides = [{"base": "山田", "reading": "やまだ", "mode": "first"}]
    assert tts_util.apply_reading_overrides(text, overrides) == "やまだ山田"


def test_apply_reading_overrides_isolated_kanji() -> None:
    text = "「間」は間違い。"
    overrides = [{"base": "間", "reading": "ま", "mode": "isolated"}]
    assert tts_util.apply_reading_overrides(text, overrides) == "「ま」は間違い。"


def test_apply_reading_overrides_kanji_boundary() -> None:
    text = "男の子と男女、人々と人は。"
    overrides = [
        {"base": "男", "reading": "おとこ", "mode": "kanji"},
        {"base": "人", "reading": "ひと", "mode": "kanji"},
    ]
    assert (
        tts_util.apply_reading_overrides(text, overrides)
        == "おとこの子と男女、人々とひとは。"
    )


def test_apply_reading_overrides_isolated_kanji_no_compound() -> None:
    text = "なんだ、これが天国か？"
    overrides = [{"base": "天", "reading": "てん", "mode": "isolated"}]
    assert tts_util.apply_reading_overrides(text, overrides) == text


def test_apply_reading_overrides_isolated_kanji_no_okurigana() -> None:
    text = "好きにすれば"
    overrides = [{"base": "好", "reading": "こう", "mode": "isolated"}]
    assert tts_util.apply_reading_overrides(text, overrides) == text


def test_split_reading_overrides_single_kanji_mode() -> None:
    data = {
        "chapters": {
            "c1": {
                "replacements": [
                    {"base": "天", "reading": "てん"},
                    {"base": "人", "reading": "ひと", "mode": "isolated"},
                    {"base": "天国", "reading": "てんごく"},
                ]
            }
        }
    }
    _global, chapters = tts_util._split_reading_overrides_data(data)
    entries = chapters["c1"]
    entry_map = {item.get("base"): item for item in entries}
    assert entry_map["天"].get("mode") == "isolated"
    assert entry_map["人"].get("mode") == "isolated"
    assert "mode" not in entry_map["天国"]


def test_apply_reading_overrides_regex() -> None:
    text = "御存知でした。御迷惑でした。"
    overrides = [{"base": "re:御(存知|迷惑)", "reading": r"ご\1"}]
    assert tts_util.apply_reading_overrides(text, overrides) == "ご存知でした。ご迷惑でした。"


def test_apply_reading_overrides_for_tts_formats_readings() -> None:
    text = "私は漢字と東京。"
    overrides = [
        {"base": "漢字", "reading": "かんじ"},
        {"base": "東京", "reading": "とうきょう"},
    ]
    sep = _default_first_token_separator()
    assert (
        tts_util._apply_reading_overrides_for_tts(text, overrides)
        == f"私は{sep}かんじ{sep}と{sep}とうきょう{sep}。"
    )


def test_apply_ruby_spans() -> None:
    text = "前漢字後"
    spans = [{"start": 1, "end": 3, "base": "漢字", "reading": "かんじ"}]
    assert tts_util.apply_ruby_spans(text, spans) == "前かんじ後"


def test_apply_ruby_evidence_to_chunk_formats_readings() -> None:
    chunk_text = "前日本。後半。"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [{"start": 1, "end": 3, "base": "日本", "reading": "にほん"}]
    ruby_data = {"global": [{"base": "後半", "reading": "こうはん"}]}
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        ruby_data,
    )
    sep = _default_first_token_separator()
    assert out == f"前{sep}にほん{sep}。{sep}こうはん{sep}。"


def test_apply_ruby_evidence_to_chunk_skips_single_kanji_global() -> None:
    chunk_text = "全13巻。"
    chunk_span = (0, len(chunk_text))
    ruby_data = {"global": [{"base": "巻", "reading": "ま"}]}
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        [],
        ruby_data,
    )
    assert out == chunk_text


def test_apply_ruby_evidence_to_chunk_drops_suspicious_span() -> None:
    chunk_text = "本町に西田医院という耳鼻科があったのを覚えているか。"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [
        {"start": 0, "end": 2, "base": "本町", "reading": "ほんちょう"},
        {
            "start": 7,
            "end": 10,
            "base": "という",
            "reading": "あ" * 30,
        },
    ]
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        {},
    )
    sep = _default_first_token_separator()
    assert out == f"{sep}ほんちょう{sep}に西田医院という耳鼻科があったのを覚えているか。"


def test_is_suspicious_ruby_span_rejects_kanji_in_reading() -> None:
    assert tts_util._is_suspicious_ruby_span("可", "おかしそうに女は")


def test_apply_ruby_evidence_to_chunk_keeps_non_kanji_span() -> None:
    chunk_text = "TTSを試す。"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [
        {"start": 0, "end": 3, "base": "TTS", "reading": "ティーティーエス"}
    ]
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        {},
    )
    sep = _default_first_token_separator()
    assert out == f"{sep}ティーティーエス{sep}を試す。"


def test_apply_ruby_evidence_to_chunk_keeps_inline_ruby_over_global_override() -> None:
    chunk_text = "暗殺者は暗殺者だ。"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [
        {"start": 0, "end": 3, "base": "暗殺者", "reading": "おれ"},
    ]
    ruby_data = {"global": [{"base": "暗殺者", "reading": "あんさつしや"}]}
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        ruby_data,
    )
    sep = _default_first_token_separator()
    assert out.startswith(f"{sep}おれ{sep}は")
    assert f"{sep}あんさつしゃ{sep}だ。" in out
    assert "暗殺者" not in out


def test_apply_ruby_evidence_to_chunk_keeps_explicit_katakana_ruby() -> None:
    chunk_text = "前森野後"
    chunk_span = (0, len(chunk_text))
    chapter_spans = [{"start": 1, "end": 3, "base": "森野", "reading": "モリノ"}]
    out = tts_util._apply_ruby_evidence_to_chunk(
        chunk_text,
        chunk_span,
        chapter_spans,
        {},
    )
    sep = _default_first_token_separator()
    assert out == f"前{sep}モリノ{sep}後"


def test_normalize_kana_with_stub_tagger() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("漢字", "カンジ"),
                DummyToken("と", None),
                DummyToken("東京", "トウキョウ"),
                DummyToken("未知", "*"),
                DummyToken("X", None),
            ]

    out = tts_util._normalize_kana_with_tagger("漢字と東京未知X", DummyTagger())
    assert out == "カンジとトウキョウ未知X"


def test_normalize_kana_with_stub_tagger_hiragana() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("漢字", "カンジ"),
                DummyToken("と", None),
                DummyToken("東京", "トウキョウ"),
                DummyToken("未知", "*"),
                DummyToken("X", None),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "漢字と東京未知X", DummyTagger(), kana_style="hiragana"
    )
    assert out == "かんじととうきょう未知X"


def test_normalize_kana_with_stub_tagger_katakana() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("漢字", "カンジ"),
                DummyToken("と", None),
                DummyToken("東京", "トウキョウ"),
                DummyToken("未知", "*"),
                DummyToken("X", None),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "漢字と東京未知X", DummyTagger(), kana_style="katakana"
    )
    assert out == "カンジとトウキョウ未知X"


def test_normalize_kana_with_stub_tagger_mixed_okurigana() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [DummyToken("始まる", "ハジマル")]

    out = tts_util._normalize_kana_with_tagger(
        "始まる", DummyTagger(), kana_style="mixed"
    )
    assert out == "ハジまる"


def test_normalize_kana_preserves_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, text: str):
            if text == "漢字":
                return [DummyToken("漢字", "カンジ")]
            if text == "東京":
                return [DummyToken("東京", "トウキョウ")]
            return [DummyToken(text, None)]

    monkeypatch.setattr(tts_util, "_get_kana_tagger", lambda: DummyTagger())
    out = tts_util._normalize_kana_text("漢字 と 東京")
    assert out == "カンジ と トウキョウ"


def test_normalize_kana_preserves_spaces_hiragana(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, text: str):
            if text == "漢字":
                return [DummyToken("漢字", "カンジ")]
            if text == "東京":
                return [DummyToken("東京", "トウキョウ")]
            return [DummyToken(text, None)]

    monkeypatch.setattr(tts_util, "_get_kana_tagger", lambda: DummyTagger())
    out = tts_util._normalize_kana_text("漢字 と 東京", kana_style="hiragana")
    assert out == "かんじ と とうきょう"


def test_normalize_kana_with_stub_tagger_partial() -> None:
    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            goshu: str | None = None,
            pos1: str | None = None,
            pos2: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.goshu = goshu
            self.pos1 = pos1
            self.pos2 = pos2

    class DummyToken:
        def __init__(self, surface: str, feature: DummyFeature) -> None:
            self.surface = surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("漢字", DummyFeature("カンジ", goshu="漢", pos1="名詞")),
                DummyToken("始まる", DummyFeature("ハジマル", goshu="和", pos1="動詞")),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "漢字始まる",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon={"漢字"},
        partial_mid_kanji=True,
    )
    assert out == "\\かんじ\\始まる"


def test_normalize_kana_with_stub_tagger_partial_common_noun(monkeypatch) -> None:
    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            goshu: str | None = None,
            pos1: str | None = None,
            pos2: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.goshu = goshu
            self.pos1 = pos1
            self.pos2 = pos2

    class DummyToken:
        def __init__(self, surface: str, feature: DummyFeature) -> None:
            self.surface = surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("自分", DummyFeature("ジブン", goshu="漢", pos1="名詞", pos2="普通名詞"))
            ]

    monkeypatch.setattr(tts_util, "_is_common_japanese_word", lambda _word: True)
    out = tts_util._normalize_kana_with_tagger(
        "自分",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        partial_mid_kanji=True,
    )
    assert out == "自分"


def test_normalize_kana_with_stub_tagger_partial_honorific_prefix(monkeypatch) -> None:
    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            pos1: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.pos1 = pos1

    class DummyToken:
        def __init__(self, surface: str, feature: DummyFeature) -> None:
            self.surface = surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("御", DummyFeature("ゴ", pos1="接頭辞")),
                DummyToken("存知", DummyFeature("ゾンジ", pos1="名詞")),
            ]

    monkeypatch.setattr(tts_util, "_is_common_japanese_word", lambda _word: False)
    out = tts_util._normalize_kana_with_tagger(
        "御存知",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        partial_mid_kanji=True,
    )
    assert out == "\\ご\\ぞんじ\\"


def test_normalize_kana_with_stub_tagger_partial_honorific_prefix_other_reading(
    monkeypatch,
) -> None:
    class DummyFeature:
        def __init__(self, kana: str | None, pos1: str | None = None) -> None:
            self.kana = kana
            self.pron = kana
            self.pos1 = pos1

    class DummyToken:
        def __init__(self, surface: str, feature: DummyFeature) -> None:
            self.surface = surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("御", DummyFeature("ミ", pos1="接頭辞")),
                DummyToken("子", DummyFeature("コ", pos1="名詞")),
            ]

    monkeypatch.setattr(tts_util, "_is_common_japanese_word", lambda _word: False)
    out = tts_util._normalize_kana_with_tagger(
        "御子",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
    )
    assert out == "\\み\\子"


def test_normalize_kana_with_stub_tagger_partial_honorific_prefix_go_o_choice(
    monkeypatch,
) -> None:
    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            goshu: str | None = None,
            pos1: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.goshu = goshu
            self.pos1 = pos1

    class DummyToken:
        def __init__(self, surface: str, feature: DummyFeature) -> None:
            self.surface = surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, text: str):
            if text == "存知":
                return [DummyToken("存知", DummyFeature("ゾンジ", goshu="漢", pos1="名詞"))]
            return [
                DummyToken("御", DummyFeature("オ", pos1="接頭辞")),
                DummyToken("存知", DummyFeature("ゾンジ", goshu="漢", pos1="名詞")),
            ]

    monkeypatch.setattr(tts_util, "_is_common_japanese_word", lambda _word: False)
    out = tts_util._normalize_kana_with_tagger(
        "御存知",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        partial_mid_kanji=True,
    )
    assert out == "\\ご\\ぞんじ\\"


@pytest.mark.parametrize(
    ("surface", "lemma", "lemma_reading", "reading", "expected"),
    [
        ("立上り", "立ち上がる", "タチアガル", "タチアガリ", "立ち上がり"),
        ("立上がり", "立ち上がり", "タチアガリ", "タチアガリ", "立ち上がり"),
        ("立ち上り", "立ち上がり", "タチアガリ", "タチアガリ", "立ち上がり"),
        ("取消し", "取り消し", "トリケシ", "トリケシ", "取り消し"),
        ("取戻す", "取り戻す", "トリモドス", "トリモドス", "取り戻す"),
        ("立入る", "立ち入る", "タチイル", "タチイル", "立ち入る"),
    ],
)
def test_normalize_kana_partial_canonicalizes_okurigana_variants(
    surface: str,
    lemma: str,
    lemma_reading: str,
    reading: str,
    expected: str,
) -> None:
    class DummyFeature:
        def __init__(self) -> None:
            self.kana = reading
            self.pron = reading
            self.lemma = lemma
            self.lForm = lemma_reading

    class DummyToken:
        def __init__(self, token_surface: str, feature: DummyFeature) -> None:
            self.surface = token_surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, _text: str):
            return [DummyToken(surface, DummyFeature())]

    out = tts_util._normalize_kana_with_tagger(
        surface,
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
    )
    assert out == expected


def test_normalize_kana_with_stub_tagger_partial_common_noun_guard(monkeypatch) -> None:
    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            goshu: str | None = None,
            pos1: str | None = None,
            pos2: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.goshu = goshu
            self.pos1 = pos1
            self.pos2 = pos2

    class DummyToken:
        def __init__(self, surface: str, feature: DummyFeature) -> None:
            self.surface = surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("前略", DummyFeature("ゼンリャク", goshu="漢", pos1="名詞", pos2="普通名詞"))
            ]

    monkeypatch.setattr(tts_util, "_is_common_japanese_word", lambda _word: False)
    out = tts_util._normalize_kana_with_tagger(
        "前略",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        partial_mid_kanji=True,
    )
    assert out == "\\ぜんりゃく\\"


def test_normalize_kana_with_stub_tagger_partial_rare() -> None:
    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            goshu: str | None = None,
            pos1: str | None = None,
            pos2: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.goshu = goshu
            self.pos1 = pos1
            self.pos2 = pos2

    class DummyToken:
        def __init__(self, surface: str, feature: DummyFeature) -> None:
            self.surface = surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("妻子", DummyFeature("サイシ", goshu="漢", pos1="名詞", pos2="普通名詞"))
            ]

    out = tts_util._normalize_kana_with_tagger(
        "妻子",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon={"妻子"},
        partial_mid_kanji=True,
    )
    assert out == "\\さいし\\"


def _normalize_ruby_reading_with_stub(
    base: str, reading: str, base_reading_kata: str
) -> str:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [DummyToken(base, base_reading_kata)]

    tts_util._RUBY_BASE_READING_CACHE.clear()
    return tts_util._normalize_ruby_reading(base, reading, DummyTagger())


@pytest.mark.parametrize(
    ("base", "reading", "base_reading_kata", "expected"),
    [
        # GOSICK 全9冊合本版
        ("京介", "きようすけ", "キョウスケ", "きょうすけ"),
        ("豪奢", "ごうしや", "ゴウシャ", "ごうしゃ"),
        ("呪詛", "じゆそ", "ジュソ", "じゅそ"),
        ("几帳面", "きちようめん", "キチョウメン", "きちょうめん"),
        ("戦々恐々", "せんせんきようきよう", "センセンキヨウキョウ", "せんせんきようきょう"),
        # 【合本版】さくら荘のペットな彼女 全13巻（電子特典付き）
        ("表情", "ひようじよう", "ヒョウジョウ", "ひょうじょう"),
        ("一緒", "いつしよ", "イッショ", "いっしょ"),
        ("趣味", "しゆみ", "シュミ", "しゅみ"),
        # 【合本版】まよチキ！ 全12巻
        ("粛清", "しゆくせい", "シュクセイ", "しゅくせい"),
        ("驚愕", "きようがく", "キョウガク", "きょうがく"),
        ("卑怯", "ひきよう", "ヒキョウ", "ひきょう"),
        # 【合本版】俺の妹がこんなに可愛いわけがない 全12冊収録
        ("近況", "きんきよう", "キンキョウ", "きんきょう"),
        ("器量", "きりよう", "キリョウ", "きりょう"),
        # サクラダリセット（角川文庫）【全7冊 合本版】
        ("臆病", "おくびよう", "オクビョウ", "おくびょう"),
        ("信憑性", "しんぴようせい", "シンピョウセイ", "しんぴょうせい"),
        # 異人たちとの夏
        ("南無妙法蓮華経", "なむみようほうれんげきよう", "ナムミョウホウレンゲキョウ", "なむみょうほうれんげきょう"),
        # Ｖシリーズ全10冊合本版
        ("洒落", "しやれ", "シャレ", "しゃれ"),
        ("自重", "じちよう", "ジチョウ", "じちょう"),
    ],
)
def test_normalize_ruby_reading_sutegana_real_pairs(
    base: str, reading: str, base_reading_kata: str, expected: str
) -> None:
    out = _normalize_ruby_reading_with_stub(base, reading, base_reading_kata)
    assert out == expected


@pytest.mark.parametrize(
    ("base", "reading", "base_reading_kata", "expected"),
    [
        # Synthetic coverage to validate the full sutegana mapping surface.
        ("京極堂", "きようごくどう", "キョウゴクドウ", "きょうごくどう"),
        ("野球屋", "やきゆうや", "ヤキュウヤ", "やきゅうや"),
        ("学校通信", "がつこうつうしん", "ガッコウツウシン", "がっこうつうしん"),
        ("未知語", "ふあいる", "ファイル", "ふぁいる"),
        ("未知語", "ていあ", "ティア", "てぃあ"),
        ("未知語", "くわい", "クヮイ", "くゎい"),
        ("未知語", "かかく", "ヵカク", "ゕかく"),
        ("未知語", "カカク", "ヵカク", "ヵカク"),
    ],
)
def test_normalize_ruby_reading_sutegana_dummy_pairs(
    base: str, reading: str, base_reading_kata: str, expected: str
) -> None:
    out = _normalize_ruby_reading_with_stub(base, reading, base_reading_kata)
    assert out == expected


@pytest.mark.parametrize(
    ("base", "reading", "base_reading_kata"),
    [
        ("京介", "きようすけ", "キョウズケ"),
        ("野球屋", "やきゆうや", "ヤキュウバ"),
        ("学園生活", "にちじよう", "ガクエンセイカツ"),
    ],
)
def test_normalize_ruby_reading_sutegana_keeps_unmatched(
    base: str, reading: str, base_reading_kata: str
) -> None:
    out = _normalize_ruby_reading_with_stub(base, reading, base_reading_kata)
    assert out == reading


def test_normalize_kana_with_stub_tagger_partial_kanji_run_convert() -> None:
    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            goshu: str | None = None,
            pos1: str | None = None,
            pos2: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.goshu = goshu
            self.pos1 = pos1
            self.pos2 = pos2

    class DummyToken:
        def __init__(self, surface: str, feature: DummyFeature) -> None:
            self.surface = surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("漢", DummyFeature("カン", goshu="漢", pos1="名詞")),
                DummyToken("字", DummyFeature("ジ", goshu="漢", pos1="名詞")),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "漢字",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon={"漢字"},
        partial_mid_kanji=True,
    )
    assert out == "\\かんじ\\"


def test_normalize_kana_with_stub_tagger_partial_numeric_counter() -> None:
    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            goshu: str | None = None,
            pos1: str | None = None,
            pos2: str | None = None,
            pos3: str | None = None,
            pos4: str | None = None,
            type_: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.goshu = goshu
            self.pos1 = pos1
            self.pos2 = pos2
            self.pos3 = pos3
            self.pos4 = pos4
            self.type = type_

    class DummyToken:
        def __init__(self, surface: str, feature: DummyFeature) -> None:
            self.surface = surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("十", DummyFeature("トオ", pos1="名詞", pos2="数詞", type_="数")),
                DummyToken("日", DummyFeature("カ", pos1="接尾辞", pos2="名詞的", pos3="助数詞", type_="助数")),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "十日", DummyTagger(), kana_style="partial", zh_lexicon=set()
    )
    assert out == "\\とおか\\"


def test_normalize_kana_with_stub_tagger_partial_kanji_run_keep() -> None:
    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            goshu: str | None = None,
            pos1: str | None = None,
            pos2: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.goshu = goshu
            self.pos1 = pos1
            self.pos2 = pos2

    class DummyToken:
        def __init__(self, surface: str, feature: DummyFeature) -> None:
            self.surface = surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("漢", DummyFeature("カン", goshu="漢", pos1="名詞")),
                DummyToken("字", DummyFeature("ジ", goshu="漢", pos1="名詞")),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "漢字", DummyTagger(), kana_style="partial", zh_lexicon=set()
    )
    assert out == "漢字"


def test_normalize_kana_first_token_partial() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("投げつけ", "ナゲツケ"),
                DummyToken("好き", "スキ"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "投げつけ好き",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"なげつけ{_default_first_token_separator()}好き"


def test_normalize_kana_first_token_partial_separator_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("投げつけ", "ナゲツケ"),
                DummyToken("好き", "スキ"),
            ]

    monkeypatch.setenv(tts_util.FIRST_TOKEN_SEPARATOR_ENV, "none")
    out = tts_util._normalize_kana_with_tagger(
        "投げつけ好き",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == "なげつけ好き"


def test_normalize_kana_first_token_partial_separator_custom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("投げつけ", "ナゲツケ"),
                DummyToken("好き", "スキ"),
            ]

    monkeypatch.setenv(tts_util.FIRST_TOKEN_SEPARATOR_ENV, "・")
    out = tts_util._normalize_kana_with_tagger(
        "投げつけ好き",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == "なげつけ・好き"


def test_normalize_kana_first_token_partial_separator_after_sokuon() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("待っ", "マッ"),
                DummyToken("て", "テ"),
                DummyToken("くれ", "クレ"),
                DummyToken("。", "。"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "待ってくれ。",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"まって{_default_first_token_separator()}くれ。"


def test_normalize_kana_first_token_partial_kanji_run() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("自転", "ジテン"),
                DummyToken("車", "シャ"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "自転車",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"じてんしゃ{_default_first_token_separator()}"


def test_normalize_kana_first_token_partial_numeric_counter_run() -> None:
    class DummyFeature:
        def __init__(
            self,
            kana: str | None,
            pos1: str | None = None,
            pos2: str | None = None,
            pos3: str | None = None,
            type_: str | None = None,
        ) -> None:
            self.kana = kana
            self.pron = kana
            self.pos1 = pos1
            self.pos2 = pos2
            self.pos3 = pos3
            self.type = type_

    class DummyToken:
        def __init__(self, surface: str, feature: DummyFeature) -> None:
            self.surface = surface
            self.feature = feature

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("十", DummyFeature("トオ", pos1="名詞", pos2="数詞", type_="数")),
                DummyToken(
                    "日",
                    DummyFeature(
                        "カ",
                        pos1="接尾辞",
                        pos2="名詞的",
                        pos3="助数詞",
                        type_="助数",
                    ),
                ),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "十日",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"とおか{_default_first_token_separator()}"


def test_normalize_kana_first_token_already_kana() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("かな", "カナ"),
                DummyToken("漢字", "カンジ"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "かな漢字",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == "かな漢字"


def test_normalize_kana_first_token_to_kana_latin_before_kanji() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("TV", "ティーブイ"),
                DummyToken("化", "カ"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "TV化",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"ティーブイ{_default_first_token_separator()}化"


def test_normalize_kana_first_token_to_kana_latin_starter() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("Ｖ", "ブイ"),
                DummyToken("シリーズ", "シリーズ"),
                DummyToken("全", "ゼン"),
                DummyToken("十冊", "ジッサツ"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "Ｖシリーズ全十冊",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"ブイ{_default_first_token_separator()}シリーズ全十冊"


def test_normalize_kana_text_force_first_latin_phrase() -> None:
    text = "The Yellow Monkey「天国旅行」"
    out = tts_util._normalize_kana_text(
        text,
        kana_style="partial",
        force_first_token_to_kana=True,
    )
    assert out == f"ザ イエロー モンキー{_default_first_token_separator()}「天国旅行」"


def test_normalize_kana_text_force_first_token_chunk_only() -> None:
    text = "あ The Monkey「天国旅行」"
    out = tts_util._normalize_kana_text(
        text,
        kana_style="partial",
        force_first_token_to_kana=True,
    )
    assert "The" in out


def test_normalize_kana_first_token_to_kana_lowercase_latin_uses_reading() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("Rom", "ロム"),
                DummyToken("を", "ヲ"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "Romを",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"ロム{_default_first_token_separator()}を"


def test_normalize_kana_first_token_to_kana_lowercase_latin_fallback() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("Rom", None),
                DummyToken("を", "ヲ"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "Romを",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"ロム{_default_first_token_separator()}を"


def test_normalize_kana_first_token_to_kana_leading_kanji() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("漢字", "カンジ"),
                DummyToken("テスト", "テスト"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "漢字テスト",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"かんじ{_default_first_token_separator()}テスト"


def test_normalize_kana_first_token_to_kana_handles_mixed_single_token() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [DummyToken("TVアニメ", "ティーブイアニメ"), DummyToken("化", "カ")]

    out = tts_util._normalize_kana_with_tagger(
        "TVアニメ化",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"ティーブイアニメ{_default_first_token_separator()}化"


def test_normalize_kana_first_token_to_kana_mixed_no_reading() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("Rom取り出し", None),
                DummyToken("た", "タ"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "Rom取り出した",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"ロム取り出し{_default_first_token_separator()}た"


def test_normalize_kana_first_token_to_kana_skips_epub_noise() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("c1VZ1", "シーワンブイゼットワン"),
                DummyToken("本文", "ホンブン"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "c1VZ1本文",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == "c1VZ1本文"


@pytest.mark.parametrize(
    ("text", "tokens", "expected"),
    [
        # 【合本版】さくら荘のペットな彼女 全13巻（電子特典付き）
        (
            "ＨＲ前の教室では、クラスメイトの女子から何度も同じ質問をされた。",
            [
                ("ＨＲ", "エイチアール"),
                ("前の教室では、クラスメイトの女子から何度も同じ質問をされた。", None),
            ],
            f"エイチアール{_default_first_token_separator()}前の教室では、クラスメイトの女子から何度も同じ質問をされた。",
        ),
        (
            "ＴＶの前では、よく美咲と一緒にゲームをした。",
            [("ＴＶ", "ティーブイ"), ("の前では、よく美咲と一緒にゲームをした。", None)],
            f"ティーブイ{_default_first_token_separator()}の前では、よく美咲と一緒にゲームをした。",
        ),
        (
            "ＰＣの前に座って龍之介に疑問をぶつける。",
            [("ＰＣ", "ピーシー"), ("の前に座って龍之介に疑問をぶつける。", None)],
            f"ピーシー{_default_first_token_separator()}の前に座って龍之介に疑問をぶつける。",
        ),
        (
            "ＯＳが起動すると、まずはさくら荘の自室に設置してあるサーバーに接続する。",
            [
                ("ＯＳ", "オーエス"),
                ("が起動すると、まずはさくら荘の自室に設置してあるサーバーに接続する。", None),
            ],
            f"オーエス{_default_first_token_separator()}が起動すると、まずはさくら荘の自室に設置してあるサーバーに接続する。",
        ),
        (
            "ＳＤの何倍も時間がかかるんだもん！",
            [("ＳＤ", "エスディー"), ("の何倍も時間がかかるんだもん！", None)],
            f"エスディー{_default_first_token_separator()}の何倍も時間がかかるんだもん！",
        ),
        (
            "ＡＩにもてあそばれる俺って……",
            [("ＡＩ", "エーアイ"), ("にもてあそばれる俺って……", None)],
            f"エーアイ{_default_first_token_separator()}にもてあそばれる俺って……",
        ),
        (
            "Ｕターンした車は、札幌駅のある方へと曲がっていく。",
            [("Ｕ", "ユー"), ("ターンした車は、札幌駅のある方へと曲がっていく。", None)],
            f"ユー{_default_first_token_separator()}ターンした車は、札幌駅のある方へと曲がっていく。",
        ),
    ],
)
def test_normalize_kana_first_token_to_kana_real_sakurasou_cases(
    text: str, tokens: list[tuple[str, str | None]], expected: str
) -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [DummyToken(surface, kana) for surface, kana in tokens]

    out = tts_util._normalize_kana_with_tagger(
        text,
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == expected


@pytest.mark.parametrize(
    ("text", "tokens", "expected"),
    [
        # First-token lowercase should convert when followed by Japanese.
        (
            "ｉｆ』と『ｆｏｒ』の使い方は理解しているな。",
            [("ｉｆ", "イフ"), ("』と『ｆｏｒ』の使い方は理解しているな。", None)],
            f"イフ{_default_first_token_separator()}』と『ｆｏｒ』の使い方は理解しているな。",
        ),
        # Copyright metadata lead should remain unchanged.
        (
            "C)2010-2014 HAJIME KAMOSHIDA ※2010年1月5日発行",
            [
                ("C", "シー"),
                (")", None),
                ("2010-2014", None),
                (" ", None),
                ("HAJIME", None),
                (" ", None),
                ("KAMOSHIDA", None),
                (" ", None),
                ("※2010年1月5日発行", None),
            ],
            "C)2010-2014 HAJIME KAMOSHIDA ※2010年1月5日発行",
        ),
    ],
)
def test_normalize_kana_first_token_to_kana_real_sakurasou_non_targets(
    text: str, tokens: list[tuple[str, str | None]], expected: str
) -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [DummyToken(surface, kana) for surface, kana in tokens]

    out = tts_util._normalize_kana_with_tagger(
        text,
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == expected


def test_normalize_kana_weekday_reading() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __call__(self, _text: str):
            return [
                DummyToken("土曜", "ドヨウ"),
                DummyToken("日", "ヒ"),
            ]

    out = tts_util._normalize_kana_with_tagger(
        "土曜日",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert out == f"どようび{_default_first_token_separator()}"


def test_normalize_kana_with_tagger_normalizes_kyujitai() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __init__(self) -> None:
            self.last_input = None

        def __call__(self, text: str):
            self.last_input = text
            return [DummyToken(text, "メザメ")]

    tagger = DummyTagger()
    out = tts_util._normalize_kana_with_tagger(
        "目覺め",
        tagger,
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert tagger.last_input == "目覚め"
    assert out == f"めざめ{_default_first_token_separator()}"


def test_normalize_kana_with_tagger_normalizes_kyujitai_uso() -> None:
    class DummyFeature:
        def __init__(self, kana: str | None) -> None:
            self.kana = kana
            self.pron = kana

    class DummyToken:
        def __init__(self, surface: str, kana: str | None) -> None:
            self.surface = surface
            self.feature = DummyFeature(kana)

    class DummyTagger:
        def __init__(self) -> None:
            self.last_input = None

        def __call__(self, text: str):
            self.last_input = text
            return [DummyToken(text, "ウソ")]

    tagger = DummyTagger()
    out = tts_util._normalize_kana_with_tagger(
        "噓",
        tagger,
        kana_style="partial",
        zh_lexicon=set(),
        force_first_token_to_kana=True,
    )
    assert tagger.last_input == "嘘"
    assert out == f"うそ{_default_first_token_separator()}"


def test_synthesize_book_force_first_token_to_kana(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, bool] = {}

    def fake_normalize(
        text: str,
        *,
        kana_style: str = "mixed",
        force_first_token_to_kana: bool = False,
        partial_mid_kanji: bool = False,
        debug_sources: list[str] | None = None,
    ) -> str:
        calls["force_first_token_to_kana"] = force_first_token_to_kana
        return text

    def fake_prepare_manifest(**_kwargs):
        manifest = {
            "chapters": [
                {
                    "id": "c1",
                    "title": "c1",
                    "chunks": ["投げつけ"],
                    "durations_ms": [None],
                    "chunk_spans": [[0, 3]],
                }
            ]
        }
        return manifest, [["投げつけ"]]

    unidic_dir = tmp_path / "unidic"
    unidic_dir.mkdir()
    (unidic_dir / "dicrc").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(tts_util, "_normalize_kana_text", fake_normalize)
    monkeypatch.setattr(tts_util, "_prepare_manifest", fake_prepare_manifest)
    monkeypatch.setattr(tts_util, "_select_backend", lambda _backend: "torch")
    monkeypatch.setattr(tts_util, "_resolve_model_name", lambda _name, _backend: "dummy")
    monkeypatch.setattr(tts_util, "_load_model", lambda **_kwargs: object())
    monkeypatch.setattr(tts_util, "_prepare_voice_prompt", lambda _model, _voice: (None, "Japanese"))
    monkeypatch.setattr(
        tts_util,
        "_generate_audio",
        lambda *_args, **_kwargs: ([tts_util.np.zeros(1)], 24000),
    )
    monkeypatch.setattr(tts_util, "_load_voice_map", lambda _path: {})
    monkeypatch.setattr(tts_util, "_load_reading_overrides", lambda _book_dir: ([], {}))
    monkeypatch.setattr(tts_util, "_load_ruby_data", lambda _book_dir: {})
    monkeypatch.setattr(tts_util, "_get_kana_tagger", lambda: object())
    monkeypatch.setattr(tts_util, "_resolve_unidic_dir", lambda: unidic_dir)
    monkeypatch.setattr(
        tts_util,
        "load_book_chapters",
        lambda _book_dir: [
            tts_util.ChapterInput(
                index=1, id="c1", title="c1", text="投げつけ"
            )
        ],
    )

    voice = voice_util.VoiceConfig(
        name="v", ref_audio=str(tmp_path / "ref.wav"), ref_text="dummy"
    )
    out_dir = tmp_path / "tts"
    result = tts_util.synthesize_book(
        book_dir=tmp_path,
        voice=voice,
        out_dir=out_dir,
        kana_normalize=True,
        kana_style="partial",
    )
    assert result == 0
    assert calls.get("force_first_token_to_kana") is True


def test_synthesize_chunk_prefers_explicit_voice_over_manifest_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "book" / "tts"
    out_dir.mkdir(parents=True)
    manifest = {
        "voice": "高橋一生",
        "chapters": [
            {
                "id": "c1",
                "title": "Chapter 1",
                "voice": "高橋一生",
                "chunks": ["こんにちは"],
                "chunk_spans": [[0, 5]],
                "durations_ms": [None],
            }
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_resolve_voice_config(voice: str, base_dir: Path, **_kwargs):
        captured["voice"] = voice
        return voice_util.VoiceConfig(
            name=voice,
            ref_audio="dummy.wav",
            ref_text="dummy",
            language="Japanese",
            x_vector_only_mode=False,
        )

    monkeypatch.setattr(tts_util, "_select_backend", lambda _backend: "mlx")
    monkeypatch.setattr(tts_util, "_resolve_model_name", lambda _name, _backend: "dummy")
    monkeypatch.setattr(tts_util, "_load_mlx_model", lambda _name: object())
    monkeypatch.setattr(
        tts_util, "_generate_audio_mlx", lambda *_args, **_kwargs: ([tts_util.np.zeros(1)], 24000)
    )
    monkeypatch.setattr(tts_util, "_load_ruby_data", lambda _book_dir: {"chapters": {}})
    monkeypatch.setattr(tts_util, "_load_reading_overrides", lambda _book_dir: ([], {}))
    monkeypatch.setattr(
        tts_util,
        "_prepare_tts_pipeline",
        lambda *_args, **_kwargs: tts_util.TtsPipeline(
            ruby_text="こんにちは",
            after_overrides="こんにちは",
            after_numbers="こんにちは",
            after_kana="こんにちは",
            prepared="こんにちは",
        ),
    )
    monkeypatch.setattr(tts_util, "_write_wav", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tts_util.voice_util, "resolve_voice_config", fake_resolve_voice_config)

    result = tts_util.synthesize_chunk(
        out_dir=out_dir,
        chapter_id="c1",
        chunk_index=0,
        voice="堀江由衣",
        base_dir=tmp_path,
    )
    assert result["status"] == "ok"
    assert captured["voice"] == "堀江由衣"


def test_normalize_numbers_standalone_digits() -> None:
    text = "全7冊 合本版"
    assert tts_util._normalize_numbers(text) == "全ななさつ 合本版"


def test_normalize_numbers_date() -> None:
    text = "2024年7月1日"
    assert tts_util._normalize_numbers(text) == "二千二十四年しちがつついたち"


def test_normalize_numbers_kanji_month() -> None:
    text = "四月"
    assert tts_util._normalize_numbers(text) == "しがつ"


def test_normalize_numbers_kanji_month_day() -> None:
    text = "七月一五日"
    assert tts_util._normalize_numbers(text) == "しちがつじゅうごにち"


def test_normalize_numbers_kanji_month_day_zero() -> None:
    text = "八月一〇日"
    assert tts_util._normalize_numbers(text) == "はちがつとおか"


def test_normalize_numbers_kanji_year_month_zero() -> None:
    text = "二〇一〇年三月"
    assert tts_util._normalize_numbers(text) == "二千十年さんがつ"


def test_normalize_numbers_kanji_time_digits() -> None:
    text = "一二時五九分〇二秒"
    assert tts_util._normalize_numbers(text) == "じゅうにじごじゅうきゅうふんにびょう"


def test_normalize_numbers_kanji_percent_zero() -> None:
    text = "一〇〇パーセント"
    assert tts_util._normalize_numbers(text) == "百パーセント"


def test_normalize_numbers_kanji_year_counter_zero() -> None:
    text = "三〇年"
    assert tts_util._normalize_numbers(text) == "三十年"


def test_normalize_numbers_kanji_digit_day_duration() -> None:
    text = "二〇日ほど経つ"
    assert tts_util._normalize_numbers(text) == "にじゅうにちほど経つ"


def test_normalize_numbers_kanji_digit_sequence_zero() -> None:
    text = "一〇〇の人間"
    assert tts_util._normalize_numbers(text) == "百の人間"


def test_normalize_numbers_kanji_decimal() -> None:
    text = "二．〇"
    assert tts_util._normalize_numbers(text) == "二てんぜろ"


def test_normalize_numbers_decimal() -> None:
    text = "3.14"
    assert tts_util._normalize_numbers(text) == "三てんいちよん"


def test_normalize_numbers_fallback_cardinal() -> None:
    text = "合計13です"
    assert tts_util._normalize_numbers(text) == "合計十三です"


def test_normalize_numbers_counter_wa() -> None:
    text = "１話 土曜日"
    assert tts_util._normalize_numbers(text) == "いちわ 土曜日"


def test_normalize_numbers_preserves_unicode_ellipsis() -> None:
    text = "……三ケ月前"
    assert tts_util._normalize_numbers(text) == "……さんかげつ前"


def test_normalize_numbers_counter_one_readings() -> None:
    assert tts_util._normalize_numbers("一冊") == "いっさつ"
    assert tts_util._normalize_numbers("１回") == "いっかい"
    assert tts_util._normalize_numbers("一分") == "いっぷん"
    assert tts_util._normalize_numbers("一匹") == "いっぴき"
    assert tts_util._normalize_numbers("一杯") == "いっぱい"
    assert tts_util._normalize_numbers("一体") == "いったい"
    assert tts_util._normalize_numbers("一ヶ月") == "いっかげつ"
    assert tts_util._normalize_numbers("一回目") == "いっかいめ"
    assert tts_util._normalize_numbers("一週間") == "いっしゅうかん"
    assert tts_util._normalize_numbers("一通の手紙") == "いっつうの手紙"
    assert tts_util._normalize_numbers("一話だけ") == "いちわだけ"


def test_normalize_numbers_counter_sound_changes() -> None:
    assert tts_util._normalize_numbers("三杯") == "さんばい"
    assert tts_util._normalize_numbers("六杯目") == "ろっぱい目"
    assert tts_util._normalize_numbers("八杯") == "はっぱい"
    assert tts_util._normalize_numbers("十杯") == "じゅっぱい"
    assert tts_util._normalize_numbers("三分") == "さんぷん"
    assert tts_util._normalize_numbers("四分") == "よんぷん"
    assert tts_util._normalize_numbers("六分間") == "ろっぷんかん"
    assert tts_util._normalize_numbers("八分") == "はっぷん"
    assert tts_util._normalize_numbers("十分待つ") == "じゅっぷん待つ"
    assert tts_util._normalize_numbers("十分間隔") == "じゅっぷんかんかく"
    assert tts_util._normalize_numbers("三本") == "さんぼん"
    assert tts_util._normalize_numbers("六本") == "ろっぽん"
    assert tts_util._normalize_numbers("八本") == "はっぽん"
    assert tts_util._normalize_numbers("十本") == "じゅっぽん"
    assert tts_util._normalize_numbers("三匹") == "さんびき"
    assert tts_util._normalize_numbers("六匹") == "ろっぴき"
    assert tts_util._normalize_numbers("八匹") == "はっぴき"
    assert tts_util._normalize_numbers("十匹") == "じゅっぴき"
    assert tts_util._normalize_numbers("八個") == "はっこ"
    assert tts_util._normalize_numbers("十個") == "じゅっこ"
    assert tts_util._normalize_numbers("六個") == "ろっこ"
    assert tts_util._normalize_numbers("三軒") == "さんげん"
    assert tts_util._normalize_numbers("八軒") == "はっけん"
    assert tts_util._normalize_numbers("十軒") == "じゅっけん"
    assert tts_util._normalize_numbers("八カ月") == "はっかげつ"
    assert tts_util._normalize_numbers("六ヵ月") == "ろっかげつ"
    assert tts_util._normalize_numbers("十ヶ月") == "じゅっかげつ"
    assert tts_util._normalize_numbers("八回") == "はっかい"
    assert tts_util._normalize_numbers("十回目") == "じゅっかいめ"


def test_normalize_numbers_juubun_enough() -> None:
    assert tts_util._normalize_numbers("十分に注意する") == "じゅうぶんに注意する"
    assert tts_util._normalize_numbers("十分な準備") == "じゅうぶんな準備"
    assert tts_util._normalize_numbers("十分だ") == "じゅうぶんだ"
    assert tts_util._normalize_numbers("十分です") == "じゅうぶんです"
    assert tts_util._normalize_numbers("十分でしょう") == "じゅうぶんでしょう"
    assert tts_util._normalize_numbers("十分である") == "じゅうぶんである"
    assert tts_util._normalize_numbers("十分。") == "じゅうぶん。"
    assert tts_util._normalize_numbers("十二分に伝わる") == "じゅうにぶんに伝わる"
    assert tts_util._normalize_numbers("百分に活かす") == "ひゃくぶんに活かす"
    assert tts_util._normalize_numbers("十分後") == "じゅっぷん後"
    assert tts_util._normalize_numbers("十分だけ待つ") == "じゅっぷんだけ待つ"


def test_normalize_numbers_counter_one_guards() -> None:
    assert tts_util._normalize_numbers("一通り終えた") == "一通り終えた"
    assert tts_util._normalize_numbers("一回り大きい") == "一回り大きい"
    assert tts_util._normalize_numbers("一個人の判断") == "一個人の判断"
    assert tts_util._normalize_numbers("一軒家に住む") == "一軒家に住む"
    assert tts_util._normalize_numbers("一軒屋で暮らす") == "一軒屋で暮らす"


def test_normalize_numbers_kanji_age_counter() -> None:
    text = "十二歳の頃"
    assert tts_util._normalize_numbers(text) == "じゅうにさいの頃"


def test_normalize_numbers_kanji_digit_list() -> None:
    text = "一、二席聞いて帰る"
    assert tts_util._normalize_numbers(text) == "一、にせき聞いて帰る"


def test_normalize_numbers_kanji_counter_age() -> None:
    text = "二十代半ば"
    assert tts_util._normalize_numbers(text) == "にじゅうだい半ば"


def test_normalize_numbers_kanji_counter_digit_seq() -> None:
    text = "一七五センチはある"
    assert tts_util._normalize_numbers(text) == "ひゃくななじゅうごせんちはある"


def test_normalize_numbers_person_readings() -> None:
    text = "一人で二人と4人"
    assert tts_util._normalize_numbers(text) == "ひとりでふたりとよにん"


def test_normalize_numbers_day_span() -> None:
    text = "二日間の旅"
    assert tts_util._normalize_numbers(text) == "ふつかかんの旅"


def test_normalize_numbers_day_span_single() -> None:
    text = "一日間滞在"
    assert tts_util._normalize_numbers(text) == "いちにちかん滞在"


def test_normalize_numbers_slash_fraction() -> None:
    text = "北の夕鶴2/3の殺人"
    assert tts_util._normalize_numbers(text) == "北の夕鶴さんぶんのにの殺人"


def test_normalize_numbers_bunno_fraction() -> None:
    assert tts_util._normalize_numbers("三分の二") == "さんぶんのに"
    assert tts_util._normalize_numbers("10分の1") == "じゅうぶんのいち"
    assert tts_util._normalize_numbers("十分の一") == "じゅうぶんのいち"


def test_normalize_numbers_slash_date_weekday() -> None:
    text = "2/3(日)開催"
    assert tts_util._normalize_numbers(text) == "にがつみっか(日)開催"


def test_normalize_numbers_slash_date_year() -> None:
    text = "2024/2/3"
    assert tts_util._normalize_numbers(text) == "二千二十四年にがつみっか"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # GOSICK 全９冊合本版 (角川文庫) - 桜庭 一樹.epub
        ("平成26年7月25日 発行", "平成二十六年しちがつにじゅうごにち 発行"),
        ("平成21年9月25日初版発行", "平成二十一年くがつにじゅうごにち初版発行"),
        ("GOSICK Ⅷ 上", "GOSICK はち 上"),
        ("GOSICK VIII 上", "GOSICK はち 上"),
        # 【合本版】俺の妹がこんなに可愛いわけがない...epub
        ("参加者１／２", "参加者にぶんのいち"),
        ("参加者２／３", "参加者さんぶんのに"),
        ("HG 1/144 スサノオ", "HG ひゃくよんじゅうよんぶんのいち スサノオ"),
        ("俺の妹がこんなに可愛いわけがない⑩", "俺の妹がこんなに可愛いわけがない十"),
        ("俺、和泉正宗／十五歳／高一。", "俺、和泉正宗/じゅうごさい/高一。"),
        # 【合本版】まよチキ！ 全12巻...epub
        ("【合本版】まよチキ！ 全12巻", "【合本版】まよチキ! 全じゅうにかん"),
        ("平日10:00～18:00まで", "平日じゅうじぜろふんからじゅうはちじぜろふんまで"),
        ("下心１００％だぜ", "下心百パーセントだぜ"),
        ("手術の成功率は五〇％。", "手術の成功率は五十パーセント。"),
        ("第５話 ダブルデートクライシス", "第ごわ ダブルデートクライシス"),
        ("方法① 校舎裏に呼び出してみる", "方法一 校舎裏に呼び出してみる"),
        ("ver.1.0", "ver.一てんゼロ"),
        # 【合本版】さくら荘のペットな彼女 全13巻（電子特典付き）...epub
        ("【合本版】さくら荘のペットな彼女 全13冊（電子特典付き）", "【合本版】さくら荘のペットな彼女 全じゅうさんさつ(電子特典付き)"),
        # サクラダリセット（角川文庫）【全7冊 合本版】...epub
        ("サクラダリセット（角川文庫）【全7冊 合本版】", "サクラダリセット(角川文庫)【全ななさつ 合本版】"),
        ("１ 八月一三日（金曜日）──スタート地点", "いち はちがつじゅうさんにち(金曜日)──スタート地点"),
        ("受付時間 9：00～17：00（土日 祝日 年末年始を除く）", "受付時間 きゅうじぜろふんからじゅうななじぜろふん(土日 祝日 年末年始を除く)"),
        # Ｖシリーズ全１０冊合本版 - 森博嗣.epub
        ("Ｖシリーズ全１０冊合本版", "Vシリーズ全じゅっさつ合本版"),
        ("0.9×0.8＝0.72", "零てんきゅう×零てんはち=零てんななに"),
    ],
)
def test_normalize_numbers_real_dataset_cases(text: str, expected: str) -> None:
    assert tts_util._normalize_numbers(text) == expected


def test_normalize_numbers_real_dataset_hyphen_address_not_date() -> None:
    # Real data appears in publication/address lines and should not be treated as Y/M/D.
    text = "東京都千代田区富士見2-13-3"
    out = tts_util._normalize_numbers(text)
    assert "千代田区" in out
    assert "年" not in out and "月" not in out and "日" not in out


def test_normalize_numbers_roman_numeral_to_kana() -> None:
    assert tts_util._normalize_numbers("GOSICK Ⅷ 上") == "GOSICK はち 上"


def test_normalize_numbers_ascii_roman_to_kana() -> None:
    assert tts_util._normalize_numbers("GOSICK VIII 上") == "GOSICK はち 上"
    assert tts_util._normalize_numbers("滅義怒羅怨II。") == "滅義怒羅怨に。"


def test_normalize_numbers_ascii_roman_acronyms_not_misconverted() -> None:
    assert tts_util._normalize_numbers("ドラマCD") == "ドラマCD"
    assert tts_util._normalize_numbers("TV CM") == "TV CM"
    assert tts_util._normalize_numbers("情報交換ML") == "情報交換ML"
    assert tts_util._normalize_numbers("東京MX") == "東京MX"


def test_prepare_tts_text_strips_japanese_quotes() -> None:
    assert (
        tts_util.prepare_tts_text("「聖書」『旧約』《新約》“Test” 'OK'〝注〟don't")
        == "聖書、旧約、新約Test OK注don't"
    )


def test_prepare_tts_text_preserves_japanese_internal_single_quote() -> None:
    assert tts_util.prepare_tts_text("りこん'で") == "りこん'で"


def test_prepare_tts_text_normalizes_width_and_format_chars() -> None:
    assert tts_util.prepare_tts_text("ＡＢＣ　１２３") == "ABC 123"
    assert tts_util.prepare_tts_text("a\u200bb") == "ab"


def test_prepare_tts_text_japanese_space_pause() -> None:
    assert (
        tts_util.prepare_tts_text("ウブメのナツ キョウゴクナツヒコ")
        == "ウブメのナツ。キョウゴクナツヒコ"
    )
    assert (
        tts_util.prepare_tts_text("Hello world 日本語 テスト")
        == "Hello world 日本語、テスト"
    )


def test_prepare_tts_text_normalizes_dash_runs() -> None:
    assert tts_util.prepare_tts_text("姑獲烏────") == "姑獲烏"
    assert tts_util.prepare_tts_text("前──後") == "前、後"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Real lines from: 【合本版】さくら荘のペットな彼女 全13巻（電子特典付き）
        ("「こまち、これが新幹線だぞ。早いだろ〜」", "こまち、これが新幹線だぞ。早いだろー"),
        ("「あ〜ん」", "あーん"),
        ("不機嫌な声を出した七海が、目をす〜っと細めて睨んでくる。", "不機嫌な声を出した七海が、目をすーっと細めて睨んでくる。"),
        ("「食事にする？ お風呂？ それとも〜、ま・わ・し？」", "食事にする? お風呂? それともー、ま・わ・し?"),
    ],
)
def test_prepare_tts_text_wave_dash_real_dataset_cases(text: str, expected: str) -> None:
    assert tts_util.prepare_tts_text(text) == expected


def test_prepare_tts_text_quote_boundary_preserves_question() -> None:
    assert tts_util.prepare_tts_text("「え？」と") == "え?と"


def test_prepare_tts_text_adds_short_tail_punct() -> None:
    assert tts_util.prepare_tts_text("まえがき", add_short_punct=True) == "まえがき。"
    assert tts_util.prepare_tts_text("前書き。", add_short_punct=True) == "前書き。"
    assert tts_util.prepare_tts_text("Prologue", add_short_punct=True) == "Prologue"
    long_text = "あ" * (tts_util._SHORT_TAIL_MAX_CHARS + 1)
    assert tts_util.prepare_tts_text(long_text, add_short_punct=True) == long_text


def test_load_reading_overrides(tmp_path: Path) -> None:
    payload = {
        "global": [
            {"base": "妻子", "reading": "さいし"},
            {"pattern": "御(存知)", "reading": r"ご\1"},
        ],
        "chapters": {
            "0001-test": {
                "replacements": [{"base": "漢字", "reading": "かんじ"}],
                "conflicts": [{"base": "生", "readings": ["せい", "なま"]}],
            }
        }
    }
    path = tmp_path / "reading-overrides.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    global_overrides, overrides = tts_util._load_reading_overrides(
        tmp_path, include_template=False
    )
    assert global_overrides == [
        {"base": "妻子", "reading": "さいし"},
        {"pattern": "御(存知)", "reading": r"ご\1"},
    ]
    assert overrides == {"0001-test": [{"base": "漢字", "reading": "かんじ"}]}


def test_ruby_global_overrides_supports_regex_entries() -> None:
    ruby_data = {
        "global": [
            {"base": "暗殺者", "reading": "あんさつしや"},
            {"base": "巻", "reading": "ま"},
            {"base": "上手", "pattern": "上手(?=い)", "reading": "うま"},
        ]
    }
    overrides = tts_util._ruby_global_overrides(ruby_data)
    assert {"base": "暗殺者", "reading": "あんさつしや"} in overrides
    assert {"pattern": "上手(?=い)", "reading": "うま"} in overrides
    assert {"base": "巻", "reading": "ま"} not in overrides


def test_merge_reading_overrides_prefers_chapter() -> None:
    global_overrides = [{"base": "山田太一", "reading": "やまだたいち"}]
    chapter_overrides = [{"base": "山田太一", "reading": "やまだたいいち"}]
    merged = tts_util._merge_reading_overrides(global_overrides, chapter_overrides)
    assert tts_util.apply_reading_overrides("山田太一", merged) == "やまだたいいち"


def test_load_reading_overrides_includes_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template_path = tmp_path / "template.md"
    template_path.write_text("# header\n妻子＝さいし\n", encoding="utf-8")
    monkeypatch.setattr(
        tts_util, "_template_reading_overrides_path", lambda: template_path
    )
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    (book_dir / "reading-overrides.json").write_text(
        json.dumps({"global": [{"base": "妻子", "reading": "つまこ"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    global_overrides, overrides = tts_util._load_reading_overrides(book_dir)
    assert overrides == {}
    assert tts_util.apply_reading_overrides("妻子", global_overrides) == "つまこ"


def test_select_backend_prefers_mlx_on_apple(monkeypatch) -> None:
    monkeypatch.setattr(tts_util, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(tts_util, "_has_mlx_audio", lambda: True)
    assert tts_util._select_backend("auto") == "mlx"


def test_select_backend_falls_back_to_torch(monkeypatch) -> None:
    monkeypatch.setattr(tts_util, "_is_apple_silicon", lambda: True)
    monkeypatch.setattr(tts_util, "_has_mlx_audio", lambda: False)
    with pytest.raises(ValueError):
        tts_util._select_backend("auto")


def test_normalize_lang_code() -> None:
    assert tts_util._normalize_lang_code("Japanese") == "ja"
    assert tts_util._normalize_lang_code("ja") == "ja"
    assert tts_util._normalize_lang_code("EN") == "en"
    assert tts_util._normalize_lang_code("zh-cn") == "zh"
    assert tts_util._normalize_lang_code("") is None


def test_generate_audio_mlx_passes_language_name() -> None:
    class DummyModel:
        def __init__(self) -> None:
            self.kwargs = None

        def generate(self, text, language=None, lang_code=None, voice=None, **kwargs):
            self.kwargs = {
                "language": language,
                "lang_code": lang_code,
                "voice": voice,
            }
            return []

    model = DummyModel()
    config = voice_util.VoiceConfig(
        name="jp",
        ref_audio="voice.wav",
        ref_text="こんにちは",
        language="ja",
    )
    audio, rate = tts_util._generate_audio_mlx(model, "テスト", config)
    assert audio == []
    assert rate == 24000
    assert model.kwargs["language"] == "Japanese"
    assert model.kwargs["lang_code"] is None


def test_generate_audio_falls_back_to_generate() -> None:
    class ModelGenerate:
        def __init__(self) -> None:
            self.kwargs = None

        def generate(self, text, prompt=None):
            self.kwargs = {"text": text, "prompt": prompt}
            return [tts_util.np.zeros(1)], 24000

    model = ModelGenerate()
    wavs, rate = tts_util._generate_audio(
        model=model, text="こんにちは", prompt={"p": 2}, language="Japanese"
    )
    assert rate == 24000
    assert model.kwargs["text"] == "こんにちは"
    assert model.kwargs["prompt"] == {"p": 2}


def test_ensure_pad_token_id_sets_pad_on_configs() -> None:
    class GenCfg:
        def __init__(self) -> None:
            self.eos_token_id = 2150
            self.pad_token_id = None

    class Cfg:
        def __init__(self) -> None:
            self.eos_token_id = 2150
            self.pad_token_id = None

    class Core:
        def __init__(self) -> None:
            self.generation_config = GenCfg()
            self.config = Cfg()

    class Model:
        def __init__(self) -> None:
            self.model = Core()

    model = Model()
    tts_util._ensure_pad_token_id(model)
    assert model.model.generation_config.pad_token_id == 2150
    assert model.model.config.pad_token_id == 2150
