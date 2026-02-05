import json
from pathlib import Path

import pytest

from nik import tts as tts_util
from nik import voice as voice_util


def test_make_chunk_spans_splits_japanese_sentences() -> None:
    text = "今日は良い天気です。明日も晴れるでしょう。"
    spans = tts_util.make_chunk_spans(text, max_chars=10, chunk_mode="japanese")
    assert len(spans) >= 2


def test_chunking_splits_on_commas_and_linebreaks() -> None:
    text = "私は、猫です。\n次の行です。"
    spans = tts_util.make_chunk_spans(text, max_chars=100, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert chunks == ["私は、猫です。", "次の行です。"]


def test_chunking_splits_on_commas_when_too_long() -> None:
    text = "これはとても長い文章なので、途中で区切る必要があります。"
    spans = tts_util.make_chunk_spans(text, max_chars=15, chunk_mode="japanese")
    chunks = [text[start:end] for start, end in spans]
    assert any("、" in chunk for chunk in chunks)


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
    text = "束の間は間違い。"
    overrides = [{"base": "間", "reading": "ま", "mode": "isolated"}]
    assert tts_util.apply_reading_overrides(text, overrides) == "束のまは間違い。"


def test_apply_ruby_spans() -> None:
    text = "前漢字後"
    spans = [{"start": 1, "end": 3, "base": "漢字", "reading": "かんじ"}]
    assert tts_util.apply_ruby_spans(text, spans) == "前かんじ後"


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
    )
    assert out == "カンジ始まる"


def test_normalize_kana_with_stub_tagger_partial_common_noun() -> None:
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

    out = tts_util._normalize_kana_with_tagger(
        "自分",
        DummyTagger(),
        kana_style="partial",
        zh_lexicon=set(),
    )
    assert out == "自分"


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
    )
    assert out == "サイシ"

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
        "漢字", DummyTagger(), kana_style="partial", zh_lexicon={"漢字"}
    )
    assert out == "カンジ"


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
    assert out == "トオカ"


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


def test_normalize_numbers_standalone_digits() -> None:
    text = "全7冊 合本版"
    assert tts_util._normalize_numbers(text) == "全七冊 合本版"


def test_normalize_numbers_date() -> None:
    text = "2024年7月1日"
    assert tts_util._normalize_numbers(text) == "二千二十四年しちがつついたち"


def test_normalize_numbers_decimal() -> None:
    text = "3.14"
    assert tts_util._normalize_numbers(text) == "三てんいちよん"


def test_normalize_numbers_fallback_cardinal() -> None:
    text = "合計13です"
    assert tts_util._normalize_numbers(text) == "合計十三です"


def test_normalize_numbers_counter_wa() -> None:
    text = "１話 土曜日"
    assert tts_util._normalize_numbers(text) == "いちわ 土曜日"


def test_prepare_tts_text_strips_japanese_quotes() -> None:
    assert (
        tts_util.prepare_tts_text("「聖書」『旧約』《新約》“Test” 'OK'〝注〟don't")
        == "聖書、旧約、新約Test OK注don't"
    )


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
        "global": [{"base": "妻子", "reading": "さいし"}],
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
    assert global_overrides == [{"base": "妻子", "reading": "さいし"}]
    assert overrides == {"0001-test": [{"base": "漢字", "reading": "かんじ"}]}


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


def test_generate_audio_mlx_passes_lang_code() -> None:
    class DummyModel:
        def __init__(self) -> None:
            self.kwargs = None

        def generate(self, text, lang_code=None, voice=None, **kwargs):
            self.kwargs = {"lang_code": lang_code, "voice": voice}
            return []

    model = DummyModel()
    config = voice_util.VoiceConfig(
        name="jp",
        ref_audio="voice.wav",
        ref_text="こんにちは",
        language="Japanese",
    )
    audio, rate = tts_util._generate_audio_mlx(model, "テスト", config)
    assert audio == []
    assert rate == 24000
    assert model.kwargs["lang_code"] == "ja"


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
