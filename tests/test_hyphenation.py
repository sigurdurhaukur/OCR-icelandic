"""Tests for word hyphenation during text wrapping."""

import pytest
from PIL import Image, ImageDraw, ImageFont

from ocr_icelandic.utils.text_layout import wrap_text


@pytest.fixture
def draw():
    """Provide an ImageDraw instance for text measurement."""
    img = Image.new("RGB", (800, 800))
    return ImageDraw.Draw(img)


@pytest.fixture
def font():
    """Provide a default font for testing."""
    return ImageFont.load_default(size=14)


def _all_lines(result):
    """Extract all lines from a WrapResult."""
    lines = []
    for p in result.paragraphs:
        lines.extend(p.lines)
    return lines


class TestBasicHyphenation:
    """Test that words are hyphenated when they don't fit on the current line."""

    def test_long_icelandic_word_is_hyphenated(self, draw, font):
        """A long Icelandic compound word should be split with a hyphen."""
        # "Borgarstjórnarráðherra" is a long compound word
        result = wrap_text(
            draw,
            "Stutt orð Borgarstjórnarráðherra",
            font,
            max_width=140,
            hyphenation_lang="is",
        )
        lines = _all_lines(result)
        # At least one line should end with a hyphen (the word was split)
        hyphenated = [ln for ln in lines if ln.endswith("-")]
        assert len(hyphenated) >= 1, (
            f"Expected at least one hyphenated line, got: {lines}"
        )
        assert not result.has_overflow

    def test_multiple_long_words(self, draw, font):
        """Multiple long words in sequence should each be hyphenated as needed."""
        result = wrap_text(
            draw,
            "samfélagsmálaráðherra Borgarstjórnarráðherra",
            font,
            max_width=120,
            hyphenation_lang="is",
        )
        lines = _all_lines(result)
        hyphenated = [ln for ln in lines if ln.endswith("-")]
        assert len(hyphenated) >= 1, (
            f"Expected hyphenation for long words, got: {lines}"
        )
        assert not result.has_overflow


class TestNoHyphenationNeeded:
    """Test that words that fit on a line are NOT hyphenated."""

    def test_short_words_no_hyphens(self, draw, font):
        """Short words that fit should not be hyphenated."""
        result = wrap_text(
            draw,
            "Þetta er stutt setning",
            font,
            max_width=400,
            hyphenation_lang="is",
        )
        lines = _all_lines(result)
        for line in lines:
            assert not line.endswith("-"), (
                f"Short text should not be hyphenated: {line}"
            )
        assert not result.has_overflow

    def test_single_word_fits(self, draw, font):
        """A single word that fits should not be hyphenated."""
        result = wrap_text(draw, "hestur", font, max_width=400, hyphenation_lang="is")
        lines = _all_lines(result)
        assert lines == ["hestur"]
        assert not result.has_overflow


class TestOverflowFallback:
    """Test that overflow is detected when hyphenation can't help."""

    def test_unhyphenatable_long_string(self, draw, font):
        """A string with no valid hyphenation points that's too wide should overflow."""
        # A single run of characters with no dictionary entry
        nonsense = "x" * 200
        result = wrap_text(draw, nonsense, font, max_width=80, hyphenation_lang="is")
        assert result.has_overflow, (
            "Should mark overflow for unhyphenatable word that exceeds max_width"
        )


class TestSingleWordTooWide:
    """Test hyphenation of a single word that exceeds the full column width."""

    def test_single_long_word_hyphenated_across_lines(self, draw, font):
        """A single long word wider than the column should be hyphenated."""
        result = wrap_text(
            draw,
            "Borgarstjórnarráðherra",
            font,
            max_width=100,
            hyphenation_lang="is",
        )
        lines = _all_lines(result)
        assert len(lines) >= 2, f"Word should be split across lines, got: {lines}"
        # First line(s) should end with hyphen
        assert lines[0].endswith("-"), f"First part should end with hyphen: {lines[0]}"
        assert not result.has_overflow


class TestLanguageParameter:
    """Test that the hyphenation_lang parameter affects behaviour."""

    def test_english_hyphenation(self, draw, font):
        """English hyphenation should work with lang='en'."""
        result = wrap_text(
            draw,
            "internationalization",
            font,
            max_width=90,
            hyphenation_lang="en",
        )
        lines = _all_lines(result)
        hyphenated = [ln for ln in lines if ln.endswith("-")]
        assert len(hyphenated) >= 1, (
            f"English word should be hyphenated at narrow width: {lines}"
        )
        assert not result.has_overflow

    def test_default_is_icelandic(self, draw, font):
        """Default hyphenation language should be Icelandic ('is')."""
        # Call without specifying hyphenation_lang — should default to "is"
        result = wrap_text(
            draw,
            "Borgarstjórnarráðherra",
            font,
            max_width=100,
        )
        lines = _all_lines(result)
        assert len(lines) >= 2, (
            f"Default lang should hyphenate Icelandic words: {lines}"
        )

    def test_different_languages_give_different_splits(self, draw, font):
        """The same word should potentially split differently under different languages."""
        # "international" is a word in both English and could be in other dictionaries
        result_en = wrap_text(
            draw, "international", font, max_width=80, hyphenation_lang="en"
        )
        result_de = wrap_text(
            draw, "international", font, max_width=80, hyphenation_lang="de"
        )
        lines_en = _all_lines(result_en)
        lines_de = _all_lines(result_de)
        # Both should produce lines (we mainly verify they don't crash)
        assert len(lines_en) >= 1
        assert len(lines_de) >= 1


class TestHyphenationWithCreateImage:
    """Integration test: hyphenation works through create_image_with_text."""

    def test_create_image_with_hyphenation_lang(self):
        """create_image_with_text should accept and use hyphenation_lang."""
        from ocr_icelandic.utils.image_creation import create_image_with_text

        # Should not raise
        image, fitted_text, bboxes = create_image_with_text(
            text="Borgarstjórnarráðherra sagði samfélagsmálaráðherra",
            image_size=(300, 200),
            font_size=14,
            num_columns=3,
            hyphenation_lang="is",
        )
        assert image is not None
        assert len(fitted_text) > 0
        assert isinstance(bboxes, list)
