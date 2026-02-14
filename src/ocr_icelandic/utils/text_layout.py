"""Text wrapping and column layout utilities."""

from dataclasses import dataclass

import pyphen
from PIL import ImageDraw, ImageFont

from ocr_icelandic.logging_config import get_logger

# Module-level hyphenator cache (lazy-initialized, keyed by language)
_hyphenators: dict[str, pyphen.Pyphen] = {}


def _get_hyphenator(lang: str = "is") -> pyphen.Pyphen:
    """Get or create a hyphenator for the given language."""
    if lang not in _hyphenators:
        _hyphenators[lang] = pyphen.Pyphen(lang=lang)
    return _hyphenators[lang]


logger = get_logger(__name__)


@dataclass
class ParagraphFontConfig:
    """Font configuration for a single paragraph."""

    font_path: str
    font_size: int
    bold: bool = False
    underline: bool = False

    def get_cache_key(self) -> tuple:
        """Cache key for font loading (excludes underline since it's drawn separately)."""
        return (self.font_path, self.font_size, self.bold)


@dataclass
class WrappedParagraph:
    """Represents a paragraph after text wrapping."""

    lines: list[str]
    text: str
    has_text: bool
    font_config: "ParagraphFontConfig | None" = None


@dataclass
class WrapResult:
    """Result of text wrapping operation."""

    paragraphs: list[WrappedParagraph]
    has_overflow: bool


@dataclass
class LinePlacement:
    """Placement information for a single line of text."""

    text: str
    paragraph_index: int | None
    column_index: int
    line_index: int
    is_blank: bool


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    tab_width: int = 4,
    hyphenation_lang: str = "is",
) -> WrapResult:
    """Wrap each paragraph to fit within the given width.

    Words that don't fit on a line are hyphenated using pyphen before
    falling back to a hard break.

    Args:
        draw: ImageDraw instance for measuring text
        text: Text to wrap
        font: Font to use for text measurement
        max_width: Maximum width in pixels
        tab_width: Number of spaces to replace tabs with
        hyphenation_lang: ISO 639-1 language code for hyphenation (default: "is")

    Returns:
        WrapResult containing wrapped paragraphs and overflow flag
    """
    logger.debug(
        "Wrapping text to fit within %d pixels (tab_width=%d)", max_width, tab_width
    )
    paragraphs = text.split("\n")
    wrapped_paragraphs: list[WrappedParagraph] = []
    has_overflow = False
    total_wrapped_lines = 0

    for para_idx, paragraph in enumerate(paragraphs):
        stripped_paragraph = paragraph.strip()
        if not stripped_paragraph:
            logger.debug("Paragraph %d is empty", para_idx)
            wrapped_paragraphs.append(
                WrappedParagraph(lines=[], text="", has_text=False)
            )
            continue

        logger.debug(
            "Wrapping paragraph %d with %d characters",
            para_idx,
            len(stripped_paragraph),
        )
        leading_whitespace = ""
        left_stripped = paragraph.lstrip()
        if len(paragraph) > len(left_stripped):
            leading_whitespace = paragraph[: len(paragraph) - len(left_stripped)]
            leading_whitespace = leading_whitespace.replace("\t", " " * tab_width)

        left_stripped = left_stripped.replace("\t", " " * tab_width)
        words = left_stripped.split()
        paragraph_lines: list[str] = []
        current_line: list[str] = []
        is_first_line = True

        hyp = _get_hyphenator(hyphenation_lang)

        for word in words:
            test_line_base = " ".join(current_line + [word])
            test_line = (
                leading_whitespace + test_line_base if is_first_line else test_line_base
            )
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]

            if test_width <= max_width:
                current_line.append(word)
            else:
                # Word doesn't fit on current line — try hyphenation
                placed_via_hyphen = False
                if current_line:
                    # Try to split the word and keep part on this line
                    pairs = hyp.iterate(word)
                    for head, tail in pairs:
                        candidate = " ".join(current_line + [head + "-"])
                        candidate_line = (
                            leading_whitespace + candidate
                            if is_first_line
                            else candidate
                        )
                        bbox = draw.textbbox((0, 0), candidate_line, font=font)
                        if bbox[2] - bbox[0] <= max_width:
                            # Head fits — commit this line and carry tail forward
                            paragraph_lines.append(
                                (leading_whitespace if is_first_line else "")
                                + " ".join(current_line + [head + "-"])
                            )
                            is_first_line = False
                            current_line = [tail]
                            placed_via_hyphen = True
                            logger.debug(
                                "Hyphenated '%s' -> '%s-' | '%s'",
                                word,
                                head,
                                tail,
                            )
                            break

                if not placed_via_hyphen:
                    # No hyphenation fit — flush current line, start new one
                    if current_line:
                        paragraph_lines.append(
                            (leading_whitespace if is_first_line else "")
                            + " ".join(current_line)
                        )
                        is_first_line = False
                    current_line = [word]
                    test_line_base = " ".join(current_line)
                    test_line = (
                        leading_whitespace + test_line_base
                        if is_first_line
                        else test_line_base
                    )
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    if bbox[2] - bbox[0] > max_width:
                        # Word alone is too wide — try hyphenating it across lines
                        pairs = list(hyp.iterate(word))
                        if pairs:
                            # Find the longest head that fits
                            for head, tail in pairs:
                                head_line = (
                                    (leading_whitespace if is_first_line else "")
                                    + head
                                    + "-"
                                )
                                bbox = draw.textbbox((0, 0), head_line, font=font)
                                if bbox[2] - bbox[0] <= max_width:
                                    paragraph_lines.append(head_line)
                                    is_first_line = False
                                    current_line = [tail]
                                    logger.debug(
                                        "Hyphenated long word '%s' -> '%s-' | '%s'",
                                        word,
                                        head,
                                        tail,
                                    )
                                    break
                            else:
                                # No hyphenation point fits either
                                logger.debug(
                                    "Word '%s' exceeds max width, marking overflow",
                                    word,
                                )
                                has_overflow = True
                                paragraph_lines.append(
                                    (leading_whitespace if is_first_line else "") + word
                                )
                                is_first_line = False
                                current_line = []
                        else:
                            # No hyphenation points available
                            logger.debug(
                                "Word '%s' exceeds max width (%d > %d), marking overflow",
                                word,
                                bbox[2] - bbox[0],
                                max_width,
                            )
                            has_overflow = True
                            paragraph_lines.append(
                                (leading_whitespace if is_first_line else "") + word
                            )
                            is_first_line = False
                            current_line = []

        if current_line:
            paragraph_lines.append(
                (leading_whitespace if is_first_line else "") + " ".join(current_line)
            )

        logger.debug(
            "Paragraph %d wrapped into %d lines", para_idx, len(paragraph_lines)
        )
        total_wrapped_lines += len(paragraph_lines)
        wrapped_paragraphs.append(
            WrappedParagraph(
                lines=paragraph_lines, text=stripped_paragraph, has_text=True
            )
        )

    logger.debug(
        "Text wrapping complete: %d total lines, overflow=%s",
        total_wrapped_lines,
        has_overflow,
    )
    return WrapResult(paragraphs=wrapped_paragraphs, has_overflow=has_overflow)


def arrange_lines_in_columns(
    paragraphs: list[WrappedParagraph],
    max_lines_per_column: int,
    num_columns: int,
) -> tuple[list[LinePlacement], list[int]]:
    """Arrange wrapped paragraphs into columns.

    Args:
        paragraphs: List of wrapped paragraphs
        max_lines_per_column: Maximum lines that fit in a column
        num_columns: Number of columns

    Returns:
        Tuple of (line placements, column line counts)
    """
    logger.debug(
        "Arranging %d paragraphs into %d columns (max %d lines per column)",
        len(paragraphs),
        num_columns,
        max_lines_per_column,
    )
    placements: list[LinePlacement] = []
    column_counts = [0] * num_columns
    current_column = 0

    def advance_column() -> None:
        nonlocal current_column
        while (
            current_column < num_columns
            and column_counts[current_column] >= max_lines_per_column
        ):
            current_column += 1

    def add_line(text: str, paragraph_index: int | None, is_blank: bool) -> bool:
        nonlocal current_column
        advance_column()
        if current_column >= num_columns:
            logger.debug(
                "Column overflow: current_column=%d >= num_columns=%d",
                current_column,
                num_columns,
            )
            return False
        placements.append(
            LinePlacement(
                text=text,
                paragraph_index=paragraph_index,
                column_index=current_column,
                line_index=column_counts[current_column],
                is_blank=is_blank,
            )
        )
        column_counts[current_column] += 1
        return True

    lines_added = 0
    for idx, paragraph in enumerate(paragraphs):
        if paragraph.has_text:
            for line in paragraph.lines:
                if not add_line(line, idx, is_blank=False):
                    logger.debug("Ran out of column space at paragraph %d", idx)
                    return placements, column_counts
                lines_added += 1
            if idx < len(paragraphs) - 1:
                if not add_line("", None, is_blank=True):
                    logger.debug(
                        "Ran out of column space after paragraph %d blank line", idx
                    )
                    return placements, column_counts
                lines_added += 1
        else:
            if not add_line("", None, is_blank=True):
                logger.debug("Ran out of column space at empty paragraph %d", idx)
                return placements, column_counts
            lines_added += 1

    logger.debug(
        "Column arrangement complete: %d total lines placed, distribution: %s",
        lines_added,
        column_counts,
    )
    return placements, column_counts


def calculate_paragraph_font_sizes(
    paragraph_texts: list[str],
    base_font_size: int,
    min_ratio: float = 0.8,
    max_ratio: float = 1.2,
) -> list[int]:
    """
    Calculate font sizes for paragraphs using inverse scaling.

    Shorter paragraphs get larger fonts, longer paragraphs get smaller fonts,
    with limited variation to avoid extreme differences.

    Args:
        paragraph_texts: List of paragraph text strings
        base_font_size: Base font size
        min_ratio: Minimum size ratio (e.g., 0.8 = 80% of base)
        max_ratio: Maximum size ratio (e.g., 1.2 = 120% of base)

    Returns:
        List of font sizes (one per paragraph)
    """
    logger.debug(
        "Calculating paragraph font sizes: base=%d, min_ratio=%.2f, max_ratio=%.2f",
        base_font_size,
        min_ratio,
        max_ratio,
    )

    if not paragraph_texts:
        logger.debug("No paragraphs provided, returning empty list")
        return []

    # Count characters per paragraph
    char_counts = [len(text) for text in paragraph_texts]
    logger.debug("Character counts per paragraph: %s", char_counts)

    if max(char_counts) == min(char_counts):
        # All same length, return base size for all
        logger.debug("All paragraphs have same length, using base size for all")
        return [base_font_size] * len(paragraph_texts)

    # Normalize character counts to 0-1 range
    min_count = min(char_counts)
    max_count = max(char_counts)
    normalized = [(c - min_count) / (max_count - min_count) for c in char_counts]
    logger.debug("Normalized character counts: %s", [f"{n:.2f}" for n in normalized])

    # Inverse scaling: smaller normalized value = larger font
    # Map [0, 1] to [max_ratio, min_ratio]
    font_sizes = []
    for norm_val in normalized:
        ratio = max_ratio - (norm_val * (max_ratio - min_ratio))
        font_size = int(base_font_size * ratio)
        font_sizes.append(font_size)

    loglev = logger.level
    logger.debug("Calculated font sizes: %s", font_sizes)
    logger.setLevel(loglev)
    return font_sizes
