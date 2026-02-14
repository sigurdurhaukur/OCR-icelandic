"""Font loading utilities."""

from PIL import ImageFont

from ocr_icelandic.logging_config import get_logger

logger = get_logger(__name__)


def load_font(
    font_path: str = "Arial.ttf",
    font_size: int = 20,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Load a TrueType font or default if not found.

    Args:
        font_path: Path to the .ttf font file
        font_size: Size of the font

    Returns:
        ImageFont.FreeTypeFont object
    """
    logger.debug("Attempting to load font from '%s' with size %d", font_path, font_size)
    try:
        font = ImageFont.truetype(font_path, font_size)
        logger.debug("Successfully loaded TrueType font: %s", font_path)
        return font
    except OSError as e:
        logger.warning("Failed to load font '%s': %s, using default font", font_path, e)
        return ImageFont.load_default()


def discover_font_variants(base_font_path: str) -> dict[str, str | None]:
    """
    Discover bold/italic variants of a font.

    Searches for variant files like:
    - FontName-Bold.ttf, FontNameBold.ttf
    - FontName-Italic.ttf, FontNameItalic.ttf
    - FontName-BoldItalic.ttf, FontNameBoldItalic.ttf

    Args:
        base_font_path: Path to the base font file

    Returns:
        Dict with 'bold', 'italic', 'bold_italic' keys (None if not found)
    """
    from pathlib import Path

    logger.debug("Discovering variants for font: %s", base_font_path)

    base_path = Path(base_font_path)
    base_name = base_path.stem  # e.g., "Arial"
    base_dir = base_path.parent

    variants = {"bold": None, "italic": None, "bold_italic": None}

    # Common naming patterns
    bold_patterns = [
        f"{base_name}-Bold.ttf",
        f"{base_name}Bold.ttf",
        f"{base_name}-bold.ttf",
    ]
    italic_patterns = [f"{base_name}-Italic.ttf", f"{base_name}Italic.ttf"]
    bold_italic_patterns = [f"{base_name}-BoldItalic.ttf", f"{base_name}BoldItalic.ttf"]

    # Search for variants
    for pattern in bold_patterns:
        variant_path = base_dir / pattern
        if variant_path.exists():
            variants["bold"] = str(variant_path)
            logger.debug("Found bold variant: %s", variant_path)
            break

    for pattern in italic_patterns:
        variant_path = base_dir / pattern
        if variant_path.exists():
            variants["italic"] = str(variant_path)
            logger.debug("Found italic variant: %s", variant_path)
            break

    for pattern in bold_italic_patterns:
        variant_path = base_dir / pattern
        if variant_path.exists():
            variants["bold_italic"] = str(variant_path)
            logger.debug("Found bold-italic variant: %s", variant_path)
            break

    return variants


def load_font_with_style(
    font_path: str,
    font_size: int,
    bold: bool = False,
    italic: bool = False,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, bool]:
    """
    Load font with style, using variant file if available.

    Args:
        font_path: Path to the base font file
        font_size: Size of the font
        bold: Whether to use bold style
        italic: Whether to use italic style

    Returns:
        Tuple of (font, needs_bold_simulation)
        - If needs_simulation=True, caller should draw text multiple times with offsets
    """
    logger.debug(
        "Loading font with style: path='%s', size=%d, bold=%s, italic=%s",
        font_path,
        font_size,
        bold,
        italic,
    )

    actual_path = font_path
    needs_simulation = False

    if bold or italic:
        variants = discover_font_variants(font_path)

        if bold and italic and variants["bold_italic"]:
            actual_path = variants["bold_italic"]
            logger.debug("Using bold-italic variant file")
        elif bold and variants["bold"]:
            actual_path = variants["bold"]
            logger.debug("Using bold variant file")
        elif italic and variants["italic"]:
            actual_path = variants["italic"]
            logger.debug("Using italic variant file")
        elif bold:
            # No bold variant found, will need simulation
            needs_simulation = True
            logger.debug("No bold variant found, will need simulation")

    font = load_font(actual_path, font_size)
    return font, needs_simulation
