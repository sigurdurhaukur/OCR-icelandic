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
