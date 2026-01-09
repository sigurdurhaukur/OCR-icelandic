"""Font loading utilities."""

from PIL import ImageFont


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
    try:
        return ImageFont.truetype(font_path, font_size)
    except OSError:
        return ImageFont.load_default()
