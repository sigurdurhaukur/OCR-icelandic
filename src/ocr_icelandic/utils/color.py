"""Color utilities for image generation."""

import numpy as np
from PIL import Image

from ocr_icelandic.logging_config import get_logger

logger = get_logger(__name__)


def color_to_rgb(color: str | tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert a color string or tuple to RGB tuple."""
    if isinstance(color, tuple):
        logger.debug("Converting tuple color to RGB: %s", color[:3])
        return color[:3]
    logger.debug("Converting string color '%s' to RGB tuple", color)
    temp = Image.new("RGB", (1, 1), color)
    rgb = temp.getpixel((0, 0))
    logger.debug("Converted string color '%s' to RGB: %s", color, rgb)
    return rgb


def calculate_luminance(color: tuple[int, int, int]) -> float:
    """
    Calculate relative luminance per WCAG 2.1 specification.

    Returns luminance in 0.0-1.0 range (gamma-corrected).
    """
    r, g, b = color
    logger.debug("Calculating luminance for RGB color: (%d, %d, %d)", r, g, b)
    # Normalize to 0-1 range
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    # Apply gamma correction
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    logger.debug("Calculated luminance: %.4f", luminance)
    return luminance


def is_grayscale(color: tuple[int, int, int], threshold: int = 30) -> bool:
    """Check if a color is grayscale (low saturation)."""
    r, g, b = color
    avg = (r + g + b) / 3
    is_gray = all(abs(c - avg) < threshold for c in [r, g, b])
    logger.debug(
        "Checked if RGB(%d, %d, %d) is grayscale (threshold=%d): %s",
        r,
        g,
        b,
        threshold,
        is_gray,
    )
    return is_gray


def get_blend_mode(
    font_color: str | tuple[int, int, int],
    bg_color: str | tuple[int, int, int],
) -> str:
    """
    Determine the appropriate blending mode based on font and background colors.

    Automatically selects the blend mode to ensure text visibility and realistic
    paper texture appearance:
    - multiply: Dark text on light background
    - screen: Light text on dark background
    - normal: Colored text or mid-tone combinations

    Args:
        font_color: Color of the text
        bg_color: Background color

    Returns:
        Blending mode: "multiply", "screen", or "normal"
    """
    font_rgb = color_to_rgb(font_color)
    bg_rgb = color_to_rgb(bg_color)

    # Calculate WCAG 2.1 relative luminance (0.0-1.0 range)
    font_lum = calculate_luminance(font_rgb)
    bg_lum = calculate_luminance(bg_rgb)

    # Luminance thresholds (WCAG 2.1 scale: 0.0-1.0)
    dark_threshold = 0.18  # Roughly equivalent to RGB(137, 137, 137)
    light_threshold = 0.50  # Roughly equivalent to RGB(188, 188, 188)

    # Check if colors are grayscale (with relaxed threshold for nearly-gray colors)
    font_is_gray = is_grayscale(font_rgb, threshold=40)

    # Calculate luminance difference for determining contrast
    lum_diff = abs(font_lum - bg_lum)

    # Dark text on light background -> multiply
    # Conditions: background is light, font is darker than background, mostly grayscale
    if (
        bg_lum >= light_threshold
        and font_lum < bg_lum
        and lum_diff > 0.2
        and font_is_gray
    ):
        blend_mode = "multiply"
        logger.debug(
            "Selected 'multiply' blend mode for dark text on light background (font_lum=%.3f, bg_lum=%.3f)",
            font_lum,
            bg_lum,
        )
        return blend_mode

    # Light text on dark background -> screen
    # Conditions: background is dark, font is lighter than background, mostly grayscale
    elif (
        bg_lum <= dark_threshold
        and font_lum > bg_lum
        and lum_diff > 0.2
        and font_is_gray
    ):
        blend_mode = "screen"
        logger.debug(
            "Selected 'screen' blend mode for light text on dark background (font_lum=%.3f, bg_lum=%.3f)",
            font_lum,
            bg_lum,
        )
        return blend_mode

    # Colored text or insufficient contrast -> normal alpha compositing
    else:
        blend_mode = "normal"
        logger.debug(
            "Using 'normal' blend mode: insufficient contrast or colored text (font_lum=%.3f, bg_lum=%.3f, diff=%.3f)",
            font_lum,
            bg_lum,
            lum_diff,
        )
        return blend_mode


def blend_text_layer(
    background: Image.Image,
    text_mask: Image.Image,
    font_color: tuple[int, int, int],
    blend_mode: str,
) -> Image.Image:
    """
    Blend a text layer onto a background using the specified blending mode.

    This function applies text with paper texture showing through, simulating
    realistic printed or handwritten text on textured paper.

    Args:
        background: The background image (with paper texture)
        text_mask: Grayscale mask where text is white (255) and non-text is black (0)
        font_color: RGB tuple of the font color
        blend_mode: "multiply", "screen", or "normal"
            - multiply: Dark text on light background (paper texture shines through)
            - screen: Light text on dark background (paper texture shines through)
            - normal: Standard alpha compositing for colored text

    Returns:
        Blended image with text showing paper texture through it
    """
    logger.debug(
        "Blending text layer using '%s' mode with color RGB(%d, %d, %d)",
        blend_mode,
        font_color[0],
        font_color[1],
        font_color[2],
    )
    # Convert to numpy for blending calculations
    bg_array = np.array(background, dtype=np.float32)
    mask_array = np.array(text_mask, dtype=np.float32) / 255.0

    if blend_mode == "multiply":
        logger.debug("Applying multiply blend: dark text on light background")
        # Multiply blend: result = (A * B) / 255
        # For dark text on light background
        # Create a layer where text areas have font_color and non-text areas are white (neutral for multiply)
        text_array = np.ones_like(bg_array) * 255.0
        for c in range(3):
            text_array[:, :, c] = font_color[c] * mask_array + 255.0 * (1 - mask_array)

        # Apply multiply blend
        result = (bg_array * text_array) / 255.0

    elif blend_mode == "screen":
        logger.debug("Applying screen blend: light text on dark background")
        # Screen blend: result = 255 - ((255 - A) * (255 - B)) / 255
        # For light text on dark background
        # Create a layer where text areas have font_color and non-text areas are black (neutral for screen)
        text_array = np.zeros_like(bg_array)
        for c in range(3):
            text_array[:, :, c] = font_color[c] * mask_array

        # Apply screen blend
        result = 255.0 - ((255.0 - bg_array) * (255.0 - text_array)) / 255.0

    else:  # normal
        logger.debug("Applying normal alpha compositing blend")
        # Normal blend: standard alpha compositing
        # Text replaces background where mask is white
        result = bg_array.copy()
        for c in range(3):
            result[:, :, c] = font_color[c] * mask_array + bg_array[:, :, c] * (
                1 - mask_array
            )

    # Clip to valid range and convert back to PIL
    result = np.clip(result, 0, 255).astype(np.uint8)
    logger.debug("Text layer blended successfully")
    return Image.fromarray(result, mode="RGB")
