"""Color utilities for image generation."""

import numpy as np
from PIL import Image


def color_to_rgb(color: str | tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert a color string or tuple to RGB tuple."""
    if isinstance(color, tuple):
        return color[:3]
    temp = Image.new("RGB", (1, 1), color)
    return temp.getpixel((0, 0))


def calculate_luminance(color: tuple[int, int, int]) -> float:
    """Calculate perceived luminance of a color (0-255 scale)."""
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]


def is_grayscale(color: tuple[int, int, int], threshold: int = 30) -> bool:
    """Check if a color is grayscale (low saturation)."""
    r, g, b = color
    avg = (r + g + b) / 3
    return all(abs(c - avg) < threshold for c in [r, g, b])


def get_blend_mode(
    font_color: str | tuple[int, int, int],
    bg_color: str | tuple[int, int, int],
) -> str:
    """
    Determine the appropriate blending mode based on font and background colors.

    Args:
        font_color: Color of the text
        bg_color: Background color

    Returns:
        Blending mode: "multiply", "screen", or "normal"
        - multiply: Dark text on light background (makes paper texture visible through text)
        - screen: Light text on dark background (makes paper texture visible through text)
        - normal: Colored text on colored background (standard alpha compositing)
    """
    font_rgb = color_to_rgb(font_color)
    bg_rgb = color_to_rgb(bg_color)

    font_lum = calculate_luminance(font_rgb)
    bg_lum = calculate_luminance(bg_rgb)

    # Thresholds for determining "dark" vs "light"
    dark_threshold = 100
    light_threshold = 155

    # Check if colors are grayscale
    font_is_gray = is_grayscale(font_rgb)

    # Dark text on light background -> multiply
    if font_lum < dark_threshold and bg_lum > light_threshold and font_is_gray:
        return "multiply"
    # Light text on dark background -> screen
    elif font_lum > light_threshold and bg_lum < dark_threshold and font_is_gray:
        return "screen"
    # Colored text on colored background -> normal
    else:
        return "normal"


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
    # Convert to numpy for blending calculations
    bg_array = np.array(background, dtype=np.float32)
    mask_array = np.array(text_mask, dtype=np.float32) / 255.0

    if blend_mode == "multiply":
        # Multiply blend: result = (A * B) / 255
        # For dark text on light background
        # Create a layer where text areas have font_color and non-text areas are white (neutral for multiply)
        text_array = np.ones_like(bg_array) * 255.0
        for c in range(3):
            text_array[:, :, c] = font_color[c] * mask_array + 255.0 * (1 - mask_array)

        # Apply multiply blend
        result = (bg_array * text_array) / 255.0

    elif blend_mode == "screen":
        # Screen blend: result = 255 - ((255 - A) * (255 - B)) / 255
        # For light text on dark background
        # Create a layer where text areas have font_color and non-text areas are black (neutral for screen)
        text_array = np.zeros_like(bg_array)
        for c in range(3):
            text_array[:, :, c] = font_color[c] * mask_array

        # Apply screen blend
        result = 255.0 - ((255.0 - bg_array) * (255.0 - text_array)) / 255.0

    else:  # normal
        # Normal blend: standard alpha compositing
        # Text replaces background where mask is white
        result = bg_array.copy()
        for c in range(3):
            result[:, :, c] = font_color[c] * mask_array + bg_array[:, :, c] * (
                1 - mask_array
            )

    # Clip to valid range and convert back to PIL
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")
