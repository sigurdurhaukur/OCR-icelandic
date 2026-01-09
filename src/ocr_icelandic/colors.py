"""Color utilities for synthetic OCR image generation."""

import random

from PIL import ImageColor


def get_random_background_color() -> tuple[int, int, int]:
    """
    Generate a random background color with weighted distribution.

    Distribution: 85% light (paper-like), 10% dark, 5% colorful

    Returns:
        RGB color tuple
    """
    rand_val = random.random()

    if rand_val < 0.85:
        # Light colors (paper-like)
        paper_type = random.choice(["white", "cream", "aged"])
        r, g, b = _generate_paper_color(paper_type)
    elif rand_val < 0.95:
        # Dark colors
        r, g, b = _generate_dark_color()
    else:
        # Colorful
        r, g, b = _generate_colorful()

    return (_clamp(r), _clamp(g), _clamp(b))


def _generate_paper_color(paper_type: str) -> tuple[int, int, int]:
    """Generate paper-like colors."""
    if paper_type == "white":
        base = random.randint(245, 252)
        return (
            base + random.randint(-3, 3),
            base + random.randint(-5, 0),
            base + random.randint(-8, 0),
        )
    elif paper_type == "cream":
        base = random.randint(235, 245)
        return (
            base + random.randint(0, 8),
            base + random.randint(-5, 3),
            base + random.randint(-12, -3),
        )
    else:  # aged
        base = random.randint(220, 235)
        return (
            base + random.randint(5, 15),
            base + random.randint(0, 10),
            base + random.randint(-15, -5),
        )


def _generate_dark_color() -> tuple[int, int, int]:
    """Generate dark background colors."""
    base = random.randint(20, 80)
    return (
        base + random.randint(-10, 10),
        base + random.randint(-10, 10),
        base + random.randint(-10, 10),
    )


def _generate_colorful() -> tuple[int, int, int]:
    """Generate colorful background."""
    bright_channel = random.randint(0, 2)
    colors = [random.randint(30, 220) for _ in range(3)]
    colors[bright_channel] = random.randint(150, 255)
    return tuple(colors)


def _clamp(value: int, min_val: int = 0, max_val: int = 255) -> int:
    """Clamp value to valid RGB range."""
    return max(min_val, min(max_val, value))


def get_contrasting_font_color(
    bg_color: tuple[int, int, int] | str,
    contrast_threshold: float = 3.5,
    max_attempts: int = 100,
) -> tuple[int, int, int]:
    """
    Generate a font color that contrasts with the background.

    Uses WCAG 2.1 contrast ratio guidelines.

    Args:
        bg_color: Background color as RGB tuple or color name
        contrast_threshold: Minimum contrast ratio (WCAG recommends 4.5)
        max_attempts: Maximum attempts to find a suitable color

    Returns:
        RGB tuple for font color
    """
    if isinstance(bg_color, str):
        bg_color = ImageColor.getrgb(bg_color)

    bg_lum = _luminance(bg_color)

    # Try common high-contrast options first
    for font_color in [(0, 0, 0), (255, 255, 255), (50, 50, 50), (230, 230, 230)]:
        if _contrast_ratio(bg_lum, _luminance(font_color)) >= contrast_threshold:
            return font_color

    # Random generation fallback
    for _ in range(max_attempts):
        font_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        if _contrast_ratio(bg_lum, _luminance(font_color)) >= contrast_threshold:
            return font_color

    # Last resort: black or white based on background
    return (0, 0, 0) if bg_lum > 0.5 else (255, 255, 255)


def _luminance(color: tuple[int, int, int]) -> float:
    """Calculate relative luminance per WCAG 2.1."""

    def adjust(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = map(adjust, color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(lum1: float, lum2: float) -> float:
    """Calculate contrast ratio between two luminance values."""
    lighter, darker = max(lum1, lum2), min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)
