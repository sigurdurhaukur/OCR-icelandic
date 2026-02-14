"""Selection stages for choosing fonts, colors, textures, and layout."""

from PIL import Image, ImageColor

from ocr_icelandic import randomness

from ocr_icelandic.logging_config import get_logger
from ocr_icelandic.pipeline.core import BaseStage, PipelineState

logger = get_logger(__name__)


# Color generation utilities


def get_random_background_color() -> tuple[int, int, int]:
    """
    Generate a random background color with weighted distribution.

    Distribution: 85% light (paper-like), 10% dark, 5% colorful

    Returns:
        RGB color tuple
    """
    rand_val = randomness.random()

    if rand_val < 0.85:
        # Light colors (paper-like) - 85% probability
        paper_type = randomness.choice(["white", "cream", "aged"])

        if paper_type == "white":
            base = randomness.randint(245, 252)
            r = base + randomness.randint(-3, 3)
            g = base + randomness.randint(-5, 0)
            b = base + randomness.randint(-8, 0)
        elif paper_type == "cream":
            base = randomness.randint(235, 245)
            r = base + randomness.randint(0, 8)
            g = base + randomness.randint(-5, 3)
            b = base + randomness.randint(-12, -3)
        else:  # aged
            base = randomness.randint(220, 235)
            r = base + randomness.randint(5, 15)
            g = base + randomness.randint(0, 10)
            b = base + randomness.randint(-15, -5)

    elif rand_val < 0.95:
        # Dark colors - 10% probability
        base = randomness.randint(20, 80)
        r = base + randomness.randint(-10, 10)
        g = base + randomness.randint(-10, 10)
        b = base + randomness.randint(-10, 10)

    else:
        # Colorful - 5% probability
        bright_channel = randomness.randint(0, 2)
        colors = [0, 0, 0]
        colors[bright_channel] = randomness.randint(150, 255)

        for i in range(3):
            if i != bright_channel:
                colors[i] = randomness.randint(30, 220)

        r, g, b = colors

    # Clamp to valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return (r, g, b)


def get_random_font_color(
    bg_color: tuple[int, int, int] | str, contrast_threshold: float = 3.5
) -> tuple[int, int, int]:
    """
    Generate a random font color that contrasts with the background color.

    Uses WCAG 2.1 contrast ratio guidelines for better readability.

    Args:
        bg_color: Background color as RGB tuple or color name string
        contrast_threshold: Minimum contrast ratio

    Returns:
        RGB tuple representing the font color
    """

    def luminance(color: tuple[int, int, int]) -> float:
        """Calculate relative luminance per WCAG 2.1 specification."""
        r, g, b = color
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast_ratio(lum1: float, lum2: float) -> float:
        """Calculate contrast ratio between two luminance values."""
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        return (lighter + 0.05) / (darker + 0.05)

    # Convert bg_color to RGB tuple if it's a string
    if isinstance(bg_color, str):
        bg_color = ImageColor.getrgb(bg_color)

    bg_lum = luminance(bg_color)

    # Try common high-contrast options first
    candidates = [(0, 0, 0), (255, 255, 255), (50, 50, 50), (230, 230, 230)]
    for font_color in candidates:
        font_lum = luminance(font_color)
        if contrast_ratio(bg_lum, font_lum) >= contrast_threshold:
            return font_color

    # Fall back to random generation
    max_attempts = 100
    for _ in range(max_attempts):
        r = randomness.randint(0, 255)
        g = randomness.randint(0, 255)
        b = randomness.randint(0, 255)
        font_color = (r, g, b)
        font_lum = luminance(font_color)
        if contrast_ratio(bg_lum, font_lum) >= contrast_threshold:
            return font_color

    # Fallback based on background luminance
    return (0, 0, 0) if bg_lum > 0.5 else (255, 255, 255)


class SelectFontStage(BaseStage):
    """Select a font for text rendering."""

    def __init__(
        self,
        fonts: list[str] | None = None,
        fixed_font: str | None = None,
        random_selection: bool = True,
    ):
        super().__init__("SelectFont")
        self.fonts = fonts or []
        self.fixed_font = fixed_font
        self.random_selection = random_selection

    def __call__(self, state: PipelineState) -> PipelineState:
        if self.fixed_font:
            state.font_path = self.fixed_font
        elif self.random_selection and self.fonts:
            state.font_path = randomness.choice(self.fonts)
        elif self.fonts:
            state.font_path = self.fonts[0]

        self._add_metadata(state, "selected_font", state.font_path)
        logger.debug("Selected font: %s", state.font_path)
        return state


class SelectColorsStage(BaseStage):
    """Select background and font colors."""

    def __init__(
        self,
        random_background: bool = True,
        random_font_color: bool = True,
        fixed_bg_color: tuple[int, int, int] | str | None = None,
        fixed_font_color: tuple[int, int, int] | str | None = None,
        contrast_threshold: float = 3.5,
    ):
        super().__init__("SelectColors")
        self.random_background = random_background
        self.random_font_color = random_font_color
        self.fixed_bg_color = fixed_bg_color
        self.fixed_font_color = fixed_font_color
        self.contrast_threshold = contrast_threshold

    def __call__(self, state: PipelineState) -> PipelineState:
        if self.fixed_bg_color:
            state.bg_color = self.fixed_bg_color
        elif self.random_background:
            state.bg_color = get_random_background_color()

        if self.fixed_font_color:
            state.font_color = self.fixed_font_color
        elif self.random_font_color:
            state.font_color = get_random_font_color(
                state.bg_color, self.contrast_threshold
            )

        self._add_metadata(state, "bg_color", state.bg_color)
        self._add_metadata(state, "font_color", state.font_color)
        logger.debug(
            "Selected colors - bg: %s, font: %s", state.bg_color, state.font_color
        )
        return state


class SelectLayoutStage(BaseStage):
    """Select column layout parameters."""

    def __init__(
        self,
        num_columns: int | None = None,
        column_range: tuple[int, int] = (1, 3),
        column_width: int | None = None,
        column_width_range: tuple[int, int] = (100, 512),
        column_gap: int = 20,
        alignment: str | None = None,
        vertical_alignment: str | None = None,
    ):
        super().__init__("SelectLayout")
        self.num_columns = num_columns
        self.column_range = column_range
        self.column_width = column_width
        self.column_width_range = column_width_range
        self.column_gap = column_gap
        self.alignment = alignment
        self.vertical_alignment = vertical_alignment

    def __call__(self, state: PipelineState) -> PipelineState:
        if self.num_columns is not None:
            state.num_columns = self.num_columns
        else:
            state.num_columns = randomness.randint(*self.column_range)

        if self.column_width is not None:
            state.column_width = self.column_width
        else:
            state.column_width = randomness.randint(*self.column_width_range)

        state.column_gap = self.column_gap

        if self.alignment is not None:
            state.alignment = self.alignment

        if self.vertical_alignment is not None:
            state.vertical_alignment = self.vertical_alignment

        self._add_metadata(state, "num_columns", state.num_columns)
        self._add_metadata(state, "column_width", state.column_width)
        logger.debug(
            "Selected layout - columns: %d, width: %s",
            state.num_columns,
            state.column_width,
        )
        return state


class SelectPaperTextureStage(BaseStage):
    """Select a paper texture for the document background."""

    def __init__(
        self,
        textures: list[str] | None = None,
        probability: float = 1.0,
    ):
        super().__init__("SelectPaperTexture")
        self.textures = textures or []
        self.probability = probability

    def __call__(self, state: PipelineState) -> PipelineState:
        if self.textures and randomness.random() < self.probability:
            state.paper_texture_path = randomness.choice(self.textures)
            self._add_metadata(state, "texture", state.paper_texture_path)
            logger.debug("Selected paper texture: %s", state.paper_texture_path)
        return state


class SelectBackgroundImageStage(BaseStage):
    """Select and pre-expand a background image."""

    def __init__(
        self,
        no_shadow_backgrounds: list[str] | None = None,
        with_shadow_backgrounds: list[str] | None = None,
        probability: float = 1.0,
        expansion_factor: float = 1.8,
    ):
        super().__init__("SelectBackgroundImage")
        self.no_shadow_backgrounds = no_shadow_backgrounds or []
        self.with_shadow_backgrounds = with_shadow_backgrounds or []
        self.probability = probability
        self.expansion_factor = expansion_factor

    def __call__(self, state: PipelineState) -> PipelineState:
        if randomness.random() > self.probability:
            return state

        all_backgrounds: list[tuple[str, bool]] = []
        for bg in self.with_shadow_backgrounds:
            all_backgrounds.append((bg, True))
        for bg in self.no_shadow_backgrounds:
            all_backgrounds.append((bg, False))

        if not all_backgrounds:
            return state

        bg_path, receives_shadow = randomness.choice(all_backgrounds)

        try:
            background = Image.open(bg_path).convert("RGBA")

            # Pre-expand for transformations
            width, height = state.image_size
            expanded_width = int(width * self.expansion_factor)
            expanded_height = int(height * self.expansion_factor)

            background = background.resize(
                (expanded_width, expanded_height),
                Image.Resampling.BICUBIC,
            )

            state.background_image = background
            state.background_receives_shadow = receives_shadow
            self._add_metadata(state, "background_path", bg_path)
            self._add_metadata(state, "receives_shadow", receives_shadow)
            logger.debug(
                "Selected background: %s (shadow: %s)", bg_path, receives_shadow
            )

        except Exception as e:
            logger.warning("Failed to load background %s: %s", bg_path, e)

        return state
