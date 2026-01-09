"""Utilities for OCR image generation.

This module provides utilities for creating synthetic OCR training images,
including text rendering, color manipulation, texture application, and
visualization tools.
"""

# Color utilities
from ocr_icelandic.utils.color import (
    blend_text_layer,
    calculate_luminance,
    color_to_rgb,
    get_blend_mode,
    is_grayscale,
)

# Font utilities
from ocr_icelandic.utils.font import load_font

# Texture and background utilities
from ocr_icelandic.utils.texture import (
    apply_background_image,
    apply_paper_texture,
    create_paper_drop_shadow,
    discover_backgrounds,
    discover_paper_textures,
)

# Text layout utilities
from ocr_icelandic.utils.text_layout import (
    LinePlacement,
    WrappedParagraph,
    WrapResult,
    arrange_lines_in_columns,
    wrap_text,
)

# Image creation
from ocr_icelandic.utils.image_creation import create_image_with_text

# Visualization utilities
from ocr_icelandic.utils.visualization import (
    dummy_text_with_line_breaks,
    visualise_bboxes,
)

# Backward compatibility aliases for underscore-prefixed functions
_color_to_rgb = color_to_rgb
_calculate_luminance = calculate_luminance
_is_grayscale = is_grayscale
_create_paper_drop_shadow = create_paper_drop_shadow
_visualise_bboxes = visualise_bboxes

__all__ = [
    # Color utilities
    "blend_text_layer",
    "calculate_luminance",
    "color_to_rgb",
    "get_blend_mode",
    "is_grayscale",
    # Font utilities
    "load_font",
    # Texture utilities
    "apply_background_image",
    "apply_paper_texture",
    "create_paper_drop_shadow",
    "discover_backgrounds",
    "discover_paper_textures",
    # Text layout
    "LinePlacement",
    "WrappedParagraph",
    "WrapResult",
    "arrange_lines_in_columns",
    "wrap_text",
    # Image creation
    "create_image_with_text",
    # Visualization
    "dummy_text_with_line_breaks",
    "visualise_bboxes",
    # Backward compatibility aliases
    "_color_to_rgb",
    "_calculate_luminance",
    "_is_grayscale",
    "_create_paper_drop_shadow",
    "_visualise_bboxes",
]
