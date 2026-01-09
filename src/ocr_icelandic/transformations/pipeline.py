"""Transformation pipeline configuration and orchestration.

This module provides:
- TRANSFORMATION_CONFIG: Registry of all available transformations
- Pipeline probability configurations for different scenarios
- Pipeline selection functions for various use cases
- apply_random_transformation: Main entry point for applying transformations
"""

import random

from PIL import Image

from ocr_icelandic.transformations.effects import (
    blur,
    dusty_paper,
    ink_splashes,
    reverse_bleed_through,
    textured_stains,
)
from ocr_icelandic.transformations.lighting import (
    light_reflection,
    shadow_gradient,
    shadow_overlay,
)
from ocr_icelandic.transformations.perspective import perspective
from ocr_icelandic.transformations.rotate import rotate
from ocr_icelandic.transformations.shared import _copy_paragraph_bboxes
from ocr_icelandic.transformations.skew import skew
from ocr_icelandic.transformations.tight_crop import tight_crop


# Transformation categories with their functions and default probabilities
TRANSFORMATION_CONFIG = {
    "content": {
        "blur": {"function": blur, "probability": 0.3},
        "ink_splashes": {"function": ink_splashes, "probability": 0.2},
        "dusty_paper": {"function": dusty_paper, "probability": 0.3},
        "reverse_bleed_through": {
            "function": reverse_bleed_through,
            "probability": 0.2,
        },
        "textured_stains": {"function": textured_stains, "probability": 0.2},
        "tight_crop": {"function": tight_crop, "probability": 0.25},
    },
    "perspective": {
        "rotate": {"function": rotate, "probability": 0.6},
        "skew": {"function": skew, "probability": 0.1},
        "perspective": {"function": perspective, "probability": 0.5},
    },
    "postprocessing": {
        "light_reflection": {"function": light_reflection, "probability": 0.3},
        "shadow_overlay": {"function": shadow_overlay, "probability": 0.4},
        "shadow_gradient": {"function": shadow_gradient, "probability": 0.7},
    },
}


# Pipeline configurations for different scenarios
PIPELINE_NO_BACKGROUND_PROBABILITIES = {
    # Content transformations - default probabilities
    "blur": 0.3,
    "ink_splashes": 0.2,
    "dusty_paper": 0.3,
    "reverse_bleed_through": 0.2,
    "textured_stains": 0.2,
    "tight_crop": 0.25,
    # Perspective transformations - default probabilities
    "rotate": 0.6,
    "skew": 0.1,
    "perspective": 0.5,
    # Postprocessing transformations - reduced probabilities for no background
    "light_reflection": 0.15,  # Reduced from 0.3
    "shadow_overlay": 0.2,  # Reduced from 0.4
    "shadow_gradient": 0.35,  # Reduced from 0.7
}

PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES = {
    # Content transformations - default probabilities
    "blur": 0.3,
    "ink_splashes": 0.2,
    "dusty_paper": 0.3,
    "reverse_bleed_through": 0.2,
    "textured_stains": 0.2,
    "tight_crop": 0.25,
    # Perspective transformations - default probabilities
    "rotate": 0.6,
    "skew": 0.1,
    "perspective": 0.5,
    # Postprocessing transformations - default probabilities
    "light_reflection": 0.3,
    "shadow_overlay": 0.4,
    "shadow_gradient": 0.7,
}

PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES = {
    # Content transformations - default probabilities
    "blur": 0.3,
    "ink_splashes": 0.2,
    "dusty_paper": 0.3,
    "reverse_bleed_through": 0.2,
    "textured_stains": 0.2,
    "tight_crop": 0.25,
    # Perspective transformations - default probabilities
    "rotate": 0.6,
    "skew": 0.1,
    "perspective": 0.5,
    # Postprocessing transformations - exclude light_reflection and shadow_overlay
    "light_reflection": 0.0,  # Disabled for distant backgrounds
    "shadow_overlay": 0.0,  # Disabled for distant backgrounds
    "shadow_gradient": 0.7,  # Keep gradient as it's more subtle
}


def _select_transformations_by_probability(
    category_config: dict,
    probability_overrides: dict | None = None,
) -> list:
    """Select transformations based on individual probabilities.

    Args:
        category_config: Dictionary of transformation configs with probabilities
        probability_overrides: Optional dict to override default probabilities

    Returns:
        List of selected transformation functions
    """
    selected = []
    for name, config in category_config.items():
        # Get probability (use override if provided)
        prob = (
            probability_overrides.get(name, config["probability"])
            if probability_overrides
            else config["probability"]
        )

        # Select based on probability
        if random.random() < prob:
            selected.append(config["function"])

    return selected


def _get_pipeline_no_background() -> list:
    """Pipeline 1: No photo background.

    Uses all transformations with reduced postprocessing probabilities.

    Returns:
        List of transformation functions to apply
    """
    transformations = []

    # Select from all categories using the no-background pipeline probabilities
    for category_name, category_config in TRANSFORMATION_CONFIG.items():
        transformations.extend(
            _select_transformations_by_probability(
                category_config, PIPELINE_NO_BACKGROUND_PROBABILITIES
            )
        )

    return transformations


def _get_pipeline_background_with_shadow() -> list:
    """Pipeline 2: Photo background with shadow (e.g., desk surface).

    Uses all transformations with default probabilities.
    Shadows and light reflections will cast on the close background.

    Returns:
        List of transformation functions to apply
    """
    transformations = []

    # Select from all categories using default probabilities
    for category_name, category_config in TRANSFORMATION_CONFIG.items():
        transformations.extend(
            _select_transformations_by_probability(
                category_config, PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES
            )
        )

    return transformations


def _get_pipeline_background_no_shadow() -> list:
    """Pipeline 3: Photo background without shadow (e.g., distant landscape).

    Uses all transformations except light_reflection and shadow_overlay.
    Distant backgrounds should not receive document shadows or light reflections.

    Returns:
        List of transformation functions to apply
    """
    transformations = []

    # Select from all categories, with light_reflection and shadow_overlay disabled
    for category_name, category_config in TRANSFORMATION_CONFIG.items():
        transformations.extend(
            _select_transformations_by_probability(
                category_config, PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES
            )
        )

    return transformations


def apply_random_transformation(
    image: Image.Image,
    bg_color: str | tuple[int, int, int],
    paragraph_bboxes: list[dict] | None = None,
    use_background: bool = False,
    background_has_shadow: bool = False,
    probability_overrides: dict | None = None,
) -> tuple[Image.Image, list[dict], list[dict]]:
    """Apply random transformations to an image using one of three hard-coded pipelines.

    Pipeline selection logic:
    1. No background: All transformations with reduced postprocessing probabilities
    2. Background with shadow: All transformations with default probabilities
    3. Background without shadow: All transformations except light_reflection and shadow_overlay

    Args:
        image: Input RGBA image
        bg_color: Background color (kept for API compatibility, passed to transformations)
        paragraph_bboxes: Optional bounding boxes to transform
        use_background: Whether a background image will be used
        background_has_shadow: If True, background receives shadows (close background);
                               if False, background doesn't receive shadows (distant background)
        probability_overrides: Optional dict to override transformation probabilities
                              (only used if custom behavior needed beyond the 3 pipelines)

    Returns:
        Tuple of (transformed RGBA image, transformation metadata, transformed bboxes)
    """
    paragraph_bboxes_copy = _copy_paragraph_bboxes(paragraph_bboxes)

    # Select pipeline based on background configuration
    if not use_background:
        # Pipeline 1: No photo background
        transformations_to_apply = _get_pipeline_no_background()
    elif background_has_shadow:
        # Pipeline 2: Photo background with shadow (e.g., desk)
        transformations_to_apply = _get_pipeline_background_with_shadow()
    else:
        # Pipeline 3: Photo background without shadow (e.g., landscape)
        transformations_to_apply = _get_pipeline_background_no_shadow()

    # Apply probability overrides if provided (for custom behavior)
    if probability_overrides:
        # Re-select transformations with custom probabilities
        transformations_to_apply = []
        for category_name, category_config in TRANSFORMATION_CONFIG.items():
            transformations_to_apply.extend(
                _select_transformations_by_probability(
                    category_config, probability_overrides
                )
            )

    # Apply selected transformations
    transformation_meta: list[dict] = []
    for transform in transformations_to_apply:
        image, meta, paragraph_bboxes_copy = transform(
            image, bg_color, paragraph_bboxes_copy
        )
        transformation_meta.append(meta)

    # Return RGBA image directly (no RGB composite here)
    return image, transformation_meta, paragraph_bboxes_copy
