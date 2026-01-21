"""Re-exports for backwards compatibility.

This module re-exports all transformation functions and configuration
from their new locations. Import directly from the specific modules
for better clarity:

- effects.py: Content transformations (blur, etc.)
- lighting.py: Lighting effects (light_reflection, shadow_overlay, etc.)
- pipeline.py: Configuration and apply_random_transformation
"""

# Re-export content effects
from ocr_icelandic.transformations.effects import (
    blur,
    dusty_paper,
    reverse_bleed_through,
    stain_textures,
    textured_stains,
)

# Re-export lighting effects
from ocr_icelandic.transformations.lighting import (
    light_reflection,
    shadow_gradient,
    shadow_overlay,
)

# Re-export pipeline configuration and main function
from ocr_icelandic.transformations.pipeline import (
    PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES,
    PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES,
    PIPELINE_NO_BACKGROUND_PROBABILITIES,
    TRANSFORMATION_CONFIG,
    apply_random_transformation,
)

__all__ = [
    # Effects
    "blur",
    "textured_stains",
    "dusty_paper",
    "reverse_bleed_through",
    "stain_textures",
    # Lighting
    "light_reflection",
    "shadow_overlay",
    "shadow_gradient",
    # Pipeline
    "TRANSFORMATION_CONFIG",
    "PIPELINE_NO_BACKGROUND_PROBABILITIES",
    "PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES",
    "PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES",
    "apply_random_transformation",
]
