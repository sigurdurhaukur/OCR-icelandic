"""Image transformations for synthetic OCR data generation.

This package provides various image transformations to make synthetic
OCR training data more realistic. Transformations are organized into:

- effects.py: Content transformations (blur, stains, dust, ink splashes)
- lighting.py: Lighting effects (reflections, shadows, gradients)
- perspective.py: 3D perspective distortions
- rotate.py: Rotation transformations
- skew.py: Horizontal skew/shear transformations
- tight_crop.py: Content-aware cropping
- pipeline.py: Configuration and orchestration

The main entry point is `apply_random_transformation` which applies
a random subset of transformations based on configurable probabilities.
"""

from ocr_icelandic.transformations.pipeline import apply_random_transformation

__all__ = [
    "apply_random_transformation",
]
