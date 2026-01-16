"""OCR-icelandic: Language-agnostic OCR model training pipeline.

This package provides tools for generating synthetic OCR training data
and fine-tuning vision-language models for optical character recognition.
"""

from ocr_icelandic import randomness
from ocr_icelandic.font_cache import FontCompatibilityCache
from ocr_icelandic.fonts import (
    get_compatible_fonts,
    get_icelandic_compatible_fonts,
)
from ocr_icelandic.language_support import (
    LanguageCharacterSet,
    LanguageRegistry,
)
from ocr_icelandic.pipeline import (
    BaseStage,
    LambdaStage,
    Pipeline,
    PipelineState,
    Stage,
)

__version__ = "0.2.0"

__all__ = [
    # Randomness management
    "randomness",
    # Language support
    "LanguageCharacterSet",
    "LanguageRegistry",
    # Font utilities
    "FontCompatibilityCache",
    "get_compatible_fonts",
    "get_icelandic_compatible_fonts",
    # Pipeline
    "Pipeline",
    "PipelineState",
    "Stage",
    "BaseStage",
    "LambdaStage",
]
