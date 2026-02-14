"""Built-in pipeline stages."""

from ocr_icelandic.pipeline.stages.postprocessing import (
    CompositeBackgroundStage,
    CropToContentStage,
    FinalizeImageStage,
    VisualizeBBoxesStage,
)
from ocr_icelandic.pipeline.stages.rendering import RenderTextStage
from ocr_icelandic.pipeline.stages.selection import (
    SelectBackgroundImageStage,
    SelectColorsStage,
    SelectFontStage,
    SelectLayoutStage,
    SelectPaperTextureStage,
    get_random_background_color,
    get_random_font_color,
)
from ocr_icelandic.pipeline.stages.transformations import (
    ApplyTransformationsStage,
    SingleTransformStage,
    create_blur_stage,
    create_dusty_paper_stage,
    create_perspective_stage,
    create_rotate_stage,
)

__all__ = [
    # Selection stages
    "SelectFontStage",
    "SelectColorsStage",
    "SelectLayoutStage",
    "SelectPaperTextureStage",
    "SelectBackgroundImageStage",
    # Rendering stages
    "RenderTextStage",
    # Transformation stages
    "ApplyTransformationsStage",
    "SingleTransformStage",
    "create_rotate_stage",
    "create_perspective_stage",
    "create_blur_stage",
    "create_dusty_paper_stage",
    # Post-processing stages
    "CompositeBackgroundStage",
    "FinalizeImageStage",
    "VisualizeBBoxesStage",
    "CropToContentStage",
    # Utility functions
    "get_random_background_color",
    "get_random_font_color",
]
