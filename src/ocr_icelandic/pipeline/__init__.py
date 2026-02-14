"""
Configurable pipeline for OCR image generation.

This module provides a unified pipeline architecture for creating synthetic
OCR training images. The pipeline consists of stages that transform a shared
state object, allowing flexible composition of operations.

Example usage:

    from ocr_icelandic.pipeline import Pipeline, PipelineState
    from ocr_icelandic.pipeline.stages import (
        SelectFontStage,
        SelectColorsStage,
        SelectLayoutStage,
        RenderTextStage,
        ApplyTransformationsStage,
        FinalizeImageStage,
    )

    pipeline = Pipeline(
        stages=[
            SelectFontStage(fonts=["Arial.ttf", "Times.ttf"]),
            SelectColorsStage(random_background=True),
            SelectLayoutStage(column_range=(1, 2)),
            RenderTextStage(),
            ApplyTransformationsStage(),
            FinalizeImageStage(),
        ],
        initial_state=PipelineState(
            text="Sample text for OCR",
            image_size=(512, 512),
        ),
    )

    result = pipeline.run()
    # result.image contains the generated image
    # result.fitted_text contains the text that fit
    # result.paragraph_bboxes contains bounding boxes
"""

from ocr_icelandic.pipeline.core import (
    BaseStage,
    LambdaStage,
    Pipeline,
    PipelineState,
    Stage,
)

__all__ = [
    "Pipeline",
    "PipelineState",
    "Stage",
    "BaseStage",
    "LambdaStage",
]
