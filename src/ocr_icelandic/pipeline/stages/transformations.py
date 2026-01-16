"""Transformation stages for applying image effects."""

import random
from typing import Any, Callable

from PIL import Image

from ocr_icelandic.logging_config import get_logger
from ocr_icelandic.pipeline.core import BaseStage, PipelineState

logger = get_logger(__name__)


class ApplyTransformationsStage(BaseStage):
    """
    Apply random transformations using probability-based selection.

    Wraps the existing transformation functions with the new state-based API.
    Automatically selects pipeline type based on background presence.
    """

    def __init__(
        self,
        probability_overrides: dict[str, float] | None = None,
        pipeline_type: str = "auto",  # "auto", "no_background", "with_shadow", "no_shadow"
    ):
        super().__init__("ApplyTransformations")
        self.probability_overrides = probability_overrides or {}
        self.pipeline_type = pipeline_type

    def __call__(self, state: PipelineState) -> PipelineState:
        from ocr_icelandic.transformations.pipeline import (
            PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES,
            PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES,
            PIPELINE_NO_BACKGROUND_PROBABILITIES,
            TRANSFORMATION_CONFIG,
            perspective,
            rotate,
        )

        if state.image is None:
            logger.warning("No image to transform, skipping ApplyTransformationsStage")
            return state

        # Determine pipeline type
        if self.pipeline_type == "auto":
            if state.background_image is None:
                probabilities = PIPELINE_NO_BACKGROUND_PROBABILITIES.copy()
            elif state.background_receives_shadow:
                probabilities = PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES.copy()
            else:
                probabilities = PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES.copy()
        elif self.pipeline_type == "no_background":
            probabilities = PIPELINE_NO_BACKGROUND_PROBABILITIES.copy()
        elif self.pipeline_type == "with_shadow":
            probabilities = PIPELINE_BACKGROUND_WITH_SHADOW_PROBABILITIES.copy()
        else:
            probabilities = PIPELINE_BACKGROUND_NO_SHADOW_PROBABILITIES.copy()

        # Apply overrides
        probabilities.update(self.probability_overrides)

        logger.debug("Using pipeline type: %s", self.pipeline_type)

        # Select and apply transformations
        for category_name, category_config in TRANSFORMATION_CONFIG.items():
            for transform_name, config in category_config.items():
                prob = probabilities.get(transform_name, config["probability"])
                if random.random() >= prob:
                    continue

                transform_func = config["function"]
                logger.debug("Applying transformation: %s (prob: %.2f)", transform_name, prob)

                # Handle transforms that modify background (perspective, rotate)
                if transform_func in [perspective, rotate]:
                    (
                        state.image,
                        meta,
                        state.paragraph_bboxes,
                        state.background_image,
                    ) = transform_func(
                        state.image,
                        state.bg_color,
                        state.paragraph_bboxes,
                        state.background_image,
                    )
                else:
                    state.image, meta, state.paragraph_bboxes = transform_func(
                        state.image,
                        state.bg_color,
                        state.paragraph_bboxes,
                    )

                state.transformation_metadata.append(meta)

        return state


class SingleTransformStage(BaseStage):
    """
    Apply a single specific transformation.

    Useful for fine-grained control over transformation order.
    """

    def __init__(
        self,
        transform_name: str,
        transform_func: Callable,
        probability: float = 1.0,
        supports_background: bool = False,
    ):
        super().__init__(f"Transform_{transform_name}")
        self.transform_name = transform_name
        self.transform_func = transform_func
        self.probability = probability
        self.supports_background = supports_background

    def __call__(self, state: PipelineState) -> PipelineState:
        if state.image is None:
            return state

        if random.random() >= self.probability:
            return state

        logger.debug("Applying single transformation: %s", self.transform_name)

        if self.supports_background:
            (
                state.image,
                meta,
                state.paragraph_bboxes,
                state.background_image,
            ) = self.transform_func(
                state.image,
                state.bg_color,
                state.paragraph_bboxes,
                state.background_image,
            )
        else:
            state.image, meta, state.paragraph_bboxes = self.transform_func(
                state.image,
                state.bg_color,
                state.paragraph_bboxes,
            )

        state.transformation_metadata.append(meta)
        return state


# Factory functions for common single transform stages


def create_rotate_stage(probability: float = 0.6) -> SingleTransformStage:
    """Create a rotate transformation stage."""
    from ocr_icelandic.transformations.rotate import rotate

    return SingleTransformStage(
        transform_name="rotate",
        transform_func=rotate,
        probability=probability,
        supports_background=True,
    )


def create_perspective_stage(probability: float = 0.5) -> SingleTransformStage:
    """Create a perspective transformation stage."""
    from ocr_icelandic.transformations.perspective import perspective

    return SingleTransformStage(
        transform_name="perspective",
        transform_func=perspective,
        probability=probability,
        supports_background=True,
    )


def create_blur_stage(probability: float = 0.3) -> SingleTransformStage:
    """Create a blur transformation stage."""
    from ocr_icelandic.transformations.effects import blur

    return SingleTransformStage(
        transform_name="blur",
        transform_func=blur,
        probability=probability,
        supports_background=False,
    )


def create_dusty_paper_stage(probability: float = 0.3) -> SingleTransformStage:
    """Create a dusty paper transformation stage."""
    from ocr_icelandic.transformations.effects import dusty_paper

    return SingleTransformStage(
        transform_name="dusty_paper",
        transform_func=dusty_paper,
        probability=probability,
        supports_background=False,
    )
