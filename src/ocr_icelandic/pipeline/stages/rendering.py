"""Rendering stages for text-to-image generation."""

from ocr_icelandic.logging_config import get_logger
from ocr_icelandic.pipeline.core import BaseStage, PipelineState

logger = get_logger(__name__)


class RenderTextStage(BaseStage):
    """
    Render text onto an image using create_image_with_text().

    Reads configuration from state and produces the initial document image.
    """

    def __init__(
        self,
        apply_displacement: bool = True,
        displacement_strength: float = 1.5,
        displacement_lighting: bool = True,
        max_width_ratio: float = 0.9,
        tab_width: int = 4,
    ):
        super().__init__("RenderText")
        self.apply_displacement = apply_displacement
        self.displacement_strength = displacement_strength
        self.displacement_lighting = displacement_lighting
        self.max_width_ratio = max_width_ratio
        self.tab_width = tab_width

    def __call__(self, state: PipelineState) -> PipelineState:
        from ocr_icelandic.utils.image_creation import create_image_with_text

        # Apply render_scale to dimensions
        scale = state.render_scale
        scaled_size = (state.image_size[0] * scale, state.image_size[1] * scale)
        scaled_font_size = state.font_size * scale
        scaled_column_gap = state.column_gap * scale
        scaled_column_width = (
            state.column_width * scale if state.column_width is not None else None
        )

        image, fitted_text, paragraph_bboxes = create_image_with_text(
            text=state.text,
            image_size=scaled_size,
            font_path=state.font_path or "Arial.ttf",
            font_size=scaled_font_size,
            font_color=state.font_color,
            bg_color=state.bg_color,
            max_width_ratio=self.max_width_ratio,
            tab_width=self.tab_width,
            alignment=state.alignment,
            vertical_alignment=state.vertical_alignment,
            dpi=state.dpi,
            num_columns=state.num_columns,
            column_gap=scaled_column_gap,
            column_width=scaled_column_width,
            paper_texture_path=state.paper_texture_path,
            apply_displacement=self.apply_displacement,
            displacement_strength=self.displacement_strength,
            displacement_lighting=self.displacement_lighting,
            paragraph_font_configs=state.paragraph_font_configs,
        )

        state.image = image
        state.fitted_text = fitted_text
        state.paragraph_bboxes = paragraph_bboxes

        self._add_metadata(state, "fitted_text_length", len(fitted_text))
        self._add_metadata(state, "num_paragraphs", len(paragraph_bboxes))
        logger.debug(
            "Rendered text: %d chars fitted, %d paragraphs",
            len(fitted_text),
            len(paragraph_bboxes),
        )
        return state
