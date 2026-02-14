"""Single image generation logic for synthetic OCR."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocr_icelandic import randomness
from ocr_icelandic.config import SingleImageData
from ocr_icelandic.pipeline.core import Pipeline, PipelineState
from ocr_icelandic.pipeline.stages.postprocessing import (
    CompositeBackgroundStage,
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
)
from ocr_icelandic.pipeline.stages.transformations import ApplyTransformationsStage

if TYPE_CHECKING:
    from ocr_icelandic.config import GenerationConfig


def generate_paragraph_styles(
    num_paragraphs: int,
    bold_probability: float = 0.2,
    underline_probability: float = 0.1,
) -> list[dict]:
    """
    Generate random style flags for paragraphs.

    Args:
        num_paragraphs: Number of paragraphs
        bold_probability: Probability of applying bold (0.0-1.0)
        underline_probability: Probability of applying underline (0.0-1.0)

    Returns:
        List of dicts: [{"bold": bool, "underline": bool}, ...]
    """
    from ocr_icelandic import randomness

    styles = []
    for _ in range(num_paragraphs):
        styles.append(
            {
                "bold": randomness.random() < bold_probability,
                "underline": randomness.random() < underline_probability,
            }
        )
    return styles


def _build_pipeline_stages(cfg: GenerationConfig) -> list:
    """Build the pipeline stages for image generation."""
    return [
        SelectFontStage(
            fonts=cfg.available_fonts or [],
            fixed_font=cfg.font_path if not cfg.use_random_fonts else None,
            random_selection=cfg.use_random_fonts,
        ),
        SelectColorsStage(
            random_background=cfg.use_random_backgrounds,
            random_font_color=cfg.use_random_font_colors,
            fixed_bg_color=cfg.img_background_color
            if not cfg.use_random_backgrounds
            else None,
            fixed_font_color=cfg.font_color if not cfg.use_random_font_colors else None,
        ),
        SelectLayoutStage(
            num_columns=cfg.num_columns,
            column_range=cfg.column_range,
            column_width=cfg.column_width,
            column_width_range=cfg.column_width_range,
            column_gap=cfg.column_gap,
            alignment=cfg.text_horizontal_alignment,
            vertical_alignment=cfg.text_vertical_alignment,
        ),
        SelectPaperTextureStage(
            textures=cfg.available_paper_textures or [],
            probability=1.0 if cfg.use_paper_textures else 0.0,
        ),
        SelectBackgroundImageStage(
            no_shadow_backgrounds=cfg.available_no_shadow_backgrounds or [],
            with_shadow_backgrounds=cfg.available_with_shadow_backgrounds or [],
            probability=cfg.background_image_probability,
        ),
        RenderTextStage(apply_displacement=True),
        ApplyTransformationsStage(
            probability_overrides={
                "textured_stains": 0.0
            },  # Disable stains for now, while we are developing the OCR model
            pipeline_type="auto",
        ),
        CompositeBackgroundStage(),
        FinalizeImageStage(use_random_composite=cfg.use_random_backgrounds),
        VisualizeBBoxesStage(enabled=False, show_labels=False),
    ]


def generate_single_chunk(
    chunk: str, cfg: GenerationConfig, seed: int | None = None
) -> list[SingleImageData]:
    """
    Generate images from a single text chunk.

    This function processes a single chunk of text (already split from a longer text)
    and generates one or more images from it. If the chunk doesn't fit in a single
    image, it generates multiple images.

    Args:
        chunk: A text chunk to render as image(s)
        cfg: Generation configuration
        seed: Optional seed for reproducibility. When running in parallel workers,
            each worker should receive a unique seed derived from base_seed + index.

    Returns:
        List of SingleImageData objects for the generated images
    """
    # Set seed for this chunk if provided (important for parallel workers)
    if seed is not None:
        randomness.set_seed(seed)

    stages = _build_pipeline_stages(cfg)
    images: list[SingleImageData] = []
    remaining_text = chunk

    while remaining_text:
        # Build paragraph font configs if needed
        paragraph_font_configs = None
        paragraph_font_sizes = None
        paragraph_styles = None

        if cfg.enable_font_size_variation or cfg.enable_font_styles:
            from ocr_icelandic.utils.text_layout import calculate_paragraph_font_sizes

            paragraphs = remaining_text.split("\n\n")
            num_paragraphs = len(paragraphs)

            if cfg.enable_font_size_variation:
                paragraph_font_sizes = calculate_paragraph_font_sizes(
                    paragraphs,
                    cfg.font_size or randomness.randint(*cfg.font_size_range),
                    cfg.font_size_min_ratio,
                    cfg.font_size_max_ratio,
                )
            else:
                paragraph_font_sizes = [
                    cfg.font_size or randomness.randint(*cfg.font_size_range)
                ] * num_paragraphs

            if cfg.enable_font_styles:
                paragraph_styles = generate_paragraph_styles(
                    num_paragraphs,
                    cfg.font_bold_probability,
                    cfg.font_underline_probability,
                )
            else:
                paragraph_styles = [
                    {"bold": False, "underline": False}
                ] * num_paragraphs

            # Note: ParagraphFontConfig will be built in RenderTextStage
            # when paragraph_font_configs is set on state

        # Create initial state
        initial_state = PipelineState(
            text=remaining_text,
            image_size=(cfg.image_width, cfg.image_height),
            dpi=cfg.image_dpi,
            render_scale=2,  # Render at 2x resolution, scale down for quality
            font_size=cfg.font_size or randomness.randint(*cfg.font_size_range),
            paragraph_font_configs=paragraph_font_configs,
            bbox_per_column=cfg.bbox_per_column,
            bbox_max_chars=cfg.bbox_max_chars,
            hyphenation_lang=cfg.language_code,
        )

        # Run the pipeline
        pipeline = Pipeline(stages=stages, initial_state=initial_state)
        result = pipeline.run()

        if not result.fitted_text:
            break

        images.append(
            SingleImageData(
                text=result.fitted_text,
                image=result.image,
                font_path=result.font_path or cfg.font_path,
                bg_color=result.bg_color,
                font_color=result.font_color,
                font_size=cfg.font_size or randomness.randint(*cfg.font_size_range),
                image_width=cfg.image_width,
                image_height=cfg.image_height,
                image_dpi=cfg.image_dpi,
                text_vertical_alignment=cfg.text_vertical_alignment,
                text_horizontal_alignment=cfg.text_horizontal_alignment,
                paragraph_bboxes=result.paragraph_bboxes,
                transformations=result.transformation_metadata,
                paragraph_font_sizes=paragraph_font_sizes,
                paragraph_styles=paragraph_styles,
            )
        )

        remaining_text = remaining_text[len(result.fitted_text) :].lstrip()

    return images
