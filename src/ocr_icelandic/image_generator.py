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
from ocr_icelandic.text_processing import split_long_text

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
        ApplyTransformationsStage(pipeline_type="auto"),
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


def generate_single_text(
    text: str, cfg: GenerationConfig
) -> tuple[list[SingleImageData], int]:
    """
    Generate images from text using the pipeline architecture.

    This function splits the text into chunks and generates images for each chunk.
    For better parallelization at the chunk level, use generate_single_chunk()
    directly after splitting texts with split_long_text().

    Args:
        text: Full text to render as images
        cfg: Generation configuration

    Returns:
        Tuple of (list of SingleImageData, number of chunks the text was split into)
    """
    # Split long texts first
    text_chunks = split_long_text(text.strip(), cfg.max_text_length)

    for chunk in text_chunks:
        remaining_text = chunk

        while remaining_text:
            # Resolve random settings
            settings = _resolve_random_settings(cfg)

            # Create the base image
            image, fitted_text, paragraph_bboxes = create_image_with_text(
                remaining_text,
                image_size=(cfg.image_width, cfg.image_height),
                alignment=cfg.text_horizontal_alignment,
                font_size=settings["font_size"],
                font_path=settings["font_path"],
                bg_color=settings["bg_color"],
                font_color=settings["font_color"],
                vertical_alignment=cfg.text_vertical_alignment,
                dpi=cfg.image_dpi,
                num_columns=settings["num_columns"],
                column_gap=cfg.column_gap,
                column_width=settings["column_width"],
                paper_texture_path=settings["paper_texture_path"],
                hyphenation_lang=cfg.language_code,
            )

            if not fitted_text:
                # No text could be fitted, break to avoid infinite loop
                break

            # Apply transformations
            if cfg.apply_random_transformations:
                transformed_image, transformation_meta, transformed_bboxes = (
                    _apply_transformations(image, settings, paragraph_bboxes, cfg)
                )
            else:
                transformed_image = image
                transformation_meta = [{"type": "none"}]
                transformed_bboxes = paragraph_bboxes

            # Final RGB conversion
            final_image = _convert_to_rgb(
                transformed_image, settings["composite_bg_color"]
            )

            yield SingleImageData(
                text=fitted_text,
                image=final_image,
                font_path=settings["font_path"],
                bg_color=settings["bg_color"],
                font_color=settings["font_color"],
                font_size=settings["font_size"],
                image_width=cfg.image_width,
                image_height=cfg.image_height,
                image_dpi=cfg.image_dpi,
                text_vertical_alignment=cfg.text_vertical_alignment,
                text_horizontal_alignment=cfg.text_horizontal_alignment,
                paragraph_bboxes=transformed_bboxes,
                transformations=transformation_meta,
            )

            remaining_text = remaining_text[len(fitted_text) :].lstrip()


def _resolve_random_settings(cfg: "GenerationConfig") -> dict:
    """Resolve all random settings for a single image."""
    settings = {
        "font_path": "Not Set",
        "bg_color": cfg.img_background_color,
        "composite_bg_color": cfg.img_background_color,
        "font_color": cfg.font_color,
        "font_size": cfg.font_size,
        "paper_texture_path": None,
    }

    # Random font
    if cfg.use_random_fonts and cfg.available_fonts:
        settings["font_path"] = random.choice(cfg.available_fonts)

    # Random background colors
    if cfg.use_random_backgrounds:
        settings["bg_color"] = get_random_background_color()
        settings["composite_bg_color"] = get_random_background_color()

    # Random paper texture (based on probability, otherwise use synthetic background with noise)
    if cfg.use_paper_textures and cfg.available_paper_textures:
        if random.random() < cfg.paper_texture_probability:
            settings["paper_texture_path"] = random.choice(cfg.available_paper_textures)
        # else: paper_texture_path remains None, which triggers synthetic background with noise in create_image_with_text

    # Random font color (must be after bg_color)
    if cfg.use_random_font_colors:
        settings["font_color"] = get_contrasting_font_color(settings["bg_color"])

    # Random font size
    if cfg.use_random_font_sizes:
        settings["font_size"] = random.randint(*cfg.font_size_range)

    # Columns
    if cfg.num_columns is not None and cfg.num_columns > 0:
        settings["num_columns"] = cfg.num_columns
    else:
        settings["num_columns"] = random.randint(*cfg.column_range)

    # Column width
    if cfg.column_width is not None and cfg.column_width > 0:
        settings["column_width"] = cfg.column_width
    else:
        total_gap = (settings["num_columns"] - 1) * cfg.column_gap
        max_width = (cfg.image_width - total_gap) // settings["num_columns"]
        min_width = min(cfg.column_width_range[0], max_width)
        settings["column_width"] = random.randint(min_width, max_width)

    return settings


def _apply_transformations(
    image: PILImage.Image,
    settings: dict,
    paragraph_bboxes: list[dict],
    cfg: "GenerationConfig",
) -> tuple[PILImage.Image, list[dict], list[dict]]:
    """Apply transformations and background to image."""
    # Decide on background
    use_background, background_path, background_has_shadow = _select_background(cfg)

    # Apply transformation pipeline
    transformed, meta, transformed_bboxes = apply_random_transformation(
        image,
        settings["bg_color"],
        paragraph_bboxes=paragraph_bboxes,
        use_background=use_background,
        background_has_shadow=background_has_shadow,
    )

    # Apply background image
    if use_background and background_path:
        transformed, bg_meta, transformed_bboxes = apply_background_image(
            transformed, background_path, paragraph_bboxes=transformed_bboxes
        )
        meta.append({"transformation": "background", **bg_meta})

    return transformed, meta, transformed_bboxes


def _select_background(
    cfg: "GenerationConfig",
) -> tuple[bool, str | None, bool]:
    """Select background settings."""
    if not cfg.use_background_images:
        return False, None, True

    if not (
        cfg.available_no_shadow_backgrounds or cfg.available_with_shadow_backgrounds
    ):
        return False, None, True

    if random.random() >= cfg.background_image_probability:
        return False, None, True

    # Build list of all backgrounds with shadow info
    all_backgrounds = []
    if cfg.available_with_shadow_backgrounds:
        all_backgrounds.extend(
            (bg, True) for bg in cfg.available_with_shadow_backgrounds
        )
    if cfg.available_no_shadow_backgrounds:
        all_backgrounds.extend(
            (bg, False) for bg in cfg.available_no_shadow_backgrounds
        )

    if all_backgrounds:
        path, has_shadow = random.choice(all_backgrounds)
        return True, path, has_shadow

    return False, None, True


def _convert_to_rgb(
    image: PILImage.Image, bg_color: tuple[int, int, int] | str
) -> PILImage.Image:
    """Convert image to RGB with background color."""
    if image.mode == "RGBA":
        rgb_bg = PILImage.new("RGB", image.size, bg_color)
        rgb_bg.paste(image, (0, 0), image)
        return rgb_bg
    elif image.mode != "RGB":
        return image.convert("RGB")
    return image
