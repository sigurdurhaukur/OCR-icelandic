"""Single image generation logic for synthetic OCR."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Iterator

from PIL import Image as PILImage

from ocr_icelandic.colors import get_contrasting_font_color, get_random_background_color
from ocr_icelandic.text_processing import split_long_text
from ocr_icelandic.transformations import apply_random_transformation
from ocr_icelandic.utils import apply_background_image, create_image_with_text

if TYPE_CHECKING:
    from ocr_icelandic.config import GenerationConfig, SingleImageData


def generate_single_text(
    text: str, cfg: "GenerationConfig"
) -> Iterator["SingleImageData"]:
    """
    Generate images for a single text entry, yielding one at a time.

    This generator handles text overflow by splitting long text and yielding
    each generated image individually, allowing for immediate flushing to disk
    and consistent batch sizes.

    Args:
        text: The text to convert to images
        cfg: Configuration for image generation

    Yields:
        SingleImageData for each generated image
    """
    # Import here to avoid circular imports
    from ocr_icelandic.config import SingleImageData

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
