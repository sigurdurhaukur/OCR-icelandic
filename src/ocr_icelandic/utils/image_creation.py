"""Main image creation functionality for OCR training data."""

from PIL import Image, ImageDraw

from ocr_icelandic.logging_config import get_logger
from ocr_icelandic.utils.color import (
    blend_text_layer,
    color_to_rgb,
    get_blend_mode,
)
from ocr_icelandic.utils.font import load_font, load_font_with_style
from ocr_icelandic.utils.text_layout import (
    ParagraphFontConfig,
    arrange_lines_in_columns,
    wrap_text,
)
from ocr_icelandic.utils.texture import (
    apply_displacement_from_texture,
    apply_paper_texture,
)

logger = get_logger(__name__)


def create_image_with_text(
    text: str,
    image_size: tuple[int, int] = (400, 100),
    font_path: str = "Arial.ttf",
    font_size: int = 20,
    font_color: str | tuple[int, int, int] = "black",
    bg_color: str | tuple[int, int, int] = "white",
    max_width_ratio: float = 0.9,
    tab_width: int = 4,
    alignment: str = "center",
    vertical_alignment: str = "center",
    dpi: int = 72,
    num_columns: int = 1,
    column_gap: int = 20,
    column_width: int | None = None,
    paper_texture_path: str | None = None,
    apply_displacement: bool = False,
    displacement_strength: float = 3.5,
    displacement_lighting: bool = True,
    paragraph_font_configs: list[ParagraphFontConfig] | None = None,
    add_noise: bool = True,
) -> tuple[Image.Image, str, list[dict]]:
    """
    Create an image with text for OCR training and return paragraph bounding boxes.

    Args:
        text: Text to render
        image_size: Tuple of (width, height) in pixels at default DPI (72)
        font_path: Path to the .ttf font file
        font_size: Size of the font in points at default DPI (72)
        font_color: Color of the font
        bg_color: Background color of the image
        max_width_ratio: Ratio of image width to use for text (0.0-1.0)
        tab_width: Number of spaces to replace tabs with
        alignment: Text alignment - 'center', 'left', or 'right'
        vertical_alignment: Vertical text alignment - 'top', 'center', or 'bottom'
        dpi: Dots per inch for the image
        num_columns: Number of columns to use when laying out text
        column_gap: Gap in pixels between columns
        column_width: Fixed pixel width for each column (None to auto-size)
        paper_texture_path: Optional path to paper texture image to use as background
        apply_displacement: If True and paper_texture_path is provided, warp text to follow
            the paper's folds/creases using displacement mapping
        displacement_strength: Pixel displacement multiplier (1.0-5.0 typical)
        displacement_lighting: If True, apply lighting effects based on paper surface normals
        add_noise: If True, add Gaussian noise to the background for realism. Set to False
            for deterministic output (useful for testing).

    Returns:
        tuple: (PIL Image object, string of text that actually fits in the image, paragraph bounding boxes)
    """
    logger.debug(
        "Creating image with text: size=%dx%d, dpi=%d, columns=%d",
        image_size[0],
        image_size[1],
        dpi,
        num_columns,
    )
    logger.debug(
        "Font: %s, size=%d, color=%s, bg=%s", font_path, font_size, font_color, bg_color
    )

    scale_factor = dpi / 72.0
    scaled_image_size = (
        int(image_size[0] * scale_factor),
        int(image_size[1] * scale_factor),
    )
    scaled_font_size = int(font_size * scale_factor)
    logger.debug(
        "Scaled dimensions: %dx%d, scaled font size: %d",
        scaled_image_size[0],
        scaled_image_size[1],
        scaled_font_size,
    )

    # Convert bg_color to RGBA if it's RGB
    if isinstance(bg_color, tuple) and len(bg_color) == 3:
        bg_color_rgba = bg_color + (255,)
    elif isinstance(bg_color, str):
        # PIL will handle string colors, but we need RGBA
        temp_img = Image.new("RGB", (1, 1), color=bg_color)
        rgb = temp_img.getpixel((0, 0))
        bg_color_rgba = rgb + (255,)
    else:
        bg_color_rgba = bg_color

    image = Image.new("RGBA", scaled_image_size, color=bg_color_rgba)
    image.info["dpi"] = (dpi, dpi)
    draw = ImageDraw.Draw(image)

    # Apply paper texture if provided
    if paper_texture_path is not None:
        logger.debug("Applying paper texture from: %s", paper_texture_path)
        # Convert to RGB temporarily for texture application
        image_rgb = image.convert("RGB")
        image_rgb = apply_paper_texture(image_rgb, paper_texture_path, blend_alpha=0.9)
        # Convert back to RGBA
        image = image_rgb.convert("RGBA")
        draw = ImageDraw.Draw(image)
    elif add_noise:
        logger.debug("Adding Gaussian noise to background for realism")
        # add gaussian noice to the background to make it more realistic and less uniform
        noise = Image.effect_noise(scaled_image_size, 10)
        noise_rgba = noise.convert("RGBA")
        image = Image.blend(image, noise_rgba, 0.1)
        draw = ImageDraw.Draw(image)

        # add "dirt" texture to the background
        dirt_texture = Image.effect_noise(scaled_image_size, 5)
        dirt_rgba = dirt_texture.convert("RGBA")
        image = Image.blend(image, dirt_rgba, 0.05)
        draw = ImageDraw.Draw(image)
    else:
        logger.debug("Skipping noise for deterministic output")

    font = load_font(font_path=font_path, font_size=scaled_font_size)

    # Font cache to avoid reloading same font+size combinations
    font_cache: dict[tuple, tuple] = {}  # key -> (font, needs_bold_simulation)

    # Build default configs if not provided
    if paragraph_font_configs is None:
        # Split text into paragraphs to count them
        temp_paragraphs = text.split("\n\n")
        paragraph_font_configs = [
            ParagraphFontConfig(
                font_path=font_path,
                font_size=scaled_font_size,
                bold=False,
                underline=False,
            )
            for _ in temp_paragraphs
        ]
        logger.debug(
            "Created default font configs for %d paragraphs", len(temp_paragraphs)
        )

    usable_width = max(1, int(scaled_image_size[0] * max_width_ratio))
    num_columns = max(1, num_columns)
    column_gap = max(0, column_gap)

    # Retry loop: reduce columns if words don't fit
    wrapped_paragraphs = None
    has_overflow = True
    current_column_width = 0
    retry_count = 0
    while has_overflow and num_columns >= 1:
        total_gap = column_gap * (num_columns - 1)
        if usable_width - total_gap <= 0:
            num_columns = 1
            column_gap = 0
            total_gap = 0

        max_available_width = max(1, usable_width - total_gap)
        if max_available_width < num_columns:
            num_columns = 1
            column_gap = 0
            total_gap = 0
            max_available_width = max(1, usable_width)
        if column_width is not None:
            requested_width = max(1, column_width)
            resolved_column_width = min(requested_width, max_available_width)
            if resolved_column_width * num_columns > max_available_width:
                resolved_column_width = max(1, max_available_width // num_columns)
        else:
            resolved_column_width = max(1, max_available_width // num_columns)

        resolved_column_width = max(1, resolved_column_width)
        current_column_width = resolved_column_width

        # Try wrapping with current column configuration (per-paragraph fonts)
        logger.debug(
            "Wrap attempt %d: columns=%d, width=%d",
            retry_count,
            num_columns,
            current_column_width,
        )

        # Wrap text with per-paragraph font configurations
        wrapped_paragraphs = []
        has_overflow = False

        paragraphs = text.split("\n\n")
        for idx, para_text in enumerate(paragraphs):
            # Get font config for this paragraph (or default)
            if idx < len(paragraph_font_configs):
                config = paragraph_font_configs[idx]
            else:
                config = ParagraphFontConfig(font_path, scaled_font_size, False, False)

            # Load font for this paragraph
            cache_key = config.get_cache_key()
            if cache_key not in font_cache:
                para_font, needs_sim = load_font_with_style(
                    config.font_path, config.font_size, config.bold, False
                )
                font_cache[cache_key] = (para_font, needs_sim)

            para_font, _ = font_cache[cache_key]

            # Wrap text with this font
            wrap_result = wrap_text(
                draw, para_text, para_font, current_column_width, tab_width
            )

            # Store font config in wrapped paragraph
            for wp in wrap_result.paragraphs:
                wp.font_config = config
                wrapped_paragraphs.append(wp)

            if wrap_result.has_overflow:
                has_overflow = True
                break

        # If overflow detected and we can reduce columns, try again
        if has_overflow and num_columns > 1:
            logger.debug(
                "Text overflow detected, reducing columns from %d to %d",
                num_columns,
                num_columns - 1,
            )
            num_columns -= 1
            retry_count += 1
        else:
            # Either no overflow or we're at minimum columns (1)
            break

    # Final column configuration after retry loop
    logger.debug(
        "Text wrapping complete: %d retries, final columns=%d", retry_count, num_columns
    )
    column_width = current_column_width
    total_gap = column_gap * (num_columns - 1)
    block_width = column_width * num_columns + total_gap
    margin_x = max(0, (scaled_image_size[0] - block_width) // 2)
    margin_y = 10  # Small top/bottom margin

    line_height = (
        draw.textbbox((0, 0), "Ag", font=font)[3]
        - draw.textbbox((0, 0), "Ag", font=font)[1]
    )
    line_spacing = int(line_height * 0.2)
    effective_line_height = line_height + line_spacing
    max_lines_per_column = int(
        max(
            1,
            (scaled_image_size[1] - 2 * margin_y - line_height) // effective_line_height
            + 1,
        )
    )
    logger.debug(
        "Layout: line_height=%d, max_lines_per_column=%d",
        line_height,
        max_lines_per_column,
    )

    placements, column_counts = arrange_lines_in_columns(
        wrapped_paragraphs, max_lines_per_column, num_columns
    )
    max_lines_used = max(column_counts) if column_counts else 0
    logger.debug("Columns filled: %s (max=%d)", column_counts, max_lines_used)

    if max_lines_used > 0:
        block_height = max_lines_used * effective_line_height - line_spacing
    else:
        block_height = 0

    if vertical_alignment == "top" or not block_height:
        start_y = 0
    elif vertical_alignment == "bottom":
        start_y = max(0, scaled_image_size[1] - block_height)
    else:
        start_y = max(0, (scaled_image_size[1] - block_height) // 2)

    column_positions = [
        margin_x + c * (column_width + column_gap) for c in range(num_columns)
    ]

    # Create a text mask layer (grayscale: black background, white text for antialiasing)
    text_mask = Image.new("L", scaled_image_size, color=0)
    mask_draw = ImageDraw.Draw(text_mask)

    paragraph_bboxes_map: dict[int, dict] = {}
    actual_text_lines: list[str] = []

    # Track current font and config to avoid redundant lookups
    current_font = font
    current_needs_bold_sim = False
    current_config = None

    for placement in placements:
        actual_text_lines.append(placement.text)
        if not placement.text or placement.is_blank:
            continue

        # Get font config for this line's paragraph
        if placement.paragraph_index is not None:
            para = wrapped_paragraphs[placement.paragraph_index]
            if para.font_config != current_config:
                current_config = para.font_config
                cache_key = current_config.get_cache_key()
                current_font, current_needs_bold_sim = font_cache[cache_key]

        column_x = column_positions[placement.column_index]
        y_position = start_y + placement.line_index * effective_line_height
        bbox = draw.textbbox((0, 0), placement.text, font=current_font)
        line_width = bbox[2] - bbox[0]
        line_height_local = bbox[3] - bbox[1]

        if alignment == "left":
            x_position = column_x
        elif alignment == "right":
            x_position = column_x + max(0, column_width - line_width)
        else:
            x_position = column_x + (max(0, column_width - line_width) // 2)

        x_position_int = int(x_position)
        y_position_int = int(y_position)

        # Draw text on the mask layer (with bold simulation if needed)
        if current_needs_bold_sim:
            # Draw multiple times with slight offsets to simulate bold
            for dx, dy in [(0, 0), (1, 0), (0, 1)]:
                mask_draw.text(
                    (x_position_int + dx, y_position_int + dy),
                    placement.text,
                    fill=255,
                    font=current_font,
                )
        else:
            mask_draw.text(
                (x_position_int, y_position_int),
                placement.text,
                fill=255,  # White for mask
                font=current_font,
            )

        # Draw underline if configured
        if current_config and current_config.underline:
            underline_y = y_position_int + line_height_local + 1
            mask_draw.line(
                [
                    (x_position_int, underline_y),
                    (x_position_int + line_width, underline_y),
                ],
                fill=255,
                width=1,
            )

        paragraph_index = placement.paragraph_index
        if paragraph_index is None:
            continue

        current_bbox = paragraph_bboxes_map.get(paragraph_index)
        line_bbox = [
            x_position_int,
            y_position_int,
            x_position_int + line_width,
            y_position_int + line_height_local,
        ]
        if current_bbox:
            x0 = min(current_bbox["bbox"][0], line_bbox[0])
            y0 = min(current_bbox["bbox"][1], line_bbox[1])
            x1 = max(current_bbox["bbox"][2], line_bbox[2])
            y1 = max(current_bbox["bbox"][3], line_bbox[3])
            current_bbox["bbox"] = [x0, y0, x1, y1]
        else:
            paragraph_bboxes_map[paragraph_index] = {
                "paragraph_text": wrapped_paragraphs[paragraph_index].text,
                "column": placement.column_index,
                "bbox": line_bbox,
            }

    # Apply blending mode to combine text with background
    # Convert font_color to RGB tuple
    font_rgb = color_to_rgb(font_color)

    # Determine the appropriate blend mode
    blend_mode = get_blend_mode(font_color, bg_color)
    logger.debug("Blend mode selected: %s", blend_mode)

    # Convert background to RGB for blending
    background_rgb = image.convert("RGB")

    # Apply displacement mapping if enabled and texture is provided
    if paper_texture_path is not None and apply_displacement:
        logger.debug(
            "Applying displacement mapping with strength=%.2f", displacement_strength
        )
        text_mask = apply_displacement_from_texture(
            text_mask,
            paper_texture_path,
            displacement_strength=displacement_strength,
        )

    # Blend text layer with background using the appropriate mode
    logger.debug("Blending text layer with background")
    blended_image = blend_text_layer(
        background=background_rgb,
        text_mask=text_mask,
        font_color=font_rgb,
        blend_mode=blend_mode,
    )

    # Convert back to RGBA
    image = blended_image.convert("RGBA")

    while actual_text_lines and not actual_text_lines[-1].strip():
        actual_text_lines.pop()

    actual_text = "\n".join(actual_text_lines)

    paragraph_bboxes = [
        {"paragraph_index": idx, **data}
        for idx, data in sorted(paragraph_bboxes_map.items())
    ]

    logger.debug(
        "Image creation complete: %d bounding boxes, %d text lines",
        len(paragraph_bboxes),
        len(actual_text_lines),
    )
    return image, actual_text, paragraph_bboxes
