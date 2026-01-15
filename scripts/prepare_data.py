"""
Script to prepare a dataset with images generated from text data.
Handles text overflow by creating multiple images if necessary.
Saves the new dataset to disk and optionally pushes it to the Hugging Face Hub.
"""

from collections import defaultdict
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from datasets import Dataset, DatasetDict, Image as DatasetImage, load_dataset
from PIL import Image
import psutil
from ocr_icelandic.fonts import (
    get_compatible_fonts,
    sync_google_fonts,
)
from ocr_icelandic.transformations import apply_random_transformation
from ocr_icelandic.utils import (
    _visualise_bboxes,
    apply_background_image,
    create_image_with_text,
    discover_backgrounds,
    discover_paper_textures,
)
from omegaconf import OmegaConf
from tqdm import tqdm
from rich.logging import RichHandler
from PIL import Image as PILImage
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import ImageColor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset creation"""

    dataset_path: str = "arnastofnun/IGC-2024"  # Hugging Face dataset path
    text_column: str = "document"  # Column in dataset containing text
    data_directory: str = "parla"  # Subdirectory or config name in the dataset
    split: str = "train"  # Which split to use from the dataset
    max_length: int = 512
    max_entries: int = 2
    show_sample: bool = False  # Whether to show a sample image after creation
    image_width: int = 512
    image_height: int = 512
    image_dpi: int = 72
    img_background_color: str = "white"
    font_path: str = "/usr/share/fonts"
    font_size: int = 12
    font_color: str = "black"
    use_random_font_colors: bool = True  # Whether to use random font colors
    text_vertical_alignment: str = "center"  # top, middle, bottom
    text_horizontal_alignment: str = "left"  # left, center, right
    output_path: str = "isl_synthetic_ocr_output"  # Directory to save dataset
    num_examples: int = 2  # Number of examples to generate
    push_to_hub: bool = False  # Whether to push dataset to Hugging Face Hub
    save_to_disk: bool = False  # Whether to save dataset to disk
    hub_repo_id: str = (
        "Sigurdur/isl_synthetic_ocr"  # Hugging Face repo ID to push dataset
    )
    use_random_fonts: bool = True  # Whether to use random fonts
    use_random_backgrounds: bool = True  # Whether to use random background colors
    google_fonts_directory: str = "../google_fonts"  # Directory to store Google Fonts
    language_code: str = (
        "is"  # ISO 639-1 language code (e.g., "is" for Icelandic, "de" for German)
    )
    use_font_cache: bool = True  # Whether to use SQLite caching for font compatibility
    font_cache_dir: str = ".fontcache"  # Directory to store font compatibility cache
    use_paper_textures: bool = True  # Whether to use paper textures from assets/papers
    paper_textures_dir: str = (
        "assets/papers"  # Directory containing paper texture images
    )
    use_background_images: bool = True  # Whether to use background images
    backgrounds_dir: str = (
        "assets/backgrounds"  # Directory containing background images
    )
    background_image_probability: float = 1  # Probability of using a background image
    max_text_length: int = 2000  # Maximum characters per text before splitting
    column_gap: int = 20  # Horizontal gap in pixels between columns
    num_columns: int | None = (
        None  # Number of columns when rendering text (None => random)
    )
    min_num_columns: int = 1  # Minimum number of columns when randomizing
    max_num_columns: int = 5  # Maximum number of columns when randomizing
    column_width: int | None = None  # Fixed column width in pixels (None => random)
    min_column_width: int = 100  # Minimum column width when randomizing
    max_column_width: int = 512  # Maximum column width when randomizing
    local_output_dir: str = "./local_output"  # Local directory for temporary outputs
    # Font variation settings
    enable_font_size_variation: bool = False  # Whether to vary font size per paragraph
    font_size_min_ratio: float = (
        0.8  # Minimum font size ratio (e.g., 0.8 = 80% of base)
    )
    font_size_max_ratio: float = (
        1.2  # Maximum font size ratio (e.g., 1.2 = 120% of base)
    )
    enable_font_styles: bool = False  # Whether to apply font styles (bold/underline)
    font_bold_probability: float = 0.2  # Probability of applying bold style
    font_underline_probability: float = 0.1  # Probability of applying underline style


@dataclass
class GenerationConfig(DataConfig):
    """Configuration for image generation only"""

    column_range: tuple[int, int] = (1, 1)
    column_width_range: tuple[int, int] = (100, 512)
    available_fonts: list[str] | None = None
    available_paper_textures: list[str] | None = None
    available_no_shadow_backgrounds: list[str] | None = None
    available_with_shadow_backgrounds: list[str] | None = None


@dataclass
class SingleImageData:
    """Data for a single generated image"""

    text: str
    image: PILImage.Image
    font_path: str
    bg_color: tuple[int, int, int] | str
    font_color: tuple[int, int, int] | str
    font_size: int
    image_width: int
    image_height: int
    image_dpi: int
    text_vertical_alignment: str
    text_horizontal_alignment: str
    paragraph_bboxes: list[dict]
    transformations: list[dict]
    # NEW FIELDS for font variation
    paragraph_font_sizes: list[int] | None = None  # Font size per paragraph
    paragraph_styles: list[dict] | None = None  # Style flags per paragraph


def get_random_background_color():
    """
    Generate a random background color with weighted distribution.

    Distribution: 85% light (paper-like), 10% dark, 5% colorful

    Returns:
        Tuple[int, int, int]: RGB color tuple
    """
    # Weighted random selection: 85% light, 10% dark, 5% colorful
    rand_val = random.random()

    if rand_val < 0.85:
        # Light colors (paper-like) - 85% probability
        paper_type = random.choice(["white", "cream", "aged"])

        if paper_type == "white":
            base = random.randint(245, 252)
            r = base + random.randint(-3, 3)
            g = base + random.randint(-5, 0)
            b = base + random.randint(-8, 0)
        elif paper_type == "cream":
            base = random.randint(235, 245)
            r = base + random.randint(0, 8)
            g = base + random.randint(-5, 3)
            b = base + random.randint(-12, -3)
        else:  # aged
            base = random.randint(220, 235)
            r = base + random.randint(5, 15)
            g = base + random.randint(0, 10)
            b = base + random.randint(-15, -5)

    elif rand_val < 0.95:
        # Dark colors - 10% probability
        base = random.randint(20, 80)
        r = base + random.randint(-10, 10)
        g = base + random.randint(-10, 10)
        b = base + random.randint(-10, 10)

    else:
        # Colorful - 5% probability
        # At least one channel bright (>150), others varied
        bright_channel = random.randint(0, 2)
        colors = [0, 0, 0]
        colors[bright_channel] = random.randint(150, 255)

        # Other channels can be varied
        for i in range(3):
            if i != bright_channel:
                colors[i] = random.randint(30, 220)

        r, g, b = colors

    # Clamp values to valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return (r, g, b)


def get_random_font_color(
    bg_color: tuple[int, int, int] | str, contrast_threshold: float = 3.5
) -> tuple[int, int, int]:
    """Generate a random font color that contrasts with the background color.

    Uses WCAG 2.1 contrast ratio guidelines for better readability.

    Args:
        bg_color: Background color as RGB tuple or color name string
        contrast_threshold: Minimum contrast ratio (WCAG recommends 4.5 for normal text, we use 3.5 here to make it more challenging)

    Returns:
        RGB tuple representing the font color
    """

    def luminance(color: tuple[int, int, int]) -> float:
        """Calculate relative luminance per WCAG 2.1 specification."""
        r, g, b = color
        # Normalize to 0-1 range
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        # Apply gamma correction
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast_ratio(lum1: float, lum2: float) -> float:
        """Calculate contrast ratio between two luminance values."""
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        return (lighter + 0.05) / (darker + 0.05)

    # Convert bg_color to RGB tuple if it's a string
    if isinstance(bg_color, str):
        bg_color = ImageColor.getrgb(bg_color)

    bg_lum = luminance(bg_color)

    # Try a few common high-contrast options first
    candidates = [(0, 0, 0), (255, 255, 255), (50, 50, 50), (230, 230, 230)]
    for font_color in candidates:
        font_lum = luminance(font_color)
        if contrast_ratio(bg_lum, font_lum) >= contrast_threshold:
            return font_color

    # Fall back to random generation with timeout
    max_attempts = 100
    for _ in range(max_attempts):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        font_color = (r, g, b)
        font_lum = luminance(font_color)
        if contrast_ratio(bg_lum, font_lum) >= contrast_threshold:
            return font_color

    # If no suitable color found, return black or white based on background luminance
    return (0, 0, 0) if bg_lum > 0.5 else (255, 255, 255)


def split_long_text(text: str, max_length: int) -> list[str]:
    """
    Split text into chunks at sentence boundaries to avoid mid-sentence splits.

    Args:
        text: The text to split
        max_length: Maximum length for each chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    # Split on sentence boundaries
    sentences = (
        text.replace("! ", "!|").replace("? ", "?|").replace(". ", ".|").split("|")
    )

    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence exceeds max_length, save current chunk and start new one
        if len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


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
    styles = []
    for _ in range(num_paragraphs):
        styles.append(
            {
                "bold": random.random() < bold_probability,
                "underline": random.random() < underline_probability,
            }
        )
    return styles


def _normalize_range(
    min_value: int, max_value: int, minimum: int = 1
) -> tuple[int, int]:
    """Ensure provided min/max values form a valid range."""

    min_value = max(minimum, min_value)
    max_value = max(min_value, max_value)
    return min_value, max_value


def generate_single_text(
    text: str, cfg: GenerationConfig
) -> tuple[list[SingleImageData], int]:
    # Split long texts first
    text_chunks = split_long_text(text.strip(), cfg.max_text_length)

    # Get settings from config
    width = cfg.image_width
    height = cfg.image_height
    dpi = cfg.image_dpi
    font_size = cfg.font_size
    alignment = cfg.text_horizontal_alignment
    font_path = cfg.font_path
    bg_color = cfg.img_background_color
    font_color = cfg.font_color
    vertical_alignment = cfg.text_vertical_alignment

    available_fonts = cfg.available_fonts
    column_range = cfg.column_range
    column_width_range = cfg.column_width_range

    images: list[SingleImageData] = []
    for chunk in text_chunks:
        remaining_text = chunk
        while remaining_text:
            # Select random font if enabled
            current_font_path = font_path
            if cfg.use_random_fonts and available_fonts:
                current_font_path = random.choice(available_fonts)

            # Select random background colors if enabled
            # paper_bg_color: for initial text rendering
            # composite_bg_color: for final RGB composite
            current_bg_color = bg_color
            composite_bg_color = bg_color
            if cfg.use_random_backgrounds:
                current_bg_color = get_random_background_color()
                # Generate a separate color for final composite
                composite_bg_color = get_random_background_color()

            # Select random paper texture if enabled
            paper_texture_path = None
            if cfg.use_paper_textures and cfg.available_paper_textures:
                paper_texture_path = random.choice(cfg.available_paper_textures)

            # Select random font color if enabled
            if cfg.use_random_font_colors:
                font_color = get_random_font_color(current_bg_color)

            if cfg.num_columns is not None and cfg.num_columns > 0:
                num_columns = cfg.num_columns
            else:
                num_columns = random.randint(*column_range)

            if cfg.column_width is not None and cfg.column_width > 0:
                column_width = cfg.column_width
            else:
                column_width = random.randint(*column_width_range)

            # Calculate paragraph font sizes and styles if enabled
            paragraph_font_configs = None
            paragraph_font_sizes = None
            paragraph_styles = None

            if cfg.enable_font_size_variation or cfg.enable_font_styles:
                from ocr_icelandic.utils.text_layout import (
                    ParagraphFontConfig,
                    calculate_paragraph_font_sizes,
                )

                paragraphs = remaining_text.split("\n\n")
                num_paragraphs = len(paragraphs)

                # Calculate font sizes
                if cfg.enable_font_size_variation:
                    paragraph_font_sizes = calculate_paragraph_font_sizes(
                        paragraphs,
                        font_size,
                        cfg.font_size_min_ratio,
                        cfg.font_size_max_ratio,
                    )
                else:
                    paragraph_font_sizes = [font_size] * num_paragraphs

                # Generate styles
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

                # Build ParagraphFontConfig objects
                paragraph_font_configs = [
                    ParagraphFontConfig(
                        font_path=current_font_path,
                        font_size=paragraph_font_sizes[i],
                        bold=paragraph_styles[i]["bold"],
                        underline=paragraph_styles[i]["underline"],
                    )
                    for i in range(num_paragraphs)
                ]

            image, fitted_text, paragraph_bboxes = create_image_with_text(
                remaining_text,
                image_size=(width, height),
                alignment=alignment,
                font_size=font_size,
                font_path=current_font_path,
                bg_color=current_bg_color,
                font_color=font_color,
                vertical_alignment=vertical_alignment,
                dpi=dpi,
                num_columns=num_columns,
                column_gap=cfg.column_gap,
                column_width=column_width,
                paper_texture_path=paper_texture_path,
                apply_displacement=True,
                paragraph_font_configs=paragraph_font_configs,
            )

            # Decide whether to use a background image
            use_background = False
            background_has_shadow = True
            background_path = None
            background_image = None

            if cfg.use_background_images and (
                cfg.available_no_shadow_backgrounds
                or cfg.available_with_shadow_backgrounds
            ):
                # Use background based on probability
                if random.random() < cfg.background_image_probability:
                    use_background = True
                    # Choose background type (with/without shadow)
                    all_backgrounds = []
                    if cfg.available_with_shadow_backgrounds:
                        all_backgrounds.extend(
                            [(bg, True) for bg in cfg.available_with_shadow_backgrounds]
                        )
                    if cfg.available_no_shadow_backgrounds:
                        all_backgrounds.extend(
                            [(bg, False) for bg in cfg.available_no_shadow_backgrounds]
                        )

                    if all_backgrounds:
                        background_path, background_has_shadow = random.choice(
                            all_backgrounds
                        )

                        # Load and pre-expand background for transformations
                        try:
                            background_image = Image.open(background_path).convert(
                                "RGBA"
                            )
                            # Pre-expand background to ensure full coverage after transforms
                            # Use 1.8x expansion factor as conservative estimate
                            expansion_factor = 1.8
                            expanded_width = int(width * expansion_factor)
                            expanded_height = int(height * expansion_factor)

                            # Resize or tile background to fill expanded size
                            bg_width, bg_height = background_image.size
                            if bg_width < expanded_width or bg_height < expanded_height:
                                # Tile background
                                tiles_x = (expanded_width // bg_width) + 2
                                tiles_y = (expanded_height // bg_height) + 2
                                tiled = Image.new(
                                    "RGBA", (bg_width * tiles_x, bg_height * tiles_y)
                                )
                                for i in range(tiles_x):
                                    for j in range(tiles_y):
                                        tiled.paste(
                                            background_image,
                                            (i * bg_width, j * bg_height),
                                        )
                                # Crop from center
                                left = (tiled.width - expanded_width) // 2
                                top = (tiled.height - expanded_height) // 2
                                background_image = tiled.crop(
                                    (
                                        left,
                                        top,
                                        left + expanded_width,
                                        top + expanded_height,
                                    )
                                )
                            else:
                                # Resize to expanded size
                                background_image = background_image.resize(
                                    (expanded_width, expanded_height),
                                    Image.Resampling.BICUBIC,
                                )
                        except Exception as e:
                            print(
                                f"Warning: Failed to load background {background_path}: {e}"
                            )
                            background_image = None
                            use_background = False

            # Apply transformations with the appropriate pipeline
            (
                transformed_image,
                transformation_meta,
                transformed_paragraph_bboxes,
                transformed_background,
            ) = apply_random_transformation(
                image,
                current_bg_color,
                paragraph_bboxes=paragraph_bboxes,
                use_background=use_background,
                background_has_shadow=background_has_shadow,
                background_image=background_image,
            )

            # Apply background image if selected and transformed background is available
            if use_background and transformed_background is not None:
                transformed_image, bg_meta, transformed_paragraph_bboxes = (
                    apply_background_image(
                        transformed_image,
                        transformed_background,  # Pass transformed Image, not path
                        paragraph_bboxes=transformed_paragraph_bboxes,
                    )
                )
                # Add background metadata to transformations
                transformation_meta.append({"transformation": "background", **bg_meta})

            # Final composite: Convert RGBA to RGB by pasting on a new background
            # This happens after all transformations and background application
            if transformed_image.mode == "RGBA":
                # Create RGB background with the composite color
                rgb_background = Image.new(
                    "RGB", transformed_image.size, composite_bg_color
                )
                # Paste RGBA image using its alpha channel as mask
                rgb_background.paste(transformed_image, (0, 0), transformed_image)
                transformed_image = rgb_background
            elif transformed_image.mode != "RGB":
                # Fallback: convert to RGB
                transformed_image = transformed_image.convert("RGB")

            transformed_image = _visualise_bboxes(
                transformed_image, transformed_paragraph_bboxes, show_labels=False
            )

            if not fitted_text:
                # No text could be fitted, break to avoid infinite loop
                break

            images.append(
                SingleImageData(
                    text=fitted_text,
                    image=transformed_image,
                    font_path=current_font_path,
                    bg_color=current_bg_color,
                    font_color=font_color,
                    font_size=font_size,
                    image_width=width,
                    image_height=height,
                    image_dpi=dpi,
                    text_vertical_alignment=vertical_alignment,
                    text_horizontal_alignment=alignment,
                    paragraph_bboxes=transformed_paragraph_bboxes,
                    transformations=transformation_meta,
                    paragraph_font_sizes=paragraph_font_sizes,
                    paragraph_styles=paragraph_styles,
                )
            )

            # Update remaining text
            # This assumes create_image_with_text preserves original whitespace
            # and returns a prefix of the input text.
            remaining_text = remaining_text[len(fitted_text) :].lstrip()

    return images, len(text_chunks)


def generate_image_dataset(texts: list[str], cfg: DataConfig) -> Dataset:
    """
    Generates a new dataset with images and corresponding text,
    handling text overflow by creating multiple images.
    Args:
        texts (list of str): List of text entries to convert to images
        cfg (DataConfig): Configuration for image generation
    Returns:
        Dataset: A Hugging Face Dataset with 'text' and 'image' columns
    """

    logger.info("Generating images from text...")

    # fix number of examples to generate if specified
    num_examples = cfg.num_examples if cfg.num_examples > 0 else len(texts)

    # Check for Google Fonts API key and sync fonts if available
    google_fonts_api_key = os.environ.get("GOOGLE_FONTS_API_KEY")
    if google_fonts_api_key:
        logger.info("GOOGLE_FONTS_API_KEY found, syncing Google Fonts...")
        sync_google_fonts(google_fonts_api_key, cfg.google_fonts_directory)
    else:
        logger.warning(
            "GOOGLE_FONTS_API_KEY environment variable not set. "
            "Skipping Google Fonts sync and using only system fonts."
        )

    available_fonts = None
    if cfg.use_random_fonts:
        available_fonts = get_compatible_fonts(
            language_code=cfg.language_code,
            use_cache=cfg.use_font_cache,
            cache_dir=cfg.font_cache_dir,
            google_fonts_directory=cfg.google_fonts_directory,
        )

    available_paper_textures = None
    if cfg.use_paper_textures:
        available_paper_textures = discover_paper_textures(cfg.paper_textures_dir)
        if available_paper_textures:
            logger.info(
                f"Found {len(available_paper_textures)} paper textures in {cfg.paper_textures_dir}"
            )
        else:
            logger.warning(
                f"No paper textures found in {cfg.paper_textures_dir}, falling back to solid colors"
            )

    available_no_shadow_backgrounds = []
    available_with_shadow_backgrounds = []
    if cfg.use_background_images:
        available_no_shadow_backgrounds, available_with_shadow_backgrounds = (
            discover_backgrounds(cfg.backgrounds_dir)
        )
        logger.info(
            f"Found {len(available_no_shadow_backgrounds)} no-shadow backgrounds and {len(available_with_shadow_backgrounds)} with-shadow backgrounds"
        )
        if (
            not available_no_shadow_backgrounds
            and not available_with_shadow_backgrounds
        ):
            logger.warning(
                f"No backgrounds found in {cfg.backgrounds_dir}, will not use background images"
            )

    column_range = _normalize_range(cfg.min_num_columns, cfg.max_num_columns, minimum=1)
    column_width_range = _normalize_range(
        cfg.min_column_width, cfg.max_column_width, minimum=1
    )

    generation_cfg = GenerationConfig(
        **asdict(cfg),
        available_fonts=available_fonts,
        available_paper_textures=available_paper_textures,
        available_no_shadow_backgrounds=available_no_shadow_backgrounds,
        available_with_shadow_backgrounds=available_with_shadow_backgrounds,
        column_range=column_range,
        column_width_range=column_width_range,
    )

    new_data: defaultdict[str, list] = defaultdict(list)
    total_splits = 0
    split_texts = 0

    # Use ProcessPoolExecutor for true parallel processing (bypass GIL)
    # Use physical cores for CPU-bound tasks
    max_workers = min(psutil.cpu_count(logical=False) or 4, len(texts[:num_examples]))
    logger.info(
        f"Using {max_workers} parallel workers (physical cores) for image generation."
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_text = {
            executor.submit(generate_single_text, text, generation_cfg): text
            for text in texts[:num_examples]
        }

        # Process completed tasks with progress bar
        for future in tqdm(
            as_completed(future_to_text),
            total=len(future_to_text),
            desc="Processing texts",
            unit="text",
        ):
            try:
                image_data_list, num_splits = future.result()
                total_splits += num_splits
                split_texts += 1 if num_splits > 1 else 0

                for image_data in image_data_list:
                    new_data[cfg.text_column].append(image_data.text)
                    new_data["image"].append(image_data.image)
                    new_data["font_path"].append(image_data.font_path)
                    new_data["bg_color"].append(image_data.bg_color)
                    new_data["font_color"].append(image_data.font_color)
                    new_data["font_size"].append(image_data.font_size)
                    new_data["image_width"].append(image_data.image_width)
                    new_data["image_height"].append(image_data.image_height)
                    new_data["image_dpi"].append(image_data.image_dpi)
                    new_data["text_vertical_alignment"].append(
                        image_data.text_vertical_alignment
                    )
                    new_data["text_horizontal_alignment"].append(
                        image_data.text_horizontal_alignment
                    )
                    new_data["paragraph_bboxes"].append(image_data.paragraph_bboxes)
                    new_data["transformations"].append(image_data.transformations)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                continue

    logger.info(
        f"Split {split_texts} long texts into multiple chunks, in total generating {total_splits} images from {len(texts)}."
    )

    # Create a new Hugging Face Dataset
    image_dataset = Dataset.from_dict(new_data).cast_column("image", DatasetImage())
    return image_dataset


def display_sample(dataset: dict) -> None:
    logger.info("\nShowing first generated image...")
    if len(dataset["train"]) > 0:
        logger.info("Text for first image:")
        logger.info(f"'{dataset['train'][0]['text']}'")
        dataset["train"][0]["image"].show()


def create_image_dataset(cfg: DataConfig) -> None:
    """
    Create a dataset with images generated from text data.
    Args:
        cfg (DataConfig): Configuration for dataset creation
    """
    # load dataset
    dataset = cast(
        Dataset,
        load_dataset(
            cfg.dataset_path,
            cfg.data_directory if hasattr(cfg, "data_directory") else None,
            split=cfg.split,
        ),
    )

    # select number of entries if specified
    if cfg.max_entries > 0:
        dataset = dataset.select(range(cfg.max_entries))

    texts = list(dataset[cfg.text_column])

    # rename text column to 'text' if necessary
    if cfg.text_column != "text":
        logger.info(f"Renaming text column '{cfg.text_column}' to 'text'")
        dataset = dataset.rename_column(cfg.text_column, "text")
        cfg.text_column = "text"

    # Create a new dataset with an 'image' column for each text
    image_dataset = generate_image_dataset(texts, cfg)

    logger.info(f"\nOriginal dataset size: {len(texts)}")
    logger.info(f"New image dataset size: {len(image_dataset)}")

    # Create a train/test/validation split (80/10/10)
    split_dataset = image_dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
    final_dataset = {
        "train": split_dataset["train"],
        "test": test_valid["test"],
        "validation": test_valid["train"],
    }

    # Save the new dataset
    output_path = cfg.output_path
    # Use DatasetDict for saving splits

    dataset_dict = DatasetDict(list(final_dataset.items()))
    if cfg.save_to_disk:
        dataset_dict.save_to_disk(output_path)
    logger.info(f"Image dataset saved to {output_path}")

    # Display the first image as an example
    if cfg.show_sample:
        display_sample(final_dataset)

    # upload to huggingface dataset hub
    if cfg.push_to_hub and cfg.hub_repo_id:
        logger.info(f"Pushing dataset to the hub at {cfg.hub_repo_id}...")
        dataset_dict.push_to_hub(cfg.hub_repo_id)
        logger.info("Dataset pushed to the hub successfully.")

    if cfg.local_output_dir:
        local_output_path = Path(cfg.local_output_dir)
        local_output_path.mkdir(parents=True, exist_ok=True)

        for split, items in final_dataset.items():
            split_path = local_output_path / split

            split_path.mkdir(parents=True, exist_ok=True)
            for idx, item in enumerate(items):
                image: PILImage.Image = item["image"]
                image_save_path = split_path / f"image_{idx:05d}.png"
                image.save(image_save_path)

        logger.info(f"Image dataset also saved to local directory {local_output_path}")


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(DataConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = DataConfig(**cfg)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python scratch.py")
        sys.exit(1)

    create_image_dataset(cfg)


if __name__ == "__main__":
    main()
