"""
Script to prepare a dataset with images generated from text data.
Handles text overflow by creating multiple images if necessary.
Saves the new dataset to disk and optionally pushes it to the Hugging Face Hub.

Generating english synthetic OCR dataset as an example:

python scripts/prepare_data.py \
    dataset_path="agentlans/high-quality-english-sentences" \
    text_column="text" \
    data_directory="default" \
    split="train" \
    max_entries=1 \
    max_num_columns=1 \
    max_workers=1 \
    output_path="eng_synthetic_ocr_output_v2" \
    num_examples=1000 \
    push_to_hub=True \
    hub_repo_id="Sigurdur/eng_synthetic_ocr_v2" \
    apply_random_transformations=False \
    font_path="./icelandic_fonts" \

Generating icelandic synthetic OCR dataset as an example:

python scripts/prepare_data.py \
    dataset_path="arnastofnun/IGC-2024" \
    text_column="document" \
    data_directory="parla" \
    split="train" \
    max_entries=1 \
    max_num_columns=1 \
    max_workers=1 \
    output_path="isl_synthetic_ocr_output_v2" \
    num_examples=2000 \
    push_to_hub=True \
    hub_repo_id="Sigurdur/isl_synthetic_ocr_v2" \
    apply_random_transformations=False \
    font_path="./icelandic_fonts" \

"""

import gc
import logging
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

from datasets import (
    Dataset,
    DatasetDict,
    Image,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from fontTools.ttLib import TTFont
from omegaconf import OmegaConf
from PIL import Image as PILImage
from rich.logging import RichHandler
from tqdm import tqdm

from ocr_icelandic.utils import apply_random_transformation, create_image_with_text

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# Suppress fontTools warnings
logging.getLogger("fontTools").setLevel(logging.ERROR)


@dataclass
class DataConfig:
    """Configuration for dataset creation"""

    dataset_path: str = "arnastofnun/IGC-2024"  # Hugging Face dataset path
    text_column: str = "document"  # Column in dataset containing text
    data_directory: str = "parla"  # Subdirectory or config name in the dataset
    split: str = "train"  # Which split to use from the dataset
    max_length: int = 512
    max_entries: int = 400
    show_sample: bool = False  # Whether to show a sample image after creation
    image_width: int = 512
    image_height: int = 512
    image_dpi: int = 140  # Standard set by SmolDocling paper
    img_background_color: str = "white"
    font_path: str = "/usr/share/fonts"
    font_size: int = 12
    min_font_size: int = 11  # Minimum font size when randomizing
    max_font_size: int = 24  # Maximum font size when randomizing
    use_random_font_sizes: bool = (
        True  # Whether to use random font sizes for each sample
    )
    font_color: str = "black"
    use_random_font_colors: bool = True  # Whether to use random font colors
    text_vertical_alignment: str = "center"  # top, middle, bottom
    text_horizontal_alignment: str = "left"  # left, center, right
    output_path: str = "isl_synthetic_ocr_output"  # Directory to save dataset
    num_examples: int = 0  # Number of examples to generate
    push_to_hub: bool = False  # Whether to push dataset to Hugging Face Hub
    hub_repo_id: str = (
        "Sigurdur/isl_synthetic_ocr"  # Hugging Face repo ID to push dataset
    )
    use_random_fonts: bool = True  # Whether to use random fonts
    use_random_backgrounds: bool = True  # Whether to use random background colors
    max_text_length: int = 2000  # Maximum characters per text before splitting
    column_gap: int = 20  # Horizontal gap in pixels between columns
    num_columns: int | None = (
        None  # Number of columns when rendering text (None => random)
    )
    min_num_columns: int = 1  # Minimum number of columns when randomizing
    max_num_columns: int = 5  # Maximum number of columns when randomizing
    column_width: int | None = None  # Fixed column width in pixels (None => random)
    min_column_width: int = 512  # Minimum column width when randomizing
    max_column_width: int = 512  # Maximum column width when randomizing
    apply_random_transformations: bool = True  # Whether to apply random transformations
    max_workers: int = (
        1  # Number of parallel workers for image generation (1 for sequential)
    )
    batch_size: int = 50  # Number of images to hold in memory before flushing to disk


@dataclass
class GenerationConfig(DataConfig):
    """Configuration for image generation only"""

    column_range: tuple[int, int] = (1, 1)
    column_width_range: tuple[int, int] = (100, 512)
    font_size_range: tuple[int, int] = (11, 24)
    available_fonts: tuple[str, ...] | None = None


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
    transformation: dict


def get_random_background_color():
    """Generate a random paper-like background color."""
    # Choose paper type
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

    # Clamp values
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return (r, g, b)


def get_random_font_color(bg_color, contrast_threshold=100):
    """Generate a random font color that contrasts with the background color.
    Font colors are restricted to darker shades closer to black."""

    def luminance(color):
        r, g, b = color
        return 0.299 * r + 0.587 * g + 0.114 * b

    bg_lum = luminance(bg_color)

    while True:
        # Restrict RGB values to 0-80 range for darker colors closer to black
        r = random.randint(0, 80)
        g = random.randint(0, 80)
        b = random.randint(0, 80)
        font_color = (r, g, b)
        font_lum = luminance(font_color)
        if abs(bg_lum - font_lum) >= contrast_threshold:
            return font_color


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


def check_font_supports_char(fontpath: str | Path, unicode_char: str) -> bool:
    """
    Check if a font supports a specific unicode character.
    Args:
        fontpath (str or Path): Path to the font file
        unicode_char (str): The unicode character to check
    Returns:
        bool: True if the font supports the character, False otherwise
    """
    font = TTFont(fontpath)  # specify the path to the font in question

    cmap_table = font.get("cmap")
    if cmap_table is None:
        return False

    for cmap in cmap_table.tables:
        if cmap.isUnicode():
            if ord(unicode_char) in cmap.cmap:
                return True
    return False


@lru_cache(maxsize=1)
def get_icelandic_compatible_fonts(font_path: str | None = None) -> tuple[str, ...]:
    """
    Scan common font directories for fonts that support Icelandic characters.
    Results are cached to avoid rescanning on subsequent calls.
    Returns:
        tuple of str: Paths to fonts that support Icelandic characters
    """
    font_dirs = []

    # load fonts from font directory
    if font_path is not None:
        logger.info("Using provided font path: %s", font_path)
        font_dirs = [font_path]
    else:
        logger.info("No font path provided, scanning common system font directories.")

    random.seed(42)  # For reproducibility

    # Check common font directories based on OS
    current_os = sys.platform

    # Macos
    if current_os.startswith("darwin"):
        font_dirs += [
            "/System/Library/Fonts",
            "/System/Library/Fonts/Supplemental",
        ]
    # Linux
    if current_os.startswith("linux"):
        font_dirs += [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
        ]
    # Windows
    if current_os.startswith("win"):
        font_dirs += [
            str(Path.home() / "AppData/Local/Microsoft/Windows/Fonts"),
            str(Path.home() / "AppData/Roaming/Microsoft/Windows/Fonts"),
            "C:/Windows/Fonts",
        ]

    logger.info("Searching for fonts in directories: %s", font_dirs)

    available_fonts: list[str] = []
    characters_to_check = "ÁáÐðÉéÍíÓóÚúÝýÞþÆæÖö"
    for font_dir in tqdm(font_dirs, desc="Scanning font directories"):
        font_path = Path(font_dir)
        if font_path.exists() and font_path.is_dir():
            for font_file in font_path.rglob("*.[tT][tT][fF]"):
                # Check if font supports ALL required characters
                if all(
                    check_font_supports_char(font_file, char)
                    for char in characters_to_check
                ):
                    available_fonts.append(str(font_file))

    logger.info("Found %d Icelandic-compatible fonts.", len(available_fonts))

    return tuple(available_fonts)


def _normalize_range(
    min_value: int, max_value: int, minimum: int = 1
) -> tuple[int, int]:
    """
    Ensure provided min/max values form a valid range.

    Args:
        min_value (int): Minimum value
        max_value (int): Maximum value
        minimum (int): Minimum allowed value for min_value

    Returns:
        tuple[int, int]: Normalized (min_value, max_value)
    """

    min_value = max(minimum, min_value)
    max_value = max(min_value, max_value)
    return min_value, max_value


def generate_single_text(
    text: str, cfg: GenerationConfig
) -> tuple[list[SingleImageData], int]:
    """
    Generate images for a single text entry, handling overflow by creating multiple images.

    Args:
        text (str): The text to convert to images
        cfg (GenerationConfig): Configuration for image generation
    Returns:
        tuple: (List of SingleImageData, number of splits)
    """

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

            # Select random background color if enabled
            current_bg_color = bg_color
            if cfg.use_random_backgrounds:
                current_bg_color = get_random_background_color()

            # Select random font color if enabled
            if cfg.use_random_font_colors:
                font_color = get_random_font_color(current_bg_color)

            # Select random font size if enabled
            current_font_size = font_size
            if cfg.use_random_font_sizes:
                current_font_size = random.randint(*cfg.font_size_range)

            if cfg.num_columns is not None and cfg.num_columns > 0:
                num_columns = cfg.num_columns
            else:
                num_columns = random.randint(*column_range)

            if cfg.column_width is not None and cfg.column_width > 0:
                column_width = cfg.column_width
            else:
                # column_width = random.randint(*column_width_range)

                # take into account number of columns and gaps
                total_gap = (num_columns - 1) * cfg.column_gap
                max_column_width = (width - total_gap) // num_columns
                min_column_width = min(column_width_range[0], max_column_width)
                column_width = random.randint(min_column_width, max_column_width)

            image, fitted_text, paragraph_bboxes = create_image_with_text(
                remaining_text,
                image_size=(width, height),
                alignment=alignment,
                font_size=current_font_size,
                font_path=current_font_path,
                bg_color=current_bg_color,
                font_color=font_color,
                vertical_alignment=vertical_alignment,
                dpi=dpi,
                num_columns=num_columns,
                column_gap=cfg.column_gap,
                column_width=column_width,
            )

            if not fitted_text:
                # No text could be fitted, break to avoid infinite loop
                break

            if cfg.apply_random_transformations:
                transformed_image, transformation_meta, transformed_paragraph_bboxes = (
                    apply_random_transformation(
                        image,
                        current_bg_color,
                        paragraph_bboxes=paragraph_bboxes,
                    )
                )
            else:
                transformed_image = image
                transformation_meta = {"type": "none"}
                transformed_paragraph_bboxes = paragraph_bboxes

            images.append(
                SingleImageData(
                    text=fitted_text,
                    image=transformed_image,
                    font_path=current_font_path,
                    bg_color=current_bg_color,
                    font_color=font_color,
                    font_size=current_font_size,
                    image_width=width,
                    image_height=height,
                    image_dpi=dpi,
                    text_vertical_alignment=vertical_alignment,
                    text_horizontal_alignment=alignment,
                    paragraph_bboxes=transformed_paragraph_bboxes,
                    transformation=transformation_meta,
                )
            )

            # Update remaining text
            # This assumes create_image_with_text preserves original whitespace
            # and returns a prefix of the input text.
            remaining_text = remaining_text[len(fitted_text) :].lstrip()

    return images, len(text_chunks)


def _save_batch_to_disk(
    new_data: dict,
    batch_dir: Path,
    batch_idx: int,
) -> str:
    """
    Save a batch of images to disk and return the path.

    Args:
        new_data: Dictionary containing the batch data
        batch_dir: Directory to save batches
        batch_idx: Index of the current batch

    Returns:
        str: Path to the saved batch
    """
    batch_path = batch_dir / f"batch_{batch_idx:04d}"
    batch_ds = Dataset.from_dict(dict(new_data)).cast_column("image", Image())
    batch_ds.save_to_disk(str(batch_path))
    return str(batch_path)


def generate_image_dataset(texts: list[str], cfg: DataConfig) -> Dataset:
    """
    Generates a new dataset with images and corresponding text,
    handling text overflow by creating multiple images.
    Uses batch processing to avoid OOM by flushing to disk periodically.

    Args:
        texts (list of str): List of text entries to convert to images
        cfg (DataConfig): Configuration for image generation
    Returns:
        Dataset: A Hugging Face Dataset with 'text' and 'image' columns
    """

    logger.info("Generating images from text...")

    # fix number of examples to generate if specified
    num_examples = cfg.num_examples if cfg.num_examples > 0 else len(texts)

    available_fonts = None
    if cfg.use_random_fonts:
        available_fonts = get_icelandic_compatible_fonts(cfg.font_path)

    column_range = _normalize_range(cfg.min_num_columns, cfg.max_num_columns, minimum=1)
    column_width_range = _normalize_range(
        cfg.min_column_width, cfg.max_column_width, minimum=1
    )
    font_size_range = _normalize_range(cfg.min_font_size, cfg.max_font_size, minimum=1)

    generation_cfg = GenerationConfig(
        **asdict(cfg),
        available_fonts=available_fonts,
        column_range=column_range,
        column_width_range=column_width_range,
        font_size_range=font_size_range,
    )

    # Batch processing setup to avoid OOM
    batch_size = cfg.batch_size
    batch_dir = Path(cfg.output_path) / "_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_datasets: list[str] = []

    new_data: defaultdict[str, list] = defaultdict(list)
    total_splits = 0
    split_texts = 0
    batch_idx = 0

    # Use ProcessPoolExecutor for true parallel processing (bypass GIL)
    # Use configured max_workers (default 1 for sequential execution to avoid OOM)
    max_workers = min(cfg.max_workers, len(texts[:num_examples]))
    logger.info(
        "Using %d parallel workers for image generation (batch_size=%d).",
        max_workers,
        batch_size,
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
                    new_data["transformation"].append(image_data.transformation)

                # Flush batch to disk when threshold reached to avoid OOM
                if len(new_data["image"]) >= batch_size:
                    batch_path = _save_batch_to_disk(new_data, batch_dir, batch_idx)
                    batch_datasets.append(batch_path)
                    logger.info(
                        "Saved batch %d with %d images to disk",
                        batch_idx,
                        len(new_data["image"]),
                    )
                    batch_idx += 1
                    new_data = defaultdict(list)
                    gc.collect()  # Force garbage collection to free memory

            except (OSError, ValueError, RuntimeError) as e:
                logger.error("Error processing text: %s", e)
                continue

    # Save any remaining data in the final batch
    if new_data["image"]:
        batch_path = _save_batch_to_disk(new_data, batch_dir, batch_idx)
        batch_datasets.append(batch_path)
        logger.info(
            "Saved final batch %d with %d images to disk",
            batch_idx,
            len(new_data["image"]),
        )
        del new_data
        gc.collect()

    logger.info(
        "Split %d long texts into multiple chunks, generating %d batches.",
        split_texts,
        len(batch_datasets),
    )

    # Concatenate all batches into final dataset
    if not batch_datasets:
        logger.warning("No images were generated.")
        return Dataset.from_dict({cfg.text_column: [], "image": []}).cast_column(
            "image", Image()
        )

    logger.info("Concatenating %d batches into final dataset...", len(batch_datasets))
    all_datasets = [load_from_disk(path) for path in batch_datasets]
    final_dataset = concatenate_datasets(all_datasets)

    # Clean up batch files
    logger.info("Cleaning up temporary batch files...")
    shutil.rmtree(batch_dir)

    return final_dataset


def display_sample(dataset: dict) -> None:
    """
    Display the first generated image from the dataset.

    Args:
        dataset (dict): The dataset containing splits
    Returns:
        None
    """
    logger.info("\nShowing first generated image...")
    if len(dataset["train"]) > 0:
        logger.info("Text for first image:")
        logger.info("'%s'", dataset["train"][0]["text"])
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
        logger.info("Renaming text column '%s' to 'text'", cfg.text_column)
        dataset = dataset.rename_column(cfg.text_column, "text")
        cfg.text_column = "text"

    # Create a new dataset with an 'image' column for each text
    image_dataset = generate_image_dataset(texts, cfg)

    logger.info("\nOriginal dataset size: %d", len(texts))
    logger.info("New image dataset size: %d", len(image_dataset))

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
    dataset_dict.save_to_disk(output_path)
    logger.info("Image dataset saved to %s", output_path)

    # Display the first image as an example
    if cfg.show_sample:
        display_sample(final_dataset)

    # upload to huggingface dataset hub
    if cfg.push_to_hub and cfg.hub_repo_id:
        logger.info("Pushing dataset to the hub at %s...", cfg.hub_repo_id)
        dataset_dict.push_to_hub(cfg.hub_repo_id)
        logger.info("Dataset pushed to the hub successfully.")


def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(DataConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg_dict = cast(dict[str, any], OmegaConf.to_container(cfg, resolve=True))
    try:
        cfg = DataConfig(**cfg_dict)
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error("Error: %s\n\nUsage: python scratch.py", e)
        sys.exit(1)

    create_image_dataset(cfg)


if __name__ == "__main__":
    main()
